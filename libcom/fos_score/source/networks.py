import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import copy
from .loss_function import TripletLoss
from torchvision.ops import RoIAlign, roi_align

class SingleScaleD(nn.Module):
    def __init__(self, loadweights=False):
        super().__init__()
        model = models.vgg19(pretrained = loadweights)
        self.backbone = model.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f3 = self.backbone[:19](x)
        f4 = self.backbone[19:28](f3)
        f5 = self.backbone[28:](f4)
        fv = self.gap(f5).flatten(1)
        out = self.sigmoid(self.fc(fv)).flatten()
        return [f5, fv], out

def conv_relu_x2(in_ch, out_ch):
    convs = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        nn.ReLU())
    return convs

def binary_classifier(in_ch):
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_ch, 1, 1, 1),
        nn.Sigmoid())
    return classifier

def get_distillation_module(distill_type, in_ch, out_ch, alignsize, downsample):
    if distill_type == 'roiconcat':
        return RoIConcatDistillation(in_ch, out_ch, alignsize, downsample)
    elif distill_type == 'roicropresize':
        return CropResizeDistillation(in_ch, out_ch, alignsize, downsample)
    elif distill_type == 'gloablconcat':
        return GlobalConcatDistillation(in_ch, out_ch)
    elif distill_type == 'roivectorconcat':
        return RoIVectorConcatDistillation(in_ch, out_ch, downsample)
    elif distill_type in ['gapconcat', 'onlyclassify']:
        return VectorConcatDistillation(in_ch, out_ch)
    else:
        return None

class CropResizeDistillation(nn.Module):
    def __init__(self, in_ch, out_ch, alignsize, downsample):
        super().__init__()
        # self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoIAlign = RoIAlign(alignsize, 1.0 / 2 ** downsample, -1)
        self.convs = conv_relu_x2(in_ch, out_ch)
        # self.scale = 1.0 / 2 ** downsample
        self.alignsize = alignsize
        self.classifier = binary_classifier(out_ch)

    def forward(self, bg_feat, fg_feat, q_box, c_box, num_bg):
        '''
        :param bg_feat: b,c,h,w
        :param fg_feat: b,c,h,w
        :param q_box: query box, b,4
        :param c_box: cropped bounding box, b,4
        :return:
        '''
        B = bg_feat.shape[0]
        index = torch.arange(B).view(-1, 1).to(q_box.device)
        idx_cbox = torch.cat((index, c_box), dim=-1).contiguous()
        # obtain the feature of cropped background region
        roi_feat = self.RoIAlign(bg_feat, idx_cbox) # b,c,roi_size,roi_size
        # compute the target bounding box for foreground feature
        tar_box = torch.zeros_like(q_box)
        crop_w  = c_box[:,2] - c_box[:,0]
        crop_h  = c_box[:,3] - c_box[:,1]
        tar_box[:, 0] = torch.floor((q_box[:,0] - c_box[:,0]) / crop_w * self.alignsize)
        tar_box[:, 1] = torch.floor((q_box[:,1] - c_box[:,1]) / crop_h * self.alignsize)
        tar_box[:, 2] = torch.ceil((q_box[:,2] - c_box[:,0])  / crop_w * self.alignsize)
        tar_box[:, 3] = torch.ceil((q_box[:,3] - c_box[:,1])  / crop_h * self.alignsize)
        tar_box = tar_box.int()
        assert tar_box.min() >= 0 and tar_box.max() <= self.alignsize, \
            f'tar box coordinates max={tar_box.max()}, min={tar_box.min()}'
        # as the target size of foreground feature varies with background image,
        # we separately process different backgrounds.

        roi_feat = rearrange(roi_feat, '(b n) c h w -> b n c h w', b=num_bg)
        fg_feat = rearrange(fg_feat, '(b n) c h w -> b n c h w', b=num_bg)
        tar_box = rearrange(tar_box, '(b n) d -> b n d', b=num_bg)

        feat_collect = []
        for i in range(num_bg):
            comp_feat = roi_feat[i].clone()
            per_fg_feat = fg_feat[i]
            x1,y1,x2,y2 = tar_box[i, 0]
            rs_w, rs_h = x2 - x1, y2 - y1
            per_fg_feat = F.interpolate(per_fg_feat, (rs_h, rs_w))
            comp_feat[:, :, y1:y2, x1:x2] = per_fg_feat
            feat_collect.append(comp_feat)
        feat_collect = torch.stack(feat_collect, dim=0)
        feat_collect = rearrange(feat_collect, 'b n c h w -> (b n) c h w')
        out_feat = self.convs(feat_collect)
        score = self.classifier(out_feat)
        return out_feat, score

class GlobalConcatDistillation(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = conv_relu_x2(in_ch*2, out_ch)
        self.classifier = binary_classifier(out_ch)

    def forward(self, bg_feat, fg_feat, q_box, c_box, num_bg):
        '''
        :param bg_feat: b,c,h,w
        :param fg_feat: b,c,h,w
        :param q_box: query box, b,4
        :param c_box: cropped bounding box, b,4
        :return:
        '''
        concat_feat = torch.cat([bg_feat, fg_feat], dim=1)
        out_feat = self.convs(concat_feat)
        score = self.classifier(out_feat)
        return out_feat, score

class RoIConcatDistillation(nn.Module):
    def __init__(self, in_ch, out_ch, alignsize, downsample):
        super().__init__()
        self.RoIAlign = RoIAlign(alignsize, 1.0 / 2 ** downsample, -1)
        self.convs = conv_relu_x2(in_ch*2, out_ch)
        self.alignsize = alignsize
        self.classifier = binary_classifier(out_ch)

    def forward(self, bg_feat, fg_feat, q_box, c_box, num_bg):
        '''
        :param bg_feat: b,c,h,w
        :param fg_feat: b,c,h,w
        :param q_box: query box, b,4
        :param c_box: cropped bounding box, b,4
        :return:
        '''
        B = bg_feat.shape[0]
        index = torch.arange(B).view(-1, 1).to(q_box.device)
        idx_cbox = torch.cat((index, c_box), dim=-1).contiguous()
        # obtain the feature of cropped background region
        roi_feat = self.RoIAlign(bg_feat, idx_cbox)  # b,c,h,w
        fg_feat_rs = F.interpolate(fg_feat, (self.alignsize, self.alignsize))
        concat_feat = torch.cat([roi_feat, fg_feat_rs], dim=1)
        out_feat = self.convs(concat_feat)
        score = self.classifier(out_feat)
        return out_feat, score

class VectorConcatDistillation(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU())
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.classifier = binary_classifier(out_ch)

    def forward(self, bg_feat, fg_feat, q_box, c_box, num_bg):
        '''
        :param bg_feat: b,c,h,w
        :param fg_feat: b,c,h,w
        :param q_box: query box, b,4
        :param c_box: cropped bounding box, b,4
        :return:
        '''
        bg_feat  = self.gap(bg_feat)
        fg_feat  = self.gap(fg_feat)
        bg_vec = bg_feat.flatten(1)
        fg_vec = fg_feat.flatten(1)
        out_vec = self.fc(torch.concat([bg_vec, fg_vec], dim=1))
        out_feat = out_vec.unsqueeze(-1).unsqueeze(-1)
        score = self.classifier(out_feat)
        return out_feat, score

class RoIVectorConcatDistillation(nn.Module):
    def __init__(self, in_ch, out_ch, downsample):
        super().__init__()
        self.RoIAlign = RoIAlign(1, 1.0 / 2 ** downsample, -1)
        self.fc = nn.Sequential(
            nn.Linear(2*in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU())
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.classifier = binary_classifier(out_ch)

    def forward(self, bg_feat, fg_feat, q_box, c_box, num_bg):
        '''
        :param bg_feat: b,c,h,w
        :param fg_feat: b,c,h,w
        :param q_box: query box, b,4
        :param c_box: cropped bounding box, b,4
        :return:
        '''
        B = bg_feat.shape[0]
        index = torch.arange(B).view(-1, 1).to(q_box.device)
        idx_cbox = torch.cat((index, c_box), dim=-1).contiguous()
        # obtain the feature of cropped background region
        roi_feat  = self.RoIAlign(bg_feat, idx_cbox)  # b,c,h,w
        bg_vec = roi_feat.flatten(1)
        fg_vec = self.gap(fg_feat).flatten(1)
        out_vec = self.fc(torch.concat([bg_vec, fg_vec], dim=1))
        out_feat = out_vec.unsqueeze(-1).unsqueeze(-1)
        score = self.classifier(out_feat)
        return out_feat, score

class StudentModel(nn.Module):
    def __init__(self, cfg, loadweights=True):
        super().__init__()
        assert cfg.backbone == 'vgg19', cfg.backbone
        model = models.vgg19(pretrained=loadweights)
        backbone = model.features
        self.base_dim = 512
        self.bg_encoder = nn.ModuleList([
            nn.Sequential(backbone[:23]),
            nn.Sequential(backbone[23:30]),
            nn.Sequential(backbone[30:])
        ])
        backbone_copy = copy.deepcopy(backbone)
        self.fg_encoder = nn.ModuleList([
            nn.Sequential(backbone_copy[:23]),
            nn.Sequential(backbone_copy[23:30]),
            nn.Sequential(backbone_copy[30:])
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = cfg.feature_dim
        out_ch = 512
        # branch of feature distillation
        '''
        encoder_feature: gap or roi
        encoder_classify: true or false
        distill_type: roiconcat, roicropresize, gloablconcat, roivectorconcat, gapconcat, onlyclassify, none
        '''
        self.encoder_feat = cfg.encoder_feature
        self.encoder_classify = cfg.encoder_classify
        self.distill_type = cfg.distill_type

        if self.encoder_classify:
            self.auxiliary_classifier = nn.Sequential(
                nn.Linear(out_ch * 2, out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, 1, bias=False),
                nn.Sigmoid())

        alignsize, downsample = 7, 5
        self.distill_module = get_distillation_module(self.distill_type,
                                                      out_ch, out_ch,
                                                      alignsize, downsample)
        self.triplet_loss = TripletLoss(cfg.triplet_margin)

    def forward_bg_encoder(self, bg_im, crop_box):
        bg_f1 = self.bg_encoder[0](bg_im)
        bg_f2 = self.bg_encoder[1](bg_f1)
        bg_f3 = self.bg_encoder[2](bg_f2)
        if self.encoder_feat == 'gap':
            bg_emb = self.gap(bg_f3).flatten(1)
        else:
            bg_emb = self.extract_bg_roi(bg_f3, crop_box)
        return bg_emb

    def forward_fg_encoder(self, fg_im):
        fg_f1 = self.fg_encoder[0](fg_im)
        fg_f2 = self.fg_encoder[1](fg_f1)
        fg_f3 = self.fg_encoder[2](fg_f2)
        fg_emb = self.gap(fg_f3).flatten(1)
        return fg_emb

    def extract_bg_roi(self, bg_fm, crop_box):
        # extract background RoI feature
        B = bg_fm.shape[0]
        index = torch.arange(B).view(-1, 1).to(crop_box.device)
        idx_cbox = torch.cat((index, crop_box), dim=-1).contiguous()
        bg_roi = roi_align(bg_fm, idx_cbox, output_size=1,
                           spatial_scale=1. / 2 ** 5)
        bg_emb = bg_roi.flatten(1)
        return bg_emb

    def forward(self, bg_im, fg_im, query_box, crop_box, num_bg=1):
        bg_f1 = self.bg_encoder[0](bg_im)
        bg_f2 = self.bg_encoder[1](bg_f1)
        bg_f3 = self.bg_encoder[2](bg_f2)
        if self.encoder_feat == 'gap':
            self.bg_emb = self.gap(bg_f3).flatten(1)
        else:
            self.bg_emb = self.extract_bg_roi(bg_f3, crop_box)

        fg_f1 = self.fg_encoder[0](fg_im)
        fg_f2 = self.fg_encoder[1](fg_f1)
        fg_f3 = self.fg_encoder[2](fg_f2)
        self.fg_emb = self.gap(fg_f3).flatten(1)

        if self.encoder_classify:
            self.aux_score = self.auxiliary_classifier(torch.cat([self.bg_emb, self.fg_emb], dim=1))

        if self.distill_type == 'none':
            self.distill_feat, self.score = None, torch.empty([])
        else:
            if num_bg == None:
                num_bg = bg_im.shape[0]
            self.distill_feat, self.score = self.distill_module(bg_f3, fg_f3, query_box, crop_box, num_bg)
            self.score = self.score.flatten()
        return self.bg_emb, self.fg_emb, self.score


    def calcu_classify_loss(self, gt_label):
        gt_label = gt_label.reshape(-1,1)
        loss = 0.
        if self.distill_type != 'none':
            pr_score = self.score.reshape(-1,1)
            loss += F.binary_cross_entropy(pr_score, gt_label)
        if self.encoder_classify:
            pr_score2 = self.aux_score.reshape(-1,1)
            loss += F.binary_cross_entropy(pr_score2, gt_label)
            if self.distill_type != 'none':
                loss /= 2
        return loss

    def calcu_triplet_loss(self, gt_label, num_bg):
        pos_index = (gt_label > 0.5)
        neg_index = (gt_label < 0.5)
        pos_emb, neg_emb = self.fg_emb[pos_index], self.fg_emb[neg_index]
        bg_emb = rearrange(self.bg_emb, '(b n) c -> b n c', b=num_bg)
        bg_emb = bg_emb[:, 0]
        pos_emb = rearrange(pos_emb, '(b n) d -> b n d', b=num_bg)
        neg_emb = rearrange(neg_emb, '(b n) d -> b n d', b=num_bg)
        loss = self.triplet_loss(bg_emb, pos_emb, neg_emb)
        return loss

    def calcu_distillation_loss(self, feat_list, loss_type='l1'):
        if self.distill_type in ['none', 'onlyclassify']:
            return 0.
        if loss_type.lower() == 'l1':
            loss_func = F.l1_loss
        elif loss_type.lower() == 'l2':
            loss_func = F.mse_loss
        elif loss_type.lower() == 'smoothl1':
            loss_func = F.smooth_l1_loss
        else:
            raise Exception(f'undefined loss type {loss_type}')
        gt_featmap, gt_featvec = feat_list
        if gt_featmap.shape[2:] != self.distill_feat.shape[2:]:
            gt_featmap = F.interpolate(gt_featmap, self.distill_feat.shape[2:])
        kd_loss = loss_func(self.distill_feat, gt_featmap)
        return kd_loss