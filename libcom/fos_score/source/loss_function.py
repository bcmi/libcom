import torch.nn as nn
import torch
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, need_norm=True):
        '''
        :param anchor:   num_anchor, emb_dim
        :param positive: num_anchor, num_pos, emb_dim
        :param negative: num_anchor, num_neg, emb_dim
        :return:
        '''
        if need_norm:
            anchor   = F.normalize(anchor,   2, -1)
            positive = F.normalize(positive, 2, -1)
            negative = F.normalize(negative, 2, -1)
        pos_cos  = torch.einsum('b d, b n d -> b n', anchor, positive)
        # print('cosine similarity', anchor.shape, positive.shape, pos_cos.shape, pos_cos.min(), pos_cos.max())
        pos_dist = 1 - pos_cos
        neg_cos  = torch.einsum('b d, b n d -> b n', anchor, negative)
        neg_dist = 1 - neg_cos
        loss = F.relu(pos_dist.unsqueeze(2) - neg_dist.unsqueeze(1) + self.margin)
        return loss.mean()

class TripletHardLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletHardLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, need_norm=True):
        '''
        :param anchor: num_anchor, emb_dim
        :param positive: num_anchor, num_pos, emb_dim
        :param negative: num_anchor, num_neg, emb_dim
        :return:
        '''
        if need_norm:
            anchor   = F.normalize(anchor,   2, -1)
            positive = F.normalize(positive, 2, -1)
            negative = F.normalize(negative, 2, -1)
        pos_cos = torch.einsum('b d, b n d -> b n', anchor, positive)
        pos_dist = 1 - pos_cos
        neg_cos = torch.einsum('b d, b n d -> b n', anchor, negative)
        neg_dist = 1 - neg_cos
        loss = F.relu(pos_dist.max(-1)[0] - neg_dist.min(-1)[0] + self.margin)
        return loss.mean()

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def kl_divergence_loss(log_var, mu):
    '''
    the KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD


def custom_kl_loss(p_mu, p_var, q_mu, q_var):
    p_distribution = torch.distributions.MultivariateNormal(p_mu, torch.diag_embed(p_var))
    q_distribution = torch.distributions.MultivariateNormal(q_mu, torch.diag_embed(q_var))
    p_q_kl = torch.distributions.kl_divergence(p_distribution, q_distribution).mean()
    return p_q_kl
    # b,d = p_mu.shape
    # normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(b, d),
    #                                                              torch.diag_embed(torch.ones(b,d)).cuda())
    # p_normal_kl = torch.distributions.kl_divergence(p_distribution, normal_distribution).mean()

def mask_iou_loss(inputs, targets):
    # flatten label and prediction tensors
    inputs = inputs.flatten(1).float()
    targets = targets.flatten(1).float()

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum(1)
    total = (inputs + targets).sum(1)
    union = total - intersection
    IoU = intersection / (union + 1e-12)
    return 1 - IoU.mean()