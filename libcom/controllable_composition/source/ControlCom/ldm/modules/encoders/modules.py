import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel,CLIPVisionModelWithProjection
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from ldm.modules.encoders.xf import LayerNorm, Transformer
import math
import torch.nn.functional as F

from transformers import logging
logging.set_verbosity_error()

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", 
                 local_hidden_index=-1, use_foreground_mask=False,
                 patchtoken_for_global=True):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper   = Transformer(
                n_ctx=257,
                width=1024,
                layers=5,
                heads=8,
            )
        self.proj_out=nn.Linear(1024, 768)
        self.freeze()
        self.local_index = local_hidden_index
        self.use_mask  = use_foreground_mask
        self.use_patchtoken = patchtoken_for_global

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True
        for param in self.proj_out.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            image, mask = inputs  
            outputs = self.transformer(pixel_values=image,
                                    output_hidden_states=True)
            all_hidden  = outputs.hidden_states
            # get deep embedding
            last_hidden = outputs.last_hidden_state # b,257,1024
            if self.use_mask and mask is not None:
                mask = F.interpolate(mask, (16, 16), mode='bilinear', align_corners=True)
                mask = mask.flatten(1).unsqueeze(-1) # b,256,1
                # prepare global embedding including cls and patch tokens
                global_cls   = last_hidden[:,0:1] # b,1,1024
                global_patch = last_hidden[:,1:] * (mask > 0.5) # b,256,1024
                global_emb   = torch.cat([global_cls, global_patch], dim=1) # b,257,1024
            else:
                if self.use_patchtoken:
                    global_emb = last_hidden
                else:
                    global_emb = last_hidden[:,0:1]
            gz = self.transformer.vision_model.post_layernorm(global_emb)
            gz = self.mapper(gz) # b,257,1024
            gz = self.final_ln(gz)
            gz = self.proj_out(gz) # b,257,768
            # prepare shallow embedding
            lz = all_hidden[self.local_index][:,1:]
            if self.use_mask and mask is not None:
                lz = lz * (mask > 0.5) # b,256,1024
            return [gz, lz]
        else:
            image = inputs
            outputs = self.transformer(pixel_values=image,
                                    output_hidden_states=True)
            all_hidden  = outputs.hidden_states
            last_hidden = outputs.last_hidden_state # b,257,1024
            gz = last_hidden[:,0:1]                 # b,1,1024
            lz = all_hidden[self.local_index][:,1:] # b,256,1024
            gz = self.transformer.vision_model.post_layernorm(gz)
            gz = self.mapper(gz) # b,257,1024
            gz = self.final_ln(gz)
            gz = self.proj_out(gz)
            return [gz, lz]
        
    def encode(self, image):
        return self(image)


if __name__ == "__main__":
    from ldm.util import count_params
    device = torch.device("cuda:0")
    model = FrozenCLIPImageEmbedder().to(device)
    count_params(model, verbose=True)
    img = torch.randn(2, 3, 224, 224).to(device)
    mask = torch.rand(2, 1, 64, 64).to(device)
    global_z, local_z = model((img, mask))
    print('global_z {}, local_z {}'.format(
        global_z.shape, local_z.shape
    ))