import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

from transformers import CLIPTokenizer, CLIPModel

import open_clip
import re
from libcom.painterly_image_harmonization.source.PHDiffusion.ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


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


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, model_path="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last"):  # clip-vit-base-patch32
        super().__init__()
        import os
        self.tokenizer   = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
        self.transformer = CLIPModel.from_pretrained(pretrained_model_name_or_path=model_path).text_model
        self.device      = device
        self.max_length  = max_length
        if freeze:
            self.freeze()
        self.layer = layer

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        device = next(self.transformer.parameters()).device
        tokens = batch_encoding["input_ids"].to(device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer != 'last')

        if self.layer == 'penultimate':
            z = outputs.hidden_states[-2]
            z = self.transformer.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)



if __name__ == "__main__":
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
