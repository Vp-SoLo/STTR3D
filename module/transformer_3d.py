from typing import Optional, Any
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from module.attention import MultiheadAttention3D
from utilities.integration_tools import get_clones, Arguments

layer_idx = 0


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, embed_dim: int, nhead: int, batch: int):
        super().__init__()
        self.self_attn = MultiheadAttention3D(embed_dim, nhead, batch=batch)

        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, feat: Tensor,
                pos: Optional[Tensor] = None):
        feat2 = self.norm1(feat)

        _, feat2 = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos)

        feat = feat + feat2

        return feat


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, embed_dim: int, nhead: int, batch: int):
        super().__init__()
        self.cross_attn = MultiheadAttention3D(embed_dim, nhead, batch=batch)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, feat_t1: Tensor, feat_t2: Tensor,
                pos: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        feat_t1_2 = self.norm1(feat_t1)
        feat_t2_2 = self.norm1(feat_t2)

        feat_t2_2 = self.cross_attn(query=feat_t2_2, key=feat_t1_2, value=feat_t1_2, pos_enc=pos)[1]

        feat_t2 = feat_t2 + feat_t2_2

        # update left features

        # normalize again the updated right features
        feat_t2_2 = self.norm2(feat_t2)
        raw_attn, feat_t1_2 = self.cross_attn(query=feat_t1_2, key=feat_t2_2, value=feat_t2_2,
                                              attn_mask=None, pos_enc=pos)

        feat_t1 = feat_t1 + feat_t1_2

        return feat_t1, feat_t2, raw_attn


class Transformer3D(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, feature_dim: int = 64, embed_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6, batch: int = 1, args: Arguments = None):
        super().__init__()

        self.downsample = args.downsample
        self_attn_layer = TransformerSelfAttnLayer(embed_dim, nhead, batch=batch)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(embed_dim, nhead, batch=batch)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.hidden_dim = embed_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor, pos_enc: Any):
        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            feat_t1 = checkpoint(create_custom_self_attn(self_attn), feat_t1, pos_enc)

            # add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            feat_t1, feat_t2, raw_attn = checkpoint(create_custom_cross_attn(cross_attn), feat_t1, feat_t2, pos_enc)

        layer_idx = 0
        return raw_attn

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor, pos_enc_1: torch.Tensor, pos_enc_2: torch.Tensor):

        # reshape
        n, c, h, w = feat_t1.shape
        feat_t1 = feat_t1.contiguous().view(n * h * w, c)
        feat_t2 = feat_t2.contiguous().view(n * h * w, c)
        # position encoding
        if pos_enc_1 is not None:
            pos_enc_1 = torch.reshape(pos_enc_1, (-1, self.hidden_dim))
            pos_enc_2 = torch.reshape(pos_enc_2, (-1, self.hidden_dim))
            feat_t1 += pos_enc_1
            feat_t2 += pos_enc_2
        attn_3d = self._alternating_attn(feat_t1=feat_t1, feat_t2=feat_t2, pos_enc=None)
        return attn_3d
