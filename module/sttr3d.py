
from module.sttr import build_sttr
from module.transformer_3d import Transformer3D
from module.regression_head import RegressionHead3D
from module.context_adjustment_layer import ContextAdjustmentLayer3D
from module.pos_encoder import MLPEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.integration_tools import Arguments, NestedTensor, batched_index_select


class STTR3D(nn.Module):
    
    def __init__(self, args: Arguments) -> None:
        super().__init__()
        self.args = args

        self.sttr = build_sttr(args)

        self.mlp_encoder = MLPEncoder(feature_dim=args.channel_dim, layers=args.mlp_encoder_channel, args=args)

        self.transformer_3d = Transformer3D(feature_dim=args.channel_dim, embed_dim=args.hidden_dim_delta, nhead=args.nheads_delta,
                                            num_attn_layers=args.num_attn_layers_delta, batch=args.batch_size, args=args)

        self.refine = ContextAdjustmentLayer3D(num_res=args.cal_num_blocks_3d, feature_dim=3, feature_res=args.cal_feat_dim, 
                                               expansion_ratio=args.cal_expansion_ratio, downsample=args.downsample, args=args)
        
        self.regression_3d = RegressionHead3D(args=args, cal=self.refine, ot=True)

        self._reset_parameters()
        
        self._disable_batchnorm_tracking()

        self._relu_inplace()
    
    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
    
    def forward(self, x: NestedTensor, y: NestedTensor):
        
        out_t1, feat_t1, _ = self.sttr(x)
        out_t2, feat_t2, _ = self.sttr(y)

        bs, _, h, w = feat_t1.shape

        mlp_enc_1 = self.mlp_encoder(feat_t1)
        mlp_enc_2 = self.mlp_encoder(feat_t2)
        # downsample
        if x.sampled_cols is not None:
            mlp_enc_1 = batched_index_select(mlp_enc_1.reshape(bs, -1, h, w), 3, x.sampled_cols)
            mlp_enc_2 = batched_index_select(mlp_enc_2.reshape(bs, -1, h, w), 3, x.sampled_cols)
            feat_t1 = batched_index_select(feat_t1, 3, x.sampled_cols)
            feat_t2 = batched_index_select(feat_t2, 3, x.sampled_cols)
        if x.sampled_rows is not None:
            mlp_enc_1 = batched_index_select(mlp_enc_1, 2, x.sampled_rows)
            mlp_enc_2 = batched_index_select(mlp_enc_2, 2, x.sampled_rows)
            feat_t1 = batched_index_select(feat_t1, 2, x.sampled_rows)
            feat_t2 = batched_index_select(feat_t2, 2, x.sampled_rows)

        attn = self.transformer_3d(feat_t1, feat_t2, 
                                   mlp_enc_1.reshape(bs, self.args.channel_dim, -1), 
                                   mlp_enc_2.reshape(bs, self.args.channel_dim, -1))

        if x.sampled_cols is not None:
            disp_t1 = batched_index_select(out_t1['disp_pred'], 2, x.sampled_cols)
            disp_t2 = batched_index_select(out_t2['disp_pred'], 2, x.sampled_cols)
        if x.sampled_rows is not None:
            disp_t1 = batched_index_select(disp_t1, 1, x.sampled_rows)
            disp_t2 = batched_index_select(disp_t2, 1, x.sampled_rows)

        disp_change = self.regression_3d(attn, disp_t1, disp_t2, x.left, y.left)

        return {'out_t1': out_t1, 'out_t2': out_t2, 'disp_change': disp_change}
    

class STTR3D_withoutMLP(nn.Module):
    
    def __init__(self, args: Arguments) -> None:
        super().__init__()
        self.args = args

        self.sttr = build_sttr(args)

        self.transformer_3d = Transformer3D(feature_dim=args.channel_dim, embed_dim=args.hidden_dim_delta, nhead=args.nheads_delta,
                                            num_attn_layers=args.num_attn_layers_delta, batch=args.batch_size, args=args)

        self.refine = ContextAdjustmentLayer3D(num_res=args.cal_num_blocks_3d, feature_dim=3, feature_res=args.cal_feat_dim, 
                                               expansion_ratio=args.cal_expansion_ratio, downsample=args.downsample, args=args)
        
        self.regression_3d = RegressionHead3D(args=args, cal=self.refine, ot=True)

        self._reset_parameters()
        
        self._disable_batchnorm_tracking()

        self._relu_inplace()
    
    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
    
    def forward(self, x: NestedTensor, y: NestedTensor):
        
        out_t1, feat_t1, _ = self.sttr(x)
        out_t2, feat_t2, _ = self.sttr(y)

        bs, _, h, w = feat_t1.shape

        # downsample
        if x.sampled_cols is not None:
            feat_t1 = batched_index_select(feat_t1, 3, x.sampled_cols)
            feat_t2 = batched_index_select(feat_t2, 3, x.sampled_cols)
        if x.sampled_rows is not None:
            feat_t1 = batched_index_select(feat_t1, 2, x.sampled_rows)
            feat_t2 = batched_index_select(feat_t2, 2, x.sampled_rows)

        attn = self.transformer_3d(feat_t1, feat_t2, 
                                   None, 
                                   None)

        if x.sampled_cols is not None:
            disp_t1 = batched_index_select(out_t1['disp_pred'], 2, x.sampled_cols)
            disp_t2 = batched_index_select(out_t2['disp_pred'], 2, x.sampled_cols)
        if x.sampled_rows is not None:
            disp_t1 = batched_index_select(disp_t1, 1, x.sampled_rows)
            disp_t2 = batched_index_select(disp_t2, 1, x.sampled_rows)

        disp_change = self.regression_3d(attn, disp_t1, disp_t2, x.left, y.left)

        return {'out_t1': out_t1, 'out_t2': out_t2, 'disp_change': disp_change}