import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import cv2
import math


class Arguments(object):
    def __init__(self):
        self.motion = True

        self.lr = 1e-4
        self.lr_decay_rate = 0.99
        self.weight_decay = 1e-4
        # self.gradient_accumulation = True
        # self.accumulation = 4

        self.batch_size = 1
        self.epochs = 50
        self.clip_max_norm = 0.1
        self.device = 'cuda:0'

        self.resume = 'runs\sceneflow_toy\dev\experiment_2\epoch_10_model.pth.tar'
        self.resume_only_sttr = False
        self.sttr_resume = 'runs/sceneflow_pretrained_model.pth.tar'
        self.train_only_dc = True
        self.ft = True
        self.start_epoch = 0
        self.eval = False
        self.inference = False
        self.save_chekpoint = 1

        self.num_workers = 0
        self.checkpoint = 'dev'
        self.pre_train = True
        self.downsample = 5
        

        self.disp_change_splited = False
        self.apex = False

        self.height = 480
        self.width = 600

        self.origin_height = 540
        self.origin_width = 960

        # for disparity change Transformer
        self.mlp = True
        self.mlp_encoder_channel = [16, 32, 64]
        self.channel_dim = 128

        # Transformer for disp
        
        self.hidden_dim = 128
        self.position_encoding = 'sine1d_rel'
        self.num_attn_layers = 6
        self.nheads = 8

        # Transformer for delta disp
        self.hidden_dim_delta = 128
        self.num_attn_layers_delta = 6
        self.nheads_delta = 8

        self.regression_head = 'ot'
        self.context_adjustment_layer = 'cal'
        self.cal_num_blocks = 8
        self.cal_num_blocks_3d = 8
        self.cal_feat_dim = 16
        self.cal_expansion_ratio = 4
        self.ot = True
        self.ot_iter = 10

        self.dataset = 'sceneflow_toy'
        self.dataset_directory = 'D:/yqt/AllDatasets/MySceneFlow/MyTrainDataset'
        self.train_validation = 'train'
        self.validation = 'validation'
        self.dataset_max_length = None
        self.eval_percentage = 0.01

        self.px_error_threshold = 3
        self.loss_weight = 'rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0'
        self.validation_max_disp = 192

    def initialize(self, file_name):
        df = pd.read_excel(file_name)
        data = df.values.tolist()
        for i in data:
            setattr(self, i[0], i[1])


class NestedTensor(object):
    def __init__(self, left, right, disp=None, sampled_cols=None, sampled_rows=None, occ_mask=None,
                 occ_mask_right=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.occ_mask = occ_mask
        self.occ_mask_right = occ_mask_right
        self.sampled_cols = sampled_cols
        self.sampled_rows = sampled_rows

    def __str__(self):
        print("left: " + str(self.left.shape))
        print("right: " + str(self.right.shape))
        print("disp: " + str(self.disp.shape))
        print("occ: " + str(self.occ_mask.shape))
        print("occ_right: " + str(self.occ_mask_right.shape))


def my_center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


def np_center_crop(x, height, width):
    shape = x.shape
    dh = math.ceil((shape[0] - height) / 2)
    dw = math.ceil((shape[1] - width) / 2)
    if dh < 0 or dw < 0:
        assert "result shape must smaller than raw shape"
    return x[dh:shape[0] - dh, dw:shape[1] - dw]


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [1 if i != dim else -1 for i in range(1, len(source.shape))]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)


def torch_1d_sample(source, sample_points, mode='linear'):
    """
    linearly sample source tensor along the last dimension
    input:
        source [N,D1,D2,D3...,Dn]
        sample_points [N,D1,D2,....,Dn-1,1]
    output:
        [N,D1,D2...,Dn-1]
    """
    idx_l = torch.floor(sample_points).long().clamp(0, source.size(-1) - 1)
    idx_r = torch.ceil(sample_points).long().clamp(0, source.size(-1) - 1)

    if mode == 'linear':
        weight_r = sample_points - idx_l
        weight_l = 1 - weight_r
    elif mode == 'sum':
        weight_r = (idx_r != idx_l).int()  # we only sum places of non-integer locations
        weight_l = 1
    else:
        raise Exception('mode not recognized')

    out = torch.gather(source, -1, idx_l) * weight_l + torch.gather(source, -1, idx_r) * weight_r
    return out.squeeze(-1)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def find_occ_mask(disp_left, disp_right):
    """
    find occlusion map
    1 indicates occlusion
    disp range [0,w]
    """
    w = disp_left.shape[-1]

    # # left occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disp_left

    # 1. negative locations will be occlusion
    occ_mask_l = right_shifted <= 0

    # 2. wrong matches will be occlusion
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype(np.int)
    disp_right_selected = np.take_along_axis(disp_right, right_shifted,
                                             axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_right_selected - disp_left) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_right_selected <= 0.0] = False
    wrong_matches[disp_left <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_l] = True  # apply case 1 occlusion to case 2
    occ_mask_l = wrong_matches

    # # right occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    left_shifted = coord + disp_right

    # 1. negative locations will be occlusion
    occ_mask_r = left_shifted >= w

    # 2. wrong matches will be occlusion
    left_shifted[occ_mask_r] = 0  # set negative locations to 0
    left_shifted = left_shifted.astype(np.int)
    disp_left_selected = np.take_along_axis(disp_left, left_shifted,
                                            axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_left_selected - disp_right) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_left_selected <= 0.0] = False
    wrong_matches[disp_right <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_r] = True  # apply case 1 occlusion to case 2
    occ_mask_r = wrong_matches

    return occ_mask_l, occ_mask_r


def save_and_clear(idx, output_file):
    with open('output-' + str(idx) + '.dat', 'wb') as f:
        torch.save(output_file, f)
    idx += 1

    # clear
    for key in output_file:
        output_file[key].clear()

    return idx


def show_tensor(x: Tensor):
    array1 = x.detach().numpy()  # 将tensor数据转为numpy数据
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
    mat = np.uint8(array1)  # float32-->uint8
    print('mat_shape:', mat.shape)
    mat = mat.transpose(1, 2, 0)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", mat)
    cv2.waitKey(0)


def center_cut(x: Tensor, height: int, width: int):
    # 用于裁切一点点像素误差的特征图……
    shape = x.shape
    # x[:, :, :1024, :128, ].shape
    # Out[21]: torch.Size([1, 3, 1024, 128])
    dh = math.ceil((shape[2] - height) / 2)
    dw = math.ceil((shape[3] - width) / 2)
    if dh < 0 or dw < 0:
        assert "result shape must smaller than raw shape"
    return x[:, :, dh:shape[2] - dh, dw:shape[3] - dw].to(x.device)


def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)

