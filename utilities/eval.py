from typing import Iterable

import torch
from tqdm import tqdm

from utilities.integration_tools import NestedTensor, Arguments

from utilities.saver_and_logger import write_summary
from utilities.saver_and_logger import TensorboardSummary


@torch.no_grad()
def evaluate(args: Arguments, model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
             device: torch.device, epoch: int, summary: TensorboardSummary, save_output: bool,
             sampled_cols: torch.Tensor, sampled_rows: torch.Tensor):
    model.eval()
    criterion.eval()
    
    # initialize stats
    eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0, 'disp_change_l1': 0.0, 'disp_change_epe': 0.0, 'disp_change_error_px': 0.0,
                  'disp_change_total_px': 0.0, }

    tbar = tqdm(data_loader)
    change_valid = 0
    for idx, data in enumerate(tbar):
        # forward pass
        # 一次读俩
        # 如果data是None，那么说明是视频间断点，直接下一个idx读取即可
        # 例如，id为9的帧没有对应的视察变化，因为10是另外一个视频序列
        if 'Error' in data:
            # print(str(i)+" break point")
            continue
        # 数据都已经排好了，不需要分裂，直接怼进去(团长，车已经准备好了！)
        left_t1, right_t1 = data['left_t1'].to(device), data['right_t1'].to(device)
        left_t2, right_t2 = data['left_t2'].to(device), data['right_t2'].to(device)
        disp_t1, disp_t2 = torch.squeeze(data['disp_t1'], dim=1).to(device), torch.squeeze(data['disp_t2'], dim=1).to(
            device)
        occ_mask_t1, occ_mask_t2 = torch.squeeze(data['occ_mask_t1'], dim=1).to(device), torch.squeeze(
            data['occ_mask_t2'], dim=1).to(device)
        occ_mask_right_t1, occ_mask_right_t2 = torch.squeeze(data['occ_mask_right_t1'], dim=1).to(
            device), torch.squeeze(data['occ_mask_right_t2'], dim=1).to(
            device)
        disp_change = torch.squeeze(data['disp_change'], dim=1).to(device)

        x = NestedTensor(left_t1, right_t1, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t1,
                         occ_mask=occ_mask_t1, occ_mask_right=occ_mask_right_t1)

        y = NestedTensor(left_t2, right_t2, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t2,
                         occ_mask=occ_mask_t2, occ_mask_right=occ_mask_right_t2)

        # forward pass
        outputs = model(x, y)
        # outputs 是个字典 {'out_t1': out_t1, 'out_t2': out_t2, "indices": indices, "disp_change": disp_change}
        # compute loss
        losses = criterion(x, y, disp_change, outputs)
        
        change_valid += 1
        # get the loss
        try:
            eval_stats['rr'] += losses['rr'].item()
            eval_stats['l1_raw'] += losses['l1_raw'].item()
            eval_stats['l1'] += losses['l1'].item()
            eval_stats['occ_be'] += losses['occ_be'].item()

            eval_stats['iou'] += losses['iou'].item()
            eval_stats['epe'] += losses['epe'].item()
            eval_stats['error_px'] += losses['error_px']
            eval_stats['total_px'] += losses['total_px']

            eval_stats['disp_change_l1'] += losses['disp_change']['l1'].item()
            eval_stats['disp_change_epe'] += losses['disp_change']['epe'].item()
            eval_stats['disp_change_error_px'] += losses['disp_change']['px_error']
            eval_stats['disp_change_total_px'] += losses['disp_change']['total_px']
        except:
            eval_stats['rr'] += losses['loss1']['rr'].item()
            eval_stats['l1_raw'] += losses['loss1']['l1_raw'].item()
            eval_stats['l1'] += losses['loss1']['l1'].item()
            eval_stats['occ_be'] += losses['loss1']['occ_be'].item()

            eval_stats['iou'] += losses['loss1']['iou'].item()
            eval_stats['epe'] += losses['loss1']['epe'].item()
            eval_stats['error_px'] += losses['loss1']['error_px']
            eval_stats['total_px'] += losses['loss1']['total_px']

            eval_stats['rr'] += losses['loss2']['rr'].item()
            eval_stats['l1_raw'] += losses['loss2']['l1_raw'].item()
            eval_stats['l1'] += losses['loss2']['l1'].item()
            eval_stats['occ_be'] += losses['loss2']['occ_be'].item()

            eval_stats['iou'] += losses['loss2']['iou'].item()
            eval_stats['epe'] += losses['loss2']['epe'].item()
            eval_stats['error_px'] += losses['loss2']['error_px']
            eval_stats['total_px'] += losses['loss2']['total_px']

            eval_stats['disp_change_l1'] += losses['disp_change']['l1'].item()
            eval_stats['disp_change_epe'] += losses['disp_change']['epe'].item()
            eval_stats['disp_change_error_px'] += losses['disp_change']['px_error']
            eval_stats['disp_change_total_px'] += losses['disp_change']['total_px']

        # clear cache
        torch.cuda.empty_cache()

    # compute avg
    eval_stats['epe'] = eval_stats['epe'] / (2 * change_valid)
    eval_stats['iou'] = eval_stats['iou'] / (2 * change_valid)
    eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']

    eval_stats['disp_change_epe'] = eval_stats['disp_change_epe'] / change_valid
    eval_stats['disp_change_px_error_rate'] = eval_stats['disp_change_error_px'] / eval_stats['disp_change_total_px']
    # write to tensorboard
    write_summary(eval_stats, summary, epoch, 'eval')

    torch.cuda.empty_cache()

    return eval_stats


downsample = 0


def set_downsample(args):
    global downsample
    downsample = args.downsample
