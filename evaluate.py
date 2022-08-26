
import torch
import numpy as np
import random
import os
import math
import sys

from tqdm import tqdm
from utilities.integration_tools import NestedTensor
from module.sttr3d import STTR3D
from module.loss import build_criterion
from dataset import build_data_loader
from utilities.integration_tools import Arguments
from utilities.saver_and_logger import Saver, TensorboardSummary, save_checkpoint, write_summary
from utilities.eval import evaluate


@torch.no_grad()
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = Arguments()
    args.dataset_directory = 'D:/yqt/AllDatasets/MySceneFlow/MyTrainDataset'
    args.save_chekpoint = 'runs/sceneflow_toy/dev/experiment_6/epoch_0_model.pth.tar'
    args.dataset_max_length = None
    args.eval_percentage = 0.9
    # args.height = 540
    # args.width = 960

    device = torch.device(args.device)

    # 初始化模型
    model = STTR3D(args).to(device)

    # 载入预训练或者恢复
    prev_best = np.inf
    if args.resume != '' and not args.resume_only_sttr:
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        print("Pre-trained model successfully loaded.")
    
    elif args.sttr_resume != '':
        if not os.path.isfile(args.sttr_resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.sttr_resume}'")
        checkpoint = torch.load(args.sttr_resume)
        pretrained_dict = checkpoint['state_dict']
        model.sttr.load_state_dict(pretrained_dict)
        print("STTR's Pre-trained model successfully loaded.")
        if not (args.ft or args.inference or args.eval):
            prev_best = checkpoint['best_pred']
    
    for i in model.parameters():
        i.requires_grad = False

    # checkpoint_saver = Saver(args)
    # summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)


    data_loader, _, _ = build_data_loader(args)

    # 损失函数评价体
    criterion = build_criterion(args)

    # using 1d index to sample
    sampled_cols = None
    sampled_rows = None
    if args.downsample > 1:
        col_offset = int(args.downsample / 2)
        row_offset = int(args.downsample / 2)
        sampled_cols = torch.arange(col_offset, args.width, args.downsample)[None,].expand(args.batch_size, -1).to(
            device)
        sampled_rows = torch.arange(row_offset, args.height, args.downsample)[None,].expand(args.batch_size, -1).to(
            device)
    
    # initialize stats
    eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0, 'disp_change_l1': 0.0, 'disp_change_epe': 0.0, 'disp_change_error_px': 0.0,
                  'disp_change_total_px': 0.0, }

    tbar = tqdm(data_loader)
    change_valid = 0
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            # forward pass
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
                device), torch.squeeze(data['occ_mask_right_t2'], dim=1).to(device)
            disp_change = torch.squeeze(data['disp_change'], dim=1).to(device)

            x = NestedTensor(left_t1, right_t1, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t1,
                            occ_mask=occ_mask_t1, occ_mask_right=occ_mask_right_t1)

            y = NestedTensor(left_t2, right_t2, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t2,
                            occ_mask=occ_mask_t2, occ_mask_right=occ_mask_right_t2)

            # forward pass
            with torch.no_grad():
                outputs = model(x, y)
                # compute loss
                losses = criterion(x, y, disp_change, outputs)
                # print(losses)
            
            if losses is None:
                continue
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

                print('dc_epe: '+ str(eval_stats['disp_change_epe'] / change_valid))
            # clear cache
            torch.cuda.empty_cache()

        # compute avg
        eval_stats['epe'] = eval_stats['epe'] / (2 * change_valid)
        eval_stats['iou'] = eval_stats['iou'] / (2 * change_valid)
        eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']

        eval_stats['disp_change_epe'] = eval_stats['disp_change_epe'] / change_valid
        eval_stats['disp_change_px_error_rate'] = eval_stats['disp_change_error_px'] / eval_stats['disp_change_total_px']

        print(eval_stats)
        torch.cuda.empty_cache()
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()

