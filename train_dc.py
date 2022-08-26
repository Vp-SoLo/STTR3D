
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


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 通过Excel文件保存参数，哈哈
    args = Arguments()

    device = torch.device(args.device)

    # 初始化模型
    model = STTR3D(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    scaler = torch.cuda.amp.GradScaler()

    # 载入预训练或者恢复
    prev_best = np.inf
    if args.resume != '' and not args.resume_only_sttr:
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        print("Pre-trained model successfully loaded.")
        if not (args.ft or args.inference or args.eval):
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            prev_best = checkpoint['best_pred']
            print("Pre-trained optimizer, lr scheduler and stats successfully loaded.")
    
    elif args.sttr_resume != '':
        if not os.path.isfile(args.sttr_resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.sttr_resume}'")
        checkpoint = torch.load(args.sttr_resume)
        pretrained_dict = checkpoint['state_dict']
        model.sttr.load_state_dict(pretrained_dict)
        print("STTR's Pre-trained model successfully loaded.")
        if not (args.ft or args.inference or args.eval):
            prev_best = checkpoint['best_pred']
    
    if args.train_only_dc:
        for i in model.sttr.parameters():
            i.requires_grad = False

    checkpoint_saver = Saver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)
    data_loader_train, data_loader_val, _ = build_data_loader(args)

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

    iter = 0
    eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0, 'disp_change_l1': 0.0, 'disp_change_epe': 0.0, 'disp_change_error_px': 0.0,
                  'disp_change_total_px': 0.0, }
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch: %d" % epoch)

        model.train()

        criterion.train()

        tbar = tqdm(data_loader_train)

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
            disp_t1, disp_t2 = torch.squeeze(data['disp_t1'], dim=1).to(device), torch.squeeze(data['disp_t2'], dim=1).to(device)
            occ_mask_t1, occ_mask_t2 = torch.squeeze(data['occ_mask_t1'], dim=1).to(device), torch.squeeze(data['occ_mask_t2'], dim=1).to(device)
            occ_mask_right_t1, occ_mask_right_t2 = torch.squeeze(data['occ_mask_right_t1'], dim=1).to(device), torch.squeeze(data['occ_mask_right_t2'], dim=1).to(
                device)
            disp_change = torch.squeeze(data['disp_change'], dim=1).to(device)

            x = NestedTensor(left_t1, right_t1, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t1,
                             occ_mask=occ_mask_t1, occ_mask_right=occ_mask_right_t1)

            y = NestedTensor(left_t2, right_t2, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp_t2,
                             occ_mask=occ_mask_t2, occ_mask_right=occ_mask_right_t2)

            # forward pass
            with torch.cuda.amp.autocast():
                outputs = model(x, y)
                # outputs 是个字典 {'out_t1': out_t1, 'out_t2': out_t2, "indices": indices, "disp_change": disp_change}
                # compute loss
                losses = criterion(x, y, disp_change, outputs)

            if losses is None:
                continue

            # terminate training if exploded
            if not math.isfinite(losses['aggregated'].item()):
                print("Loss is {}, stopping training".format(losses['aggregated'].item()))
                sys.exit(1)

            scaler.scale(losses['aggregated']).backward()

            iter += 1
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
            if iter % 10 == 0 and iter != 0:

                eval_stats['epe'] = eval_stats['epe'] / (2 * 10)
                eval_stats['iou'] = eval_stats['iou'] / (2 * 10)
                eval_stats['l1'] = eval_stats['l1'] / (2 * 10)
                eval_stats['l1_raw'] = eval_stats['l1_raw'] / (2 * 10)
                eval_stats['rr'] = eval_stats['rr'] / (2 * 10)
                eval_stats['occ_be'] = eval_stats['occ_be'] / (2 * 10)
                eval_stats['disp_change_l1'] = eval_stats['disp_change_l1'] / (2 * 10)
                eval_stats['disp_change_epe'] = eval_stats['disp_change_epe'] / (2 * 10)

                eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']
                eval_stats['disp_change_px_error_rate'] = eval_stats['disp_change_error_px'] / eval_stats['disp_change_total_px']

                print(eval_stats)
                write_summary(eval_stats, summary_writer, iter, 'train')

                eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0, 'disp_change_l1': 0.0, 'disp_change_epe': 0.0, 'disp_change_error_px': 0.0,
                  'disp_change_total_px': 0.0, }
            # clip norm
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

            # 梯度下降
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # clear cache
            torch.cuda.empty_cache()

            if iter % 5000 == 0:
                lr_scheduler.step()
                print("current learning rate", lr_scheduler.get_lr())
                save_checkpoint(iter, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, by_iter = True, amp = None)

        torch.cuda.empty_cache()

        # saver
        # save if pretrain, save every 50 epochs
        if args.pre_train or epoch % args.save_chekpoint == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, by_iter = False, amp = None)

        # # eval
        # # validate
        # eval_stats = evaluate(args=args, model=model, criterion=criterion, data_loader=data_loader_val, device=device,
        #                       epoch=epoch, summary=summary_writer, save_output=False,
        #                       sampled_cols=sampled_cols, sampled_rows=sampled_rows)
        # # save if best
        # if prev_best > eval_stats['epe'] and 0.5 > eval_stats['px_error_rate']:
        #     save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, True, None)

    # save final model
    save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, None)

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()

