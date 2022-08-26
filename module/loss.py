from collections import OrderedDict

import torch
from torch import nn, Tensor

from utilities.integration_tools import batched_index_select, NestedTensor


class Criterion(nn.Module):
    """
    Compute loss and evaluation metrics, added with disparity change data
    """

    def __init__(self, threshold: int = 3, validation_max_disp: int = -1, loss_weight: list = None):
        super(Criterion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.px_threshold = threshold
        self.validation_max_disp = validation_max_disp
        self.weights = loss_weight

        self.l1_criterion = nn.SmoothL1Loss()
        self.epe_criterion = nn.L1Loss()
        
        self.disp_change_criterion_l1 = nn.SmoothL1Loss()
        self.disp_change_criterion_epe = nn.L1Loss()

    @torch.no_grad()
    def calc_px_error(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute px error

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """

        # computing threshold-px error
        loss_dict['error_px'] = torch.sum(
            torch.abs(pred[~invalid_mask] - disp[~invalid_mask]) > self.px_threshold).item()
        loss_dict['total_px'] = torch.sum(~invalid_mask).item()

        return

    @torch.no_grad()
    def compute_epe(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute EPE

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        loss_dict['epe'] = self.epe_criterion(pred[~invalid_mask], disp[~invalid_mask])
        return

    @torch.no_grad()
    def compute_iou(self, pred: Tensor, occ_mask: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute IOU on occlusion

        :param pred: occlusion prediction [N,H,W]
        :param occ_mask: ground truth occlusion mask [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        # threshold
        pred_mask = pred > 0.5

        # iou for occluded region
        inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
        union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()

        # iou for non-occluded region
        inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
        union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()

        # aggregate
        loss_dict['iou'] = (inter_occ + inter_noc).float() / (union_occ + union_noc)

        return

    def compute_rr_loss(self, outputs: dict, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute rr loss
        
        :param outputs: dictionary, outputs from the network
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: rr loss
        """""
        if invalid_mask is not None:
            if inputs.sampled_cols is not None:
                invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)

        # compute rr loss in non-occluded region
        gt_response = outputs['gt_response']
        eps = 1e-6
        rr_loss = - torch.log(gt_response + eps)

        if invalid_mask is not None:
            rr_loss = rr_loss[~invalid_mask]

        # if there is occlusion
        try:
            rr_loss_occ_left = - torch.log(outputs['gt_response_occ_left'] + eps)
            # print(rr_loss_occ_left.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_left])
        except KeyError:
            pass
        try:
            rr_loss_occ_right = - torch.log(outputs['gt_response_occ_right'] + eps)
            # print(rr_loss_occ_right.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_right])
        except KeyError:
            pass

        return rr_loss.mean()

    def compute_l1_loss(self, pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor, fullres: bool = True):
        """
        compute smooth l1 loss

        :param pred: disparity prediction [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :param fullres: Boolean indicating if prediction is full resolution
        :return: smooth l1 loss
        """
        disp = inputs.disp
        if not fullres:
            if inputs.sampled_cols is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
                disp = batched_index_select(disp, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)
                disp = batched_index_select(disp, 1, inputs.sampled_rows)

        return self.l1_criterion(pred[~invalid_mask], disp[~invalid_mask])

    def compute_entropy_loss(self, occ_pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute binary entropy loss on occlusion mask

        :param occ_pred: occlusion prediction, [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: binary entropy loss
        """
        eps = 1e-6

        occ_mask = inputs.occ_mask

        entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
        entropy_loss_noc = - torch.log(
            1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

        entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

        return entropy_loss.mean()

    def aggregate_loss(self, loss_dict: dict):
        """
        compute weighted sum of loss

        :param loss_dict: dictionary of losses
        """
        loss = 0.0
        for key in loss_dict:
            loss += loss_dict[key] * self.weights[key]

        loss_dict['aggregated'] = loss
        return

    def generate_disp_change_loss(self, pred: Tensor, ground_truth: Tensor, invalid_mask: Tensor):
        disp_change_loss = {}
        # px error
        disp_change_loss['px_error'] = torch.sum(
            torch.abs(pred[~invalid_mask] - ground_truth[~invalid_mask]) > self.px_threshold).item()
        disp_change_loss['total_px'] = torch.sum(~invalid_mask).item()

        # epe
        disp_change_loss['epe'] = self.disp_change_criterion_epe(pred, ground_truth)

        # smooth L1
        disp_change_loss['l1'] = self.disp_change_criterion_l1(pred, ground_truth)

        return disp_change_loss

    def forward(self, inputs1: NestedTensor, inputs2: NestedTensor, disp_change: Tensor, outputs: dict):
        """
        losses = criterion(x, y, data_disp_change.to(device), outputs)
        :param inputs1: input data1
        :param inputs2: input data2
        :param outputs: output from the network, dictionary
        :param disp_change: real disp_change, 1 channel data
        :return: loss dictionary
        """
        loss = {}
        loss1 = {}
        loss2 = {}

        if self.validation_max_disp == -1:
            invalid_mask = inputs1.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(inputs1.disp <= 0.0, inputs1.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            # print("find useless sample")
            return None

        loss1['rr'] = self.compute_rr_loss(outputs['out_t1'], inputs1, invalid_mask)
        loss1['l1_raw'] = self.compute_l1_loss(outputs['out_t1']['disp_pred'], inputs1, invalid_mask, fullres=True)
        loss1['l1'] = self.compute_l1_loss(outputs['out_t1']['disp_pred'], inputs1, invalid_mask)
        loss1['occ_be'] = self.compute_entropy_loss(outputs['out_t1']['occ_pred'], inputs1, invalid_mask)

        if self.validation_max_disp == -1:
            invalid_mask = inputs2.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(inputs2.disp <= 0.0, inputs2.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            # print("find useless sample")
            return None

        loss2['rr'] = self.compute_rr_loss(outputs['out_t2'], inputs2, invalid_mask)
        loss2['l1_raw'] = self.compute_l1_loss(outputs['out_t2']['disp_pred'], inputs2, invalid_mask, fullres=True)
        loss2['l1'] = self.compute_l1_loss(outputs['out_t2']['disp_pred'], inputs2, invalid_mask)
        loss2['occ_be'] = self.compute_entropy_loss(outputs['out_t2']['occ_pred'], inputs2, invalid_mask)

        self.aggregate_loss(loss1)
        self.aggregate_loss(loss2)

        loss['disp_change'] = self.generate_disp_change_loss(outputs['disp_change'], disp_change, invalid_mask)

        # for benchmarking
        self.calc_px_error(outputs['out_t1']['disp_pred'], inputs1.disp, loss1, invalid_mask)
        self.compute_epe(outputs['out_t1']['disp_pred'], inputs1.disp, loss1, invalid_mask)
        self.compute_iou(outputs['out_t1']['occ_pred'], inputs1.occ_mask, loss1, invalid_mask)

        self.calc_px_error(outputs['out_t2']['disp_pred'], inputs2.disp, loss2, invalid_mask)
        self.compute_epe(outputs['out_t2']['disp_pred'], inputs2.disp, loss2, invalid_mask)
        self.compute_iou(outputs['out_t2']['occ_pred'], inputs2.occ_mask, loss2, invalid_mask)

        loss['loss1'] = loss1
        loss['loss2'] = loss2
        loss['aggregated'] = loss1['aggregated'] + loss2['aggregated'] + loss['disp_change']['l1']

        return OrderedDict(loss)


class Batched(nn.Module):
    """
    Compute loss and evaluation metrics, added with disparity change data
    """

    def __init__(self, threshold: int = 3, validation_max_disp: int = -1, loss_weight: list = None):
        super(Criterion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.px_threshold = threshold
        self.validation_max_disp = validation_max_disp
        self.weights = loss_weight

        self.l1_criterion = nn.SmoothL1Loss()
        self.epe_criterion = nn.L1Loss()
        
        self.disp_change_criterion_l1 = nn.SmoothL1Loss()
        self.disp_change_criterion_epe = nn.L1Loss()

    @torch.no_grad()
    def calc_px_error(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute px error

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """

        # computing threshold-px error
        loss_dict['error_px'] = torch.sum(
            torch.abs(pred[~invalid_mask] - disp[~invalid_mask]) > self.px_threshold).item()
        loss_dict['total_px'] = torch.sum(~invalid_mask).item()

        return

    @torch.no_grad()
    def compute_epe(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute EPE

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        loss_dict['epe'] = self.epe_criterion(pred[~invalid_mask], disp[~invalid_mask])
        return

    @torch.no_grad()
    def compute_iou(self, pred: Tensor, occ_mask: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute IOU on occlusion

        :param pred: occlusion prediction [N,H,W]
        :param occ_mask: ground truth occlusion mask [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        # threshold
        pred_mask = pred > 0.5

        # iou for occluded region
        inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
        union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()

        # iou for non-occluded region
        inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
        union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()

        # aggregate
        loss_dict['iou'] = (inter_occ + inter_noc).float() / (union_occ + union_noc)

        return

    def compute_rr_loss(self, outputs: dict, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute rr loss
        
        :param outputs: dictionary, outputs from the network
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: rr loss
        """""
        if invalid_mask is not None:
            if inputs.sampled_cols is not None:
                invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)

        # compute rr loss in non-occluded region
        gt_response = outputs['gt_response']
        eps = 1e-6
        rr_loss = - torch.log(gt_response + eps)

        if invalid_mask is not None:
            rr_loss = rr_loss[~invalid_mask]

        # if there is occlusion
        try:
            rr_loss_occ_left = - torch.log(outputs['gt_response_occ_left'] + eps)
            # print(rr_loss_occ_left.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_left])
        except KeyError:
            pass
        try:
            rr_loss_occ_right = - torch.log(outputs['gt_response_occ_right'] + eps)
            # print(rr_loss_occ_right.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_right])
        except KeyError:
            pass

        return rr_loss.mean()

    def compute_l1_loss(self, pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor, fullres: bool = True):
        """
        compute smooth l1 loss

        :param pred: disparity prediction [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :param fullres: Boolean indicating if prediction is full resolution
        :return: smooth l1 loss
        """
        disp = inputs.disp
        if not fullres:
            if inputs.sampled_cols is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
                disp = batched_index_select(disp, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)
                disp = batched_index_select(disp, 1, inputs.sampled_rows)

        return self.l1_criterion(pred[~invalid_mask], disp[~invalid_mask])

    def compute_entropy_loss(self, occ_pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute binary entropy loss on occlusion mask

        :param occ_pred: occlusion prediction, [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: binary entropy loss
        """
        eps = 1e-6

        occ_mask = inputs.occ_mask

        entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
        entropy_loss_noc = - torch.log(
            1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

        entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

        return entropy_loss.mean()

    def aggregate_loss(self, loss_dict: dict):
        """
        compute weighted sum of loss

        :param loss_dict: dictionary of losses
        """
        loss = 0.0
        for key in loss_dict:
            loss += loss_dict[key] * self.weights[key]

        loss_dict['aggregated'] = loss
        return

    def generate_disp_change_loss(self, pred: Tensor, ground_truth: Tensor, invalid_mask: Tensor):
        disp_change_loss = {}
        # px error
        disp_change_loss['px_error'] = torch.sum(
            torch.abs(pred[~invalid_mask] - ground_truth[~invalid_mask]) > self.px_threshold).item()
        disp_change_loss['total_px'] = torch.sum(~invalid_mask).item()

        # epe
        disp_change_loss['epe'] = self.disp_change_criterion_epe(pred, ground_truth)

        # smooth L1
        disp_change_loss['l1'] = self.disp_change_criterion_l1(pred, ground_truth)

        return disp_change_loss

    def forward(self, x: NestedTensor, y: NestedTensor, disp_change: Tensor, outputs: dict):
        """
        losses = criterion(x, y, data_disp_change.to(device), outputs)
        :param inputs1: input data1
        :param inputs2: input data2
        :param outputs: output from the network, dictionary
        :param disp_change: real disp_change, 1 channel data
        :return: loss dictionary
        """
        loss = {}
        bs, _, _, _ =  x.left.shape
        t = NestedTensor(left=torch.cat([x.left, y.left],dim=0), 
                    right=torch.cat([x.right, y.right], dim=0), disp=torch.cat([x.disp, y.disp], dim=0),
                    sampled_cols=torch.cat([x.sampled_cols, y.sampled_cols], dim=0), 
                    sampled_rows=torch.cat([x.sampled_rows, y.sampled_rows], dim=0), 
                    occ_mask=torch.cat([x.occ_mask, y.occ_mask], dim=0), 
                    occ_mask_right=torch.cat([x.occ_mask_right, y.occ_mask_right], dim=0))

        if self.validation_max_disp == -1:
            invalid_mask = t.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(t.disp <= 0.0, t.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            # print("find useless sample")
            return None

        loss['rr'] = self.compute_rr_loss(outputs['out'], t, invalid_mask)
        loss['l1_raw'] = self.compute_l1_loss(outputs['out']['disp_pred'], t, invalid_mask)
        loss['l1'] = self.compute_l1_loss(outputs['out']['disp_pred'], t, invalid_mask)
        loss['occ_be'] = self.compute_entropy_loss(outputs['out']['occ_pred'], t, invalid_mask)

        self.aggregate_loss(loss)

        loss['disp_change'] = self.generate_disp_change_loss(outputs['disp_change'], disp_change, invalid_mask[bs:])

        # for benchmarking
        self.calc_px_error(outputs['out']['disp_pred'], t.disp, loss, invalid_mask)
        self.compute_epe(outputs['out']['disp_pred'], t.disp, loss, invalid_mask)
        self.compute_iou(outputs['out']['occ_pred'], t.occ_mask, loss, invalid_mask)

        loss['aggregated'] = loss['aggregated'] + loss['disp_change']['l1']

        return OrderedDict(loss)


def build_criterion(args):
    loss_weight = {}
    for weight in args.loss_weight.split(','):
        k, v = weight.split(':')
        k = k.strip()
        v = float(v)
        loss_weight[k] = v

    return Criterion(args.px_error_threshold, args.validation_max_disp, loss_weight)


class CriterionWithoutMotion(nn.Module):
    """
    Compute loss and evaluation metrics
    """

    def __init__(self, threshold: int = 3, validation_max_disp: int = -1, loss_weight: list = None):
        super(CriterionWithoutMotion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.px_threshold = threshold
        self.validation_max_disp = validation_max_disp
        self.weights = loss_weight

        self.l1_criterion = nn.SmoothL1Loss()
        self.epe_criterion = nn.L1Loss()

    @torch.no_grad()
    def calc_px_error(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute px error

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """

        # computing threshold-px error
        loss_dict['error_px'] = torch.sum(
            torch.abs(pred[~invalid_mask] - disp[~invalid_mask]) > self.px_threshold).item()
        loss_dict['total_px'] = torch.sum(~invalid_mask).item()

        return

    @torch.no_grad()
    def compute_epe(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute EPE

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        loss_dict['epe'] = self.epe_criterion(pred[~invalid_mask], disp[~invalid_mask])
        return

    @torch.no_grad()
    def compute_iou(self, pred: Tensor, occ_mask: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute IOU on occlusion

        :param pred: occlusion prediction [N,H,W]
        :param occ_mask: ground truth occlusion mask [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        # threshold
        pred_mask = pred > 0.5

        # iou for occluded region
        inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
        union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()

        # iou for non-occluded region
        inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
        union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()

        # aggregate
        loss_dict['iou'] = (inter_occ + inter_noc).float() / (union_occ + union_noc)

        return

    def compute_rr_loss(self, outputs: dict, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute rr loss

        :param outputs: dictionary, outputs from the network
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: rr loss
        """""
        if invalid_mask is not None:
            if inputs.sampled_cols is not None:
                invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)

        # compute rr loss in non-occluded region
        gt_response = outputs['gt_response']
        eps = 1e-6
        rr_loss = - torch.log(gt_response + eps)

        if invalid_mask is not None:
            rr_loss = rr_loss[~invalid_mask]

        # if there is occlusion
        try:
            rr_loss_occ_left = - torch.log(outputs['gt_response_occ_left'] + eps)
            # print(rr_loss_occ_left.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_left])
        except KeyError:
            pass
        try:
            rr_loss_occ_right = - torch.log(outputs['gt_response_occ_right'] + eps)
            # print(rr_loss_occ_right.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_right])
        except KeyError:
            pass

        return rr_loss.mean()

    def compute_l1_loss(self, pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor, fullres: bool = True):
        """
        compute smooth l1 loss

        :param pred: disparity prediction [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :param fullres: Boolean indicating if prediction is full resolution
        :return: smooth l1 loss
        """
        disp = inputs.disp
        if not fullres:
            if inputs.sampled_cols is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 2, inputs.sampled_cols)
                disp = batched_index_select(disp, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(invalid_mask, 1, inputs.sampled_rows)
                disp = batched_index_select(disp, 1, inputs.sampled_rows)

        return self.l1_criterion(pred[~invalid_mask], disp[~invalid_mask])

    def compute_entropy_loss(self, occ_pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute binary entropy loss on occlusion mask

        :param occ_pred: occlusion prediction, [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: binary entropy loss
        """
        eps = 1e-6

        occ_mask = inputs.occ_mask

        entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
        entropy_loss_noc = - torch.log(
            1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

        entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

        return entropy_loss.mean()

    def aggregate_loss(self, loss_dict: dict):
        """
        compute weighted sum of loss

        :param loss_dict: dictionary of losses
        """
        loss = 0.0
        for key in loss_dict:
            loss += loss_dict[key] * self.weights[key]

        loss_dict['aggregated'] = loss
        return

    def forward(self, inputs1: NestedTensor, inputs2: NestedTensor, outputs: dict):
        """
        losses = criterion(x, y, outputs)
        :param inputs1: input data1
        :param inputs2: input data2
        :param outputs: output from the network, dictionary
        :return: loss dictionary
        """
        loss = {}
        loss1 = {}
        loss2 = {}

        if self.validation_max_disp == -1:
            invalid_mask = inputs1.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(inputs1.disp <= 0.0, inputs1.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            return None

        loss1['rr'] = self.compute_rr_loss(outputs['out_t1'], inputs1, invalid_mask)
        loss1['l1_raw'] = self.compute_l1_loss(outputs['out_t1']['disp_pred'], inputs1, invalid_mask, fullres=True)
        loss1['l1'] = self.compute_l1_loss(outputs['out_t1']['disp_pred'], inputs1, invalid_mask)
        loss1['occ_be'] = self.compute_entropy_loss(outputs['out_t1']['occ_pred'], inputs1, invalid_mask)

        if self.validation_max_disp == -1:
            invalid_mask = inputs2.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(inputs2.disp <= 0.0, inputs2.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            return None

        loss2['rr'] = self.compute_rr_loss(outputs['out_t2'], inputs2, invalid_mask)
        loss2['l1_raw'] = self.compute_l1_loss(outputs['out_t2']['disp_pred'], inputs2, invalid_mask, fullres=True)
        loss2['l1'] = self.compute_l1_loss(outputs['out_t2']['disp_pred'], inputs2, invalid_mask)
        loss2['occ_be'] = self.compute_entropy_loss(outputs['out_t2']['occ_pred'], inputs2, invalid_mask)

        # 先执行原作者的聚集，哈哈
        self.aggregate_loss(loss1)
        self.aggregate_loss(loss2)

        # for benchmarking
        self.calc_px_error(outputs['out_t1']['disp_pred'], inputs1.disp, loss1, invalid_mask)
        self.compute_epe(outputs['out_t1']['disp_pred'], inputs1.disp, loss1, invalid_mask)
        self.compute_iou(outputs['out_t1']['occ_pred'], inputs1.occ_mask, loss1, invalid_mask)

        self.calc_px_error(outputs['out_t2']['disp_pred'], inputs2.disp, loss2, invalid_mask)
        self.compute_epe(outputs['out_t2']['disp_pred'], inputs2.disp, loss2, invalid_mask)
        self.compute_iou(outputs['out_t2']['occ_pred'], inputs2.occ_mask, loss2, invalid_mask)

        loss['loss1'] = loss1
        loss['loss2'] = loss2
        loss['aggregated'] = loss1['aggregated'] * 0.5 + loss2['aggregated'] * 0.5

        return OrderedDict(loss)


def build_criterion_without_motion(args):
    loss_weight = {}
    for weight in args.loss_weight.split(','):
        k, v = weight.split(':')
        k = k.strip()
        v = float(v)
        loss_weight[k] = v

    return CriterionWithoutMotion(args.px_error_threshold, args.validation_max_disp, loss_weight)
