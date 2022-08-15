#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np
import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "siou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            s_cw = (c_br-c_tl)[:, 0]
            s_ch = (c_br-c_tl)[:, 1]
            cw = target[:, 0] - pred[:, 0]
            ch = target[:, 1] - pred[:, 1]
            sigma = torch.pow(cw ** 2 + ch ** 2, 0.5)
            sin_alpha = torch.abs(ch) / sigma
            sin_beta = torch.abs(cw) / sigma
            thres = torch.pow(torch.tensor(2.), 0.5) / 2
            sin_alpha = torch.where(sin_alpha < thres, sin_alpha, sin_beta)
            angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            gamma = angle_cost - 2
            rho_x = (cw / s_cw) ** 2
            rho_y = (ch / s_ch) ** 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w_pred = pred[:, 2]
            h_pred = pred[:, 3]
            W_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
            W_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)
            shape_cost = torch.pow(1 - torch.exp(-1 * W_w), 4) + torch.pow(1 - torch.exp(-1 * W_h), 4)
            siou = iou - (distance_cost + shape_cost) * 0.5
            loss = 1 - siou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
