# Copyright (c) OpenMMLab. All rights reserved.
# Modified from jbwang1997: https://github.com/jbwang1997/OBBDetection
import mmcv
import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
import math
from ..builder import ROTATED_BBOX_CODERS
from ..transforms import obb2poly, obb2xyxy, poly2obb,xy_wh_r_2_xy_sigma,sigma_xy_wh_r_2_xy,xy_wh_r_2_xy_sigm_sqr

from ..transforms import norm_angle





@ROTATED_BBOX_CODERS.register_module()
class GaussianCoder(BaseBBoxCoder):
    """Mid point offset coder. This coder encodes bbox (x1, y1, x2, y2) into \
    delta (dx, dy, dw, dh, da, db) and decodes delta (dx, dy, dw, dh, da, db) \
    back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.,  0.),
                 target_stds=(1., 1., 1., 1.,  1.),
                 angle_range='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.version = angle_range

    def encode(self, bboxes, gt_bboxes,complex_w=None,complex_h=None):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert gt_bboxes.size(-1) == 5

        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                                        self.version)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `bboxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, 5) or (N, 5).
                Note N = num_anchors * W * H when rois is a grid of anchors.

            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip=wh_ratio_clip)

        return decoded_bboxes
    def decode2gau(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `bboxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, 5) or (N, 5).
                Note N = num_anchors * W * H when rois is a grid of anchors.

            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2gau(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip, self.version)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               version='oc',complex_w=None,complex_h=None):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h, a, b of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: deltas with shape (N, 6), where columns represent dx, dy,
            dw, dh, da, db.
    """
    # proposals x1 y1 x2 y2 gt:x,y,w,h,cita
    proposals = proposals.float()
    gt = gt.float()
    gt_gaussian = xy_wh_r_2_xy_sigm_sqr(gt)
    if proposals.shape[-1]==4:
        flag = 2
    if proposals.shape[-1]==5:
        flag = 1
    if flag == 2:
        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]
        sc=1
    if flag == 1:
        px = proposals[..., 0]
        py = proposals[..., 1]
        pw = proposals[..., 2]
        ph = proposals[..., 3]
        pc = proposals[..., 4]
        # sc = 1.5 * torch.sigmoid(pc)
        sc=1
    # print(pc.min(),pc.max())
    ggama1 = gt_gaussian[1][..., 0, 0]
    ggama2 = gt_gaussian[1][..., 1, 1]
    ggama3 = gt_gaussian[1][..., 0, 1]

    hbb = obb2xyxy(gt, version)
    gx = (hbb[..., 0] + hbb[..., 2]) * 0.5
    gy = (hbb[..., 1] + hbb[..., 3]) * 0.5
    dgama1 = (ggama1 / ((pw * ph)*sc)).log()
    dgama2 = (ggama2 / ((pw * ph)*sc)).log()
    dgama3 = (ggama3 / ((pw * ph)*sc))
    dx = (gx - px) / pw
    dy = (gy - py) / ph



    deltas = torch.stack([dx, dy, dgama1, dgama2, dgama3], dim=-1)


    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    isinf = torch.all(torch.isinf(deltas))



    return deltas


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               wh_ratio_clip=16 / 1000,
               version='oc'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """

    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::5]
    dy = denorm_deltas[..., 1::5]
    dgama1 = denorm_deltas[..., 2::5]
    dgama2 = denorm_deltas[..., 3::5]
    dgama3 = denorm_deltas[..., 4::5]

    if rois.shape[-1]==4:
        flag = 'double'
    if rois.shape[-1]==5:
        flag = 'single'
    if flag=='double':

        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)


        # dx = (gx - px) / pw
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1)
        sc=1
    if flag=='single':
        px = rois[:, 0].unsqueeze(1).expand_as(dx)
        py = rois[:, 1].unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = rois[:, 2].unsqueeze(1).expand_as(dx)
        ph = rois[:, 3].unsqueeze(1).expand_as(dx)
        pc = rois[:, 4].unsqueeze(1).expand_as(dx)
        sc=1


    gx = px +pw*dx
    gy = py +ph*dy

    gama1 = ((dgama1).exp()* pw * ph*sc)
    gama2 = ((dgama2).exp()* (pw * ph))*(sc)
    gama3 = ((dgama3)* (pw * ph))*(sc)

    colc = torch.stack((gama1, gama3, gama3, gama2), dim=-1).reshape(-1,2,2)

    center_gau = ( torch.stack([gx, gy ], dim=-1).reshape(-1,2),colc)

    with torch.no_grad():
        cente_obb=sigma_xy_wh_r_2_xy(center_gau,rois,version)


    return cente_obb.view(deltas.size())

@mmcv.jit(coderize=True)
def delta2gau(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               wh_ratio_clip=16 / 1000,
               version='oc'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """


    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dgama1 = denorm_deltas[:, 2::5]
    dgama2 = denorm_deltas[:, 3::5]
    dgama3 = denorm_deltas[:, 4::5]

    if rois.shape[-1]==4:
        flag = 'double'
    if rois.shape[-1]==5:
        flag = 'single'
    if flag=="double":
        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1)
        sc=ph*0+1
    if flag=='single':
        px = rois[:, 0].unsqueeze(1).expand_as(dx)
        py = rois[:, 1].unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = rois[:, 2].unsqueeze(1).expand_as(dx)
        ph = rois[:, 3].unsqueeze(1).expand_as(dx)
        pc = rois[:, 4].unsqueeze(1).expand_as(dx)
        sc=1

    pw=pw.clamp(1e-5,1e+5)
    ph=ph.clamp(1e-5,1e+5)

    gx = px +pw*dx
    gy = py +ph*dy

    gama1 = ((dgama1).exp()*(pw * ph))*(sc)
    gama2 = ((dgama2).exp()*(pw * ph))*(sc)
    gama3 = ((dgama3)* (pw * ph))*(sc)

    colc = torch.stack((gama1, gama3, gama3, gama2), dim=-1).reshape(-1,2,2)


    center_gau = ( torch.stack([gx, gy ], dim=-1).reshape(-1,2),colc)

    return center_gau


