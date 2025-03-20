# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
import torch

def obb2poly_np_le90_points_18(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    # try:
    # center, w, h, theta = torch.split(obboxes, (2, 3, 4), axis=-1)
    center = obboxes[..., :2].reshape(-1, 2)
    w = obboxes[..., 2].reshape(-1, 1)
    h = obboxes[..., 3].reshape(-1, 1)
    theta = obboxes[..., 4].reshape(-1, 1)
    # except:  # noqa: E722
    #     results = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]).cuda()
    #     return results.reshape(1, -1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector1 = torch.cat([w / 2 * Cos, w / 2 * Sin], axis=-1).reshape(-1, 2)
    vector2 = torch.cat([-h / 2 * Sin, h / 2 * Cos], axis=-1).reshape(-1, 2)
    point1 = center - (vector1 + vector2)
    point2 = center + (vector1 - vector2)
    point3 = center + (vector1 + vector2)
    point4 = center - (vector1 - vector2)
    point5 = (point1 + point2)/2
    point6 = (point2 + point3)/2
    point7 = (point3 + point4)/2
    point8 = (point4 + point1)/2
    polys = torch.cat([point1, point2, point3, point4,point5,point6,point7,point8,center], axis=-1)
    # polys = get_best_begin_point(polys)
    return polys

def obb2poly_np_le90_points_8(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    # try:
    # center, w, h, theta = torch.split(obboxes, (2, 3, 4), axis=-1)
    center = obboxes[..., :2].reshape(-1, 2)
    w = obboxes[..., 2].reshape(-1, 1)
    h = obboxes[..., 3].reshape(-1, 1)
    theta = obboxes[..., 4].reshape(-1, 1)
    # except:  # noqa: E722
    #     results = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]).cuda()
    #     return results.reshape(1, -1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector1 = torch.cat([w / 2 * Cos, w / 2 * Sin], axis=-1).reshape(-1, 2)
    vector2 = torch.cat([-h / 2 * Sin, h / 2 * Cos], axis=-1).reshape(-1, 2)
    point1 = center - (vector1 + vector2)
    point2 = center + (vector1 - vector2)
    point3 = center + (vector1 + vector2)
    point4 = center - (vector1 - vector2)

    polys = torch.cat([point1, point2, point3, point4], axis=-1)
    # polys = get_best_begin_point(polys)
    return polys

def sigma_xy_wh_r_2_xy(xy_sigma,rois,version="oc"):
    xy = xy_sigma[0]
    sigma = xy_sigma[1]
    # eigenvalues, eigenvectors = torch.eig(sigma, eigenvectors=True)
    # R,S,R_inv = torch.svd(sigma)
    # pw = (rois[:, 2]).unsqueeze(1)
    # ph = (rois[:, 3]).unsqueeze(1)
    gama1 = sigma[..., 0, 0].reshape(-1,1)
    gama2 = sigma[..., 1, 1].reshape(-1,1)
    gama3 = sigma[..., 0, 1].reshape(-1,1)
    # pi025 = (torch.ones([gama1.shape[0]]).cuda()*math.pi*(1/4)).reshape(-1,1)
    d_gama1_2 = (gama1-gama2)
    d_gama1and2 = gama1+gama2
    # e_7 = torch.tensor([1e-7]).cuda()
    # d_gama1_2 = torch.where(d_gama1_2 == 0, e_7, d_gama1_2)
    alpha=d_gama1_2/(gama3+1e-7)
    T_tan=-0.5*(alpha-torch.sqrt(alpha*alpha+4))
    # raw_cita = 0.5*(torch.arctan(2*gama3/d_gama1_2))
    raw_cita=torch.arctan(T_tan)
    # print(raw_cita.min(),raw_cita.max())
    raw_cita = raw_cita.nan_to_num(0.0005).clamp(0.00005,1.572)
    zero=torch.zeros_like(gama1)+1e-7
    half_pi=torch.zeros_like(gama1)+1.565
    # one=torch.zeros_like(gama1)+1e-2
    raw_cita=torch.where(gama1<0.05 ,zero,raw_cita)
    # print(gama1.min(),gama1.max())
    change=(raw_cita<1e-5)*(gama1>1e-2)

    raw_cita=torch.where(change,half_pi,raw_cita)

    # print(raw_cita.isnan().any())
    # raw_cita = torch.where(gama3 ==1.5708, cita0, raw_cita)    # raw_cita = norm_angle(raw_cita, version)

    flag='sqr'
    if flag == "sqr":
        trans="new2"
        if trans=="old":
            a=d_gama1and2
            b=(1*gama3)/(raw_cita.cos()*raw_cita.sin())
            c=(1*d_gama1_2)/(raw_cita.cos()*raw_cita.cos()-raw_cita.sin()*raw_cita.sin())
            # a = 8*gama3/torch.sin(2*raw_cita)
            # b = 4*d_gama1and2

            # ab = a+b
            # ab=torch.where(ab<0,-2,1)
            # ab_r = -a+b
            # ab_r=torch.where(ab_r<0,-2,1)
            raw_w =torch.sqrt(torch.abs(2*a+b+c))
            raw_h =torch.sqrt(torch.abs((2*a-b-c)))
            # complex_w = torch.where((a+b)<=0,-1.,1.)
            # # complex_w = complex_w*raw_w
            # complex_h = torch.where((-a+b)<=0,-1.,1.)
        elif trans=="new1":
            # a=d_gama1and2*d_gama1and2-d_gama1_2*d_gama1_2-4*gama3*gama3
            a=gama1*gama2-gama3*gama3
            b=d_gama1and2
            # raw_cita=torch.where(change,half_pi,raw_cita)
            # w_long=(d_gama1_2>0)*((raw_cita.cos()*raw_cita.cos()-raw_cita.sin()*raw_cita.sin())>0)
            # squ=(gama3<zero)*((raw_cita.sin()*raw_cita.cos()).abs()>zero)*((b*b-4*a)<zero)
            w_long_1=(d_gama1_2>0)*((raw_cita.cos()*raw_cita.cos()-raw_cita.sin()*raw_cita.sin())>0)
            w_long_2= (d_gama1_2<0)*((raw_cita.cos()*raw_cita.cos()-raw_cita.sin()*raw_cita.sin())<0)
            w_long=w_long_1+w_long_2
            x=0.5*(b+(b*b-4*a).clamp(0).sqrt())
            y=0.5*(b-(b*b-4*a).clamp(0).sqrt())
            maxwh=torch.where(x>y,x,y)
            minwh=torch.where(x<=y,x,y)

            raw_w=torch.where(w_long,maxwh,minwh)
            raw_h = torch.where(w_long,minwh,maxwh)
            raw_w=raw_w.abs().sqrt()*2
            raw_h=raw_h.abs().sqrt()*2
        elif trans=="new2":
            _shape = raw_cita.shape
            r=raw_cita
            cos_r = torch.cos(r)
            sin_r = torch.sin(r)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            RT=R.permute(0, 2,1)
            S= RT.bmm(sigma).bmm(R)

            raw_w = S[..., 0, 0].reshape(-1,1)
            raw_h = S[..., 1, 1].reshape(-1,1)
            # print(S[..., 0, 1].reshape(-1,1))
            raw_w=raw_w.abs().sqrt()*2
            raw_h=raw_h.abs().sqrt()*2
    elif flag=='single':

        a = d_gama1and2

        b = (d_gama1_2)/torch.cos(2*raw_cita)
        c = (2*gama3)/torch.sin(2*raw_cita)
        ab = a+b
        ac = a+c
        # raw_w = torch.abs(0.5*(a+b))+torch.abs(0.5*(a+c))
        # raw_h = torch.abs( 0.5*(-a + b)) + torch.abs(0.5*(-a + c))
        raw_w = torch.abs(0.5*(a+b))+torch.abs(0.5*(a+c)).clamp(min=1e-7,max=256)
        raw_h = torch.abs( 0.5*(a-c)) + torch.abs(0.5*(a-b)).clamp(1e-7,max=256)
        complex_w = torch.where((a+b)<=0,-1.,1.)
        complex_h = torch.where((-a+b)<=0,-1.,1.)





    # raw_w=torch.abs(2*gama1)
    # raw_h=torch.abs(2*gama2)
    # raw_w = torch.where(gama3==0,torch.abs(2*gama1),raw_w)
    # raw_h = torch.where(gama3==0,torch.abs(2*gama2),raw_h)

    # complex_h = complex_h*raw_h
    # print(torch.all((a+b)>0),torch.all((-a+b)>0))
    # raw_w = torch.sqrt(ab*0.5*(a+b))
    # raw_h = torch.sqrt(ab_r*0.5*(-a+b))

    # raw_w = torch.where(raw_cita!=math.pi*(1/4),raw_w, torch.sqrt(2*gama1+2*gama3))
    # raw_h = torch.where(raw_cita!=math.pi*(1/4),raw_h, torch.sqrt(2*gama1-2*gama3))
    # rev_5_ = torch.full_like(raw_w, 5)
    # print(pw[torch.where(pw<0)])
    # rev_w = torch.where(torch.isnan(rev_w), rev_5_, rev_w)
    # rev_h = torch.where(torch.isnan(rev_h), rev_5_, rev_h)
    rev_w = torch.nan_to_num(raw_w,1e-7)
    rev_h = torch.nan_to_num(raw_h,1e-7)
    # rev_w=torch.where(squ,maxwh,rev_w)
    # rev_h=torch.where(squ,maxwh,rev_h)
    # rev_w = torch.where(torch.isnan(raw_w), 0.7*pw, raw_w)
    # rev_h = torch.where(torch.isnan(raw_h), 0.7*ph, raw_h)
    # rev_w = torch.where(rev_w<0, pw, rev_w)
    # rev_h = torch.where(rev_h<0, ph, rev_h)

    # rev_w = torch.where(rev_w<0, rev_5_, rev_w)
    # rev_h = torch.where(rev_h<0, rev_5_, rev_h)

    raw_x=xy[:,0].reshape(-1,1)
    raw_y = xy[:,1].reshape(-1,1)

    # print(torch.any(rev_w<0),torch.any(rev_h<0))


    # rev_w = torch.where(raw_w_0==5,raw_h_0,raw_w_0).reshape(-1,1)
    # rev_h = torch.where(raw_h_0==5,raw_w_0,raw_h_0).reshape(-1,1)
    wh = torch.cat((rev_w, rev_h), axis=1)
    if version == 'le90' or version == 'le135':
        rev_cita = torch.where(raw_cita < 0, raw_cita + 0.5 * math.pi, raw_cita - 0.5 * math.pi).reshape(-1, 1)
        # wh = torch.nan_to_num(wh,nan=1e+01)
        wh_sor, indices  = torch.sort(wh,descending=True)
        cita_a_b = torch.cat((rev_cita,raw_cita),axis=1)
        out_cita = torch.sum((cita_a_b * indices),dim=1).reshape(-1,1)
        # out_com_w = torch.sum((complex_w * indices),dim=1).reshape(-1,1)
        # out_com_w = torch.where(rev_w>rev_h,complex_w,complex_h)
        # out_com_h = torch.sum((complex_h * indices),dim=1).reshape(-1,1)
        # out_com_h = torch.where(rev_w > rev_h, complex_h, complex_w)



        pre_out = torch.cat((raw_x,raw_y,wh_sor,out_cita),axis=1)
    if version == 'oc':
        # pos_cita = torch.where(raw_cita>0,1,0).reshape(-1,1)
        # neg_cita = 1-pos_cita
        # out_w=torch.where(raw_w > raw_h, raw_w, raw_h)
        # out_h = torch.where(raw_w > raw_h, raw_h, raw_w)
        # hw = torch.cat((rev_h, rev_w), axis=1)
        # wh_sor = wh * pos_cita + neg_cita * hw
        # out_cita = torch.where(raw_cita >0, raw_cita, raw_cita+0.5*math.pi).reshape(-1, 1)
        # raw_cita=torch.where(change,half_pi,raw_cita)
        # rev_w=torch.where(change,rev_h,rev_w)
        # rev_h=torch.where(change,rev_w,rev_h)
        # print(raw_cita.min(),raw_cita.max())
        # print(rev_w.min(),rev_w.max())
        # print(rev_h.min(),rev_h.max())
        pre_out = torch.cat((raw_x, raw_y, rev_w,rev_h, raw_cita), axis=1)
    if version == 'le135':
        obb90=obb2poly_le90(pre_out)
        pre_out=poly2obb(obb90, version='le135')
    # print(torch.any(wh_sor<0))
    # print(torch.all(pre_out[:,2]>=pre_out[:,3]))
    # if torch.any(wh_sor<0):
    #     print('pl')
    # out_com_w = torch.where(rev_w > rev_h, complex_w, complex_h)
    # # out_com_h = torch.sum((complex_h * indices),dim=1).reshape(-1,1)
    # out_com_h = torch.where(rev_w > rev_h, complex_h, complex_w)
    return pre_out


def gaussian2bbox_s(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    # try:
    #     from torch_batch_svd import svd
    # except ImportError:
    #     svd = None
    L = 3
    var = gmm[1]
    mu = gmm[0]

    # assert mu.size()[1:] == (1, 2)
    # assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    # if svd is None:
    #     raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = torch.svd(var)
    x=mu[:,0].reshape(-1,1)
    y=mu[:,1].reshape(-1,1)
    w=s[:,0].reshape(-1,1)
    h=s[:,1].reshape(-1,1)
    w=torch.nan_to_num(w,5)
    h=torch.nan_to_num(h,5)
    cita=torch.arccos(U[:,0,0].sqrt()).reshape(-1,1)
    # size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    # mu = mu.repeat(1, 4, 1)
    #dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=torch.float32,device=size_half.device)
    # bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)
    bboxes=torch.stack((x,y,w*2,h*2,cita),dim=0)

    return bboxes


def sigma5_xy_wh_r_2_xy(xy_sigma,version):

    raw_x = xy_sigma[:, 0::5]
    raw_y = xy_sigma[:, 1::5]
    gama1 = xy_sigma[:, 2::5]
    gama2 = xy_sigma[:, 3::5]
    gama3 = xy_sigma[:, 4::5]

    # pi025 = (torch.ones([gama1.shape[0]]).cuda()*math.pi*(1/4)).reshape(-1,1)
    d_gama1_2 = (gama1-gama2)
    d_gama1and2 = gama1+gama2
    e_7 = torch.tensor([1e-7]).cuda()
    # d_gama1_2 = torch.where(d_gama1_2 == 0, e_7, d_gama1_2)
    raw_cita = 0.5*(torch.arctan((2*gama3)/d_gama1_2))

    a = 8*gama3/torch.sin(2*raw_cita)
    b = 4*d_gama1and2
    raw_w =torch.sqrt(0.5*(a+b))
    raw_h = torch.sqrt(0.5*(-a+b))

    # raw_w = torch.where(raw_cita!=math.pi*(1/4),raw_w, torch.sqrt(2*gama1+2*gama3))
    # raw_h = torch.where(raw_cita!=math.pi*(1/4),raw_h, torch.sqrt(2*gama1-2*gama3))
    # rev_5_ = torch.full_like(raw_w, 5)
    # print(pw[torch.where(pw<0)])
    # rev_w = torch.where(torch.isnan(rev_w), rev_5_, rev_w)
    # rev_h = torch.where(torch.isnan(rev_h), rev_5_, rev_h)
    rev_w = torch.nan_to_num(raw_w,5)
    rev_h = torch.nan_to_num(raw_h,5)
    # rev_w = torch.where(torch.isnan(raw_w), 0.7*pw, raw_w)
    # rev_h = torch.where(torch.isnan(raw_h), 0.7*ph, raw_h)
    # rev_w = torch.where(rev_w<0, pw, rev_w)
    # rev_h = torch.where(rev_h<0, ph, rev_h)

    # rev_w = torch.where(rev_w<0, rev_5_, rev_w)
    # rev_h = torch.where(rev_h<0, rev_5_, rev_h)

    # raw_x=xy[:,0].reshape(-1,1)
    # raw_y = xy[:,1].reshape(-1,1)

    # print(torch.any(rev_w<0),torch.any(rev_h<0))


    # rev_w = torch.where(raw_w_0==5,raw_h_0,raw_w_0).reshape(-1,1)
    # rev_h = torch.where(raw_h_0==5,raw_w_0,raw_h_0).reshape(-1,1)
    wh = torch.cat((rev_w, rev_h), axis=1)
    if version == 'le90':
        rev_cita = torch.where(raw_cita < 0, raw_cita + 0.5 * math.pi, raw_cita - 0.5 * math.pi).reshape(-1, 1)
        # wh = torch.nan_to_num(wh,nan=1e+01)
        wh_sor, indices  = torch.sort(wh,descending=True)
        cita_a_b = torch.cat((rev_cita,raw_cita),axis=1)
        out_cita = torch.sum((cita_a_b * indices),dim=1).reshape(-1,1)



        pre_out = torch.cat((raw_x,raw_y,wh_sor,out_cita),axis=1)
    if version == 'oc':
        # pos_cita = torch.where(rev_cita>0,1,0).reshape(-1,1)
        # neg_cita = 1-pos_cita
        # hw = torch.cat((rev_h, rev_w), axis=1)
        # wh_sor = wh * pos_cita + neg_cita * hw
        # out_cita = torch.where(rev_cita >0, rev_cita, rev_cita+0.5*math.pi).reshape(-1, 1)

        pre_out = torch.cat((raw_x, raw_y, wh_sor, out_cita), axis=1)
    # print(torch.any(wh_sor<0))
    # print(torch.all(pre_out[:,2]>=pre_out[:,3]))

    return pre_out

def xy_wh_r_2_xy_sigm_sqr(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma
def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S).bmm(R.permute(0, 2,1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma
def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    version = 'oc'
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    if version == 'oc':
        rotated_flag = (bboxes[:, 4] != np.pi / 2)
        flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    else:
        flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], version)
    return flipped


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes[:, :4] = new_bboxes[:, :4] / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def poly2obb(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_oc(polys)
    elif version == 'le135':
        results = poly2obb_le135(polys)
    elif version == 'le90':
        results = poly2obb_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def obb2hbb(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    if version == 'oc':
        results = obb2hbb_oc(rbboxes)
    elif version == 'le135':
        results = obb2hbb_le135(rbboxes)
    elif version == 'le90':
        results = obb2hbb_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2xyxy(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    if version == 'oc':
        results = obb2xyxy_oc(rbboxes)
    elif version == 'le135':
        results = obb2xyxy_le135(rbboxes)
    elif version == 'le90':
        results = obb2xyxy_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def hbb2obb(hbboxes, version='oc'):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = hbb2obb_oc(hbboxes)
    elif version == 'le135':
        results = hbb2obb_le135(hbboxes)
    elif version == 'le90':
        results = hbb2obb_le90(hbboxes)
    else:
        raise NotImplementedError
    return results


def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_le135(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le135')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[..., 0]
    y = rboxes[..., 1]
    w = rboxes[..., 2]
    h = rboxes[..., 3]
    a = rboxes[..., 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2hbb_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,pi/2]
    """
    w = rbboxes[..., 2::5]
    h = rbboxes[..., 3::5]
    a = rbboxes[..., 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    hbboxes = rbboxes.clone().detach()
    hbboxes[..., 2::5] = hbbox_h
    hbboxes[..., 3::5] = hbbox_w
    hbboxes[..., 4::5] = np.pi / 2
    return hbboxes


def obb2hbb_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = bboxes.new_zeros(bboxes.size(0))
    inds = edges1 < edges2
    rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    rotated_boxes[inds, 2] = edges2[inds]
    rotated_boxes[inds, 3] = edges1[inds]
    rotated_boxes[inds, 4] = np.pi / 2.0
    return rotated_boxes


def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack([_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_oc(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    rbboxes = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    return rbboxes


def hbb2obb_le135(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_le90(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def obb2xyxy_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    # pi/2 >= a > 0, so cos(a)>0, sin(a)>0
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def obb2xyxy_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    N = rotatex_boxes.shape[0]
    if N == 0:
        return rotatex_boxes.new_zeros((rotatex_boxes.size(0), 4))
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def obb2xyxy_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # N = obboxes.shape[0]
    # if N == 0:
    #     return obboxes.new_zeros((obboxes.size(0), 4))
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)





def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    try:
        from torch_batch_svd import svd
    except ImportError:
        svd = None
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    if svd is None:
        raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = svd(var)
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes


def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))
