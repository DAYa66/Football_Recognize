from collections.abc import Iterable
import cv2
import numpy as np
from Football_Recognize.utils.math import npsafe_divide


def cut_and_resize(img,bboxes,size=(288,384)):
    res = []
    bboxes = np.array(bboxes).astype(np.int32)
    bboxes = np.maximum(bboxes,0)
    bboxes[...,0::2] = np.minimum(bboxes[...,0::2],img.shape[1])
    bboxes[...,1::2] = np.minimum(bboxes[...,1::2],img.shape[0])
    for bbox in bboxes:
        cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        if size is not None:
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
        res.append(cur_img)
    return np.array(res)

def npchangexyorder(bboxes):
    if len(bboxes)==0:
        return bboxes
    bboxes = np.array(bboxes)
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    data = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return data

def npscale_bboxes(bboxes,scale,correct=False,max_size=None):
    if not isinstance(scale, Iterable):
        scale = [scale,scale]
    bboxes = np.array(bboxes)
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    h = scale[0]*h
    w = scale[1]*w
    ymin = cy - h / 2.
    ymax = cy + h / 2.
    xmin = cx - w / 2.
    xmax = cx + w / 2.
    xmin = np.maximum(xmin,0)
    ymin = np.maximum(ymin,0)
    if max_size is not None:
        xmax = np.minimum(xmax,max_size[1]-1)
        ymax = np.minimum(ymax,max_size[0]-1)
    data = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return data

'''
bbox_ref:[1,4], [[ymin,xmin,ymax,xmax]]
bboxes:[N,4],[[ymin,xmin,ymax,xmax],...]
'''
def npbboxes_jaccard(bbox_ref, bboxes, name=None):

    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = -inter_vol \
                + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
                + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    jaccard = npsafe_divide(inter_vol, union_vol)
    return jaccard

'''
bboxes: [N,4] (ymin,xmin,ymax,xmax)
'''
def npbbxoes_nms(bboxes,nms_thrsh=0.5):
    bboxes_nr = len(bboxes)
    bboxes = np.array(bboxes)
    if bboxes_nr<=1:
        return bboxes,[True]
    mask = np.ones([bboxes_nr],dtype=np.bool)
    for i in range(bboxes_nr-1):
        ious = npbboxes_jaccard([bboxes[i]],bboxes[i+1:])
        for j in range(len(ious)):
            if ious[j]>nms_thrsh:
                mask[i+1+j] = False
    mask = mask.tolist()
    bboxes = bboxes[mask]
    return bboxes,mask


def is_point_in_bbox(p,bbox):
    '''

    Args:
        p: (x,y)
        bbox: (x0,y0,x1,y1)

    Returns:
    '''
    if p[0]>=bbox[0] and p[0]<=bbox[2] and p[1]>=bbox[1] and p[1]<=bbox[3]:
        return True
    return False

'''
bboxes0: [N,4]/[1,4] [ymin,xmin,ymax,xmax)
bboxes1: [N,4]/[1,4]ymin,xmin,ymax,xmax)
return:
[-1,1]
'''
def npgiou(bboxes0, bboxes1):
    # 1. calulate intersection over union
    bboxes0 = np.array(bboxes0)
    bboxes1 = np.array(bboxes1)
    area_1 = (bboxes0[..., 2] - bboxes0[..., 0]) * (bboxes0[..., 3] - bboxes0[..., 1])
    area_2 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    intersection_wh = np.minimum(bboxes0[..., 2:], bboxes1[..., 2:]) - np.maximum(bboxes0[..., :2], bboxes1[..., :2])
    intersection_wh = np.maximum(intersection_wh, 0)

    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = (area_1 + area_2) - intersection

    ious = intersection / np.maximum(union, 1e-10)

    # 2. (C - (A U B))/C
    C_wh = np.maximum(bboxes0[..., 2:], bboxes1[..., 2:]) - np.minimum(bboxes0[..., :2], bboxes1[..., :2])
    C_wh = np.maximum(C_wh, 1e-10)
    C = C_wh[..., 0] * C_wh[..., 1]

    giou = ious - (C - union) /C
    return giou


'''
box0:[N,4], or [1,4],[ymin,xmin,ymax,xmax],...
box1:[N,4], or [1,4]
return:
[N],返回box0,box1交叉面积占box0的百分比
'''
def npbboxes_intersection_of_box0(box0,box1):

    bbox_ref= np.transpose(box0)
    bboxes = np.transpose(box1)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    jaccard = npsafe_divide(inter_vol, union_vol)
    return jaccard


