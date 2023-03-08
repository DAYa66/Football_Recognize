import os
import copy
import cv2
import numpy as np


def imwrite(filename, img,size=None):
    '''
    size: (W,H)
    '''
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    dir_path = os.path.dirname(filename)
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(img.shape)==3 and img.shape[2]==3:
        img = copy.deepcopy(img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
    if size is not None:
        img = resize_img(img,size=size,keep_aspect_ratio=True)
    cv2.imwrite(filename, img)


'''
size:(w,h)
return:
 resized img, resized_img.size <= size
'''
def resize_img(img,size,keep_aspect_ratio=False,interpolation=cv2.INTER_LINEAR,align=None):

    img_shape = img.shape
    if size[0] == img.shape[1] and size[1]==img.shape[0]:
        return img

    if np.any(np.array(img_shape)==0):
        img_shape = list(img_shape)
        img_shape[0] = size[1]
        img_shape[1] = size[0]
        return np.zeros(img_shape,dtype=img.dtype)
    if keep_aspect_ratio:
        if size[1]*img_shape[1] != size[0]*img_shape[0]:
            if size[1]*img_shape[1]>size[0]*img_shape[0]:
                ratio = size[0]/img_shape[1]
            else:
                ratio = size[1]/img_shape[0]
            size = list(copy.deepcopy(size))
            size[0] = int(img_shape[1]*ratio)
            size[1] = int(img_shape[0]*ratio)

            if align:
                size[0] = (size[0]+align-1)//align*align
                size[1] = (size[1] + align - 1) // align * align

    if not isinstance(size,tuple):
        size = tuple(size)
    if size[0]==img_shape[0] and size[1]==img_shape[1]:
        return img

    img = cv2.resize(img,dsize=size,interpolation=interpolation)

    if len(img_shape)==3 and len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    
    return img

def resize_height(img,h,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_h = h
    new_w = int(shape[1]*new_h/shape[0])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)
