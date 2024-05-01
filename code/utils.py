import numpy as np
from PIL import Image
import random
import os

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import cv2

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def _3dts2np(ts):
    ts=ts.permute(1,2,0)
    arr=((ts+1)*127.5).numpy().astype(np.uint8)
    return arr
    
def _pil2nts(pil):
    ts=torch.from_numpy(np.array(pil)).float().permute(2,0,1)
    ts=ts/127.5-1
    return ts
    
def _np2nts(arr):
    ts=torch.from_numpy(arr).float().permute(2,0,1)
    ts=ts/127.5-1
    return ts
    
def add_dimension(dict,dim):
    for k in dict.keys():
        if type(dict[k])==str:
            dict[k]=[dict[k]]
        else:
            dict[k]=dict[k].unsqueeze(dim)
    return dict

def mask2bbox(mask):
    assert len(mask.shape)==2
    result=np.zeros_like(mask) 
    
    ys,xs=np.nonzero(mask)
    xmin, ymin, xmax, ymax=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
    xmin, ymin, xmax, ymax=int(xmin), int(ymin), int(xmax), int(ymax)    
    result[ymin:ymax,xmin:xmax]=1.0
    return result
    
def __crop(img, pos):
    x1, y1, tw, th = pos
    return img.crop((x1, y1, x1 + tw, y1 + th))

def get_resize_params(size,intSize=256):
    w, h = size
    if type(intSize)==tuple:
        h=intSize[0]
        w=intSize[1]
    else:    
        fixed_scale = intSize/max(w,h)
        w = round(fixed_scale*w)
        h = round(fixed_scale*h)
        
        assert w==intSize or h==intSize    
    new_w = w 
    new_h = h 
    return {'crop_param': (0, 0, w, h), 'scale_size':(new_h, new_w)}    



def get_transform(param, method=Image.BICUBIC, normalize=True, toTensor=True,constant=0):
    transform_list = []
    if 'scale_size' in param and param['scale_size'] is not None:
        osize = param['scale_size']
        transform_list.append(transforms.Resize(osize, interpolation=method))

    if 'crop_param' in param and param['crop_param'] is not None:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, param['crop_param'])))
        length=max(param['crop_param'][-1],param['crop_param'][-2])
        pad_t=(length-param['crop_param'][-1])//2
        pad_l=(length-param['crop_param'][-2])//2
        pad_b=length-pad_t-param['crop_param'][-1]
        pad_r=length-pad_l-param['crop_param'][-2]
        transform_list.append(transforms.Pad((pad_l,pad_t,pad_r,pad_b),fill=constant))
    
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
       transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)