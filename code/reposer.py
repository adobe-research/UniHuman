
import sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import os
from PIL import Image, ImageOps
import os.path as osp
import torch.nn as nn
import torch.nn.functional as Func

import PIL.Image
from scipy.interpolate import griddata
import cv2
from sklearn.neighbors import NearestNeighbors

from dp2coor import dp2coor

from torchvision.transforms import ToPILImage
import pickle
import time


def uv2target(dp, tex,dp_uv_lookup_256_np, tex_mask=None,device=None):
    # dp: target dense pose b,h,w,3(uvi) -> pytorch tensor b,3,h,w
    # tex: uv space texture b,h,w,2 -> pytorch tensor b,3,h,w
    # tex mask: mask of valid texture in uv space - pytorch tensor b,1,h,w
	
    dp=dp.transpose(3,2).transpose(2,1).to(device)
    tex=tex.transpose(3,2).transpose(2,1)
    b,_, h, w = dp.shape
    ch = tex.shape[1]
    UV_MAPs = []
    tex[0,:,0,0]=0

    for idx in range(b):
        iuv = dp[idx]
        point_pos = iuv[2, :, :] > 0
        iuv_raw_i = iuv[2][point_pos]
        iuv_raw_u = iuv[0][point_pos]
        iuv_raw_v = iuv[1][point_pos]
        iuv_raw = torch.stack([iuv_raw_i,iuv_raw_u,iuv_raw_v], 0)
        i = iuv_raw[0, :] - 1 ## dp_uv_lookup_256_np does not contain BG class
        u = iuv_raw[2, :]
        v = iuv_raw[1, :]
        
        # convert to numpy otherwise, shape error
        # works on numpy but fails with tensors
        i = i.long().tolist()#.cpu().numpy().astype(int)
        u = u.long().tolist()#.cpu().numpy().astype(int)
        v = v.long().tolist()#.cpu().numpy().astype(int)
       
        uv_smpl = dp_uv_lookup_256_np[i, v, u]
        uv_map = torch.zeros((2,h,w)).to(tex).float()
        uv_map = uv_map-1
        ## normalize [0,1] to [-1,1] for the grid sample of Pytorch
        u_map = uv_smpl[:, 0] * 2 - 1
        v_map = (1 - uv_smpl[:, 1]) * 2 - 1
        uv_map[0][point_pos] = u_map#torch.from_numpy(u_map).to(tex.float()).float()
        uv_map[1][point_pos] = v_map#torch.from_numpy(v_map).to(tex.float()).float()
        UV_MAPs.append(uv_map)
    uv_map = torch.stack(UV_MAPs, 0)
    # warping
    # before warping validate sizes
    _, _, h_x, w_x = tex.shape
    _, _, h_t, w_t = uv_map.shape
    #if h_t != h_x or w_t != w_x:
    #    uv_map = torch.nn.functional.interpolate(uv_map, size=(h_x, w_x), mode='bilinear', align_corners=True)
    uv_map = uv_map.permute(0, 2, 3, 1)
    warped_image = torch.nn.functional.grid_sample(tex.float(), uv_map.float(),align_corners=False)
    if tex_mask is not None:
        if len(tex_mask.shape)==2:
            tex_mask=torch.from_numpy(tex_mask)[None,None].to(warped_image) #1,1,h,w
        warped_mask = torch.nn.functional.grid_sample(tex_mask.float(), uv_map.float(),align_corners=False)
        warped_mask = torch.gt(warped_mask,0).float()
        final_warped = warped_image * warped_mask
        return final_warped.cpu(), warped_mask.cpu()
    else:
        return warped_image,None

def warp_source(target_pose, texture_map,dp_uv_lookup_256_np,tex_mask=None,device=None):

    warp_im,warped_mask= uv2target(torch.from_numpy(np.expand_dims(target_pose,0)),texture_map[None],dp_uv_lookup_256_np, tex_mask=tex_mask,device=device) # uint8 and float
    
    warp_im =warp_im[0].transpose(0,1).transpose(1,2).numpy()#np.uint8()
    if warped_mask is not None:
        warped_mask=warped_mask[0,0].numpy()
        #warped_mask=Image.fromarray(warped_mask*255) 

    #warp_im = Image.fromarray(warp_im) 

    return warp_im,warped_mask

def pad_PIL(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def pad_numpy(im,top, right, bottom, left, color=0):
    height, width,C = im.shape
    new_width = width + right + left
    new_height = height + top + bottom
    new_im=np.ones((new_height,new_width,C),dtype=im.dtype)*color
    new_im[top:top+height,left:left+width]=im
    return new_im
    

def complete_corr(im, uv_coor, uv_mask, uv_symm_mask, coor_completion_generator,device=None): 

    #im = ImageOps.exif_transpose(im)
    #w, h = im.size
    h,w,_=im.shape

    if (True):
       
        # uv coor
        shift = int((h-w)/2)
        uv_coor[:,:,0] = uv_coor[:,:,0] + shift # put in center
        uv_coor = ((2*uv_coor/(h-1))-1)
        uv_coor = uv_coor*np.expand_dims(uv_mask,2) + (-10*(1-np.expand_dims(uv_mask,2)))

        x1 = shift
        x2 = h-(w+x1)
        #im = pad_PIL(im, 0, x2, 0, x1, color=(0, 0, 0))
        im=pad_numpy(im,0,x2,0,x1,color=0)

        ## coordinate completion
        complete_coor = torch.from_numpy(uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
        
        #im = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float()
        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float()
        rgb_uv = torch.nn.functional.grid_sample(im.to(device), complete_coor.permute(0,2,3,1).to(device),align_corners=False)
        rgb_uv = rgb_uv[0].permute(1,2,0)#.data.cpu().numpy()

        # saving
        #save_image(rgb_uv, os.path.join('final_txmap/complete_%s'%phase, im_name.split('.')[0]+'_txmap.png'))
        #rgb_uv=Image.fromarray(rgb_uv.astype(np.uint8))
        #rgb_uv=rgb_uv.astype(np.uint8)
        #rgb_uv=rgb_uv.long()
        return rgb_uv


class Synthesis():
    def __init__(self,device=torch.device('cuda')):
        super(Synthesis, self).__init__()
        self.mod_type = "2d"
        #self.dp_uv_lookup_256_np = np.load("densepose_data/dp_uv_lookup_256_new.npy")
        self.dp_uv_lookup_256_np=torch.from_numpy(np.load('densepose_data/dp_uv_lookup_256_new.npy').astype(np.float32)).to(device)
        self.device=device


    def synthesize_with_pose(self, pose_target, pose_source, img_source):
        '''
        :param pose_target: dense uv for target, height-by-width-by-channel (channel: i,u,v)
        :param pose_source: dense uv for source, height-by-width-by-channel (channel: u,v,i)(i,u,v)
        :param img_source: height-by-width-by-channel
        :return:
        '''
        #convert pose_target to PIL image
        pose_target = PIL.Image.fromarray(np.uint8(pose_target)).convert('RGB')
        warped_img = []
        generated_results, texturemap, warped_img = self.get_results_2d(pose_target, pose_source, img_source)

        return generated_results, texturemap, warped_img


    def process_data(self, texturemap_source, pose_target, warp_img=None):
        texturemap_source = texturemap_source.resize((512, 512), PIL.Image.BICUBIC)
        texturemap_source = (TF.to_tensor(texturemap_source) - 0.5) / 0.5

        if warp_img is not None:
            warp_img = warp_img.resize((512, 512), PIL.Image.BICUBIC)
            warp_img = (TF.to_tensor(warp_img) - 0.5) / 0.5

        pose_target = pose_target.resize((512, 512), PIL.Image.NEAREST)
        pose_target = torch.Tensor(np.array(pose_target))
        pose_target = torch.transpose(pose_target, 1, 2)
        pose_target = torch.transpose(pose_target, 0, 1)
        pose_target = pose_target / 255
        pose_target[2, :, :] = pose_target[2, :, :] * 10
        pose_target = (pose_target - 0.5) / 0.5
        return texturemap_source, pose_target, warp_img


    def getUVtextureBackward(self, input, iuv, resolution=256):
        '''
        For each pixel in the texture map, finds the closest pixel on the input image based on the iuv labels
        :param input: Input image
        :param iuv: Dense iuv rendering for the input image
        :param resolution: resolution of the texture map
        :return:
        '''
        iuv = iuv.astype(int)  # uint8 -> int64

        texture_path = "densepose_data/%d.png" % resolution
        texture_map =cv2.imread(texture_path)
        texture_map = texture_map.astype(int)
        texture_map = texture_map[:, :, [2, 0, 1]]  # uvi -> iuv

        texture = np.zeros(texture_map.shape)

        thres = 10

        texture_map[:, :, 0] = texture_map[:, :, 0] * thres
        iuv[:, :, 0] = iuv[:, :, 0] * thres

        liuv = iuv.reshape(iuv.shape[0] * iuv.shape[1], 3)
        linp = input.reshape(input.shape[0] * input.shape[1], 3)
        uliuv, iuv_indices = np.unique(liuv, axis=0, return_index=True)

        ltmp = texture_map.reshape(texture_map.shape[0] * texture_map.shape[1], 3)
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(uliuv)
        distances, knn_indices = nbrs.kneighbors(ltmp)

        distances = np.reshape(distances, [texture_map.shape[0], texture_map.shape[1], 4])
        knn_indices = np.reshape(knn_indices, [texture_map.shape[0], texture_map.shape[1], 4])

        tex_mask=np.zeros((texture_map.shape[0],texture_map.shape[1]))

        for p1 in range(texture_map.shape[0]):
            for p2 in range(texture_map.shape[1]):
                if texture_map[p1, p2, 0] == 0:
                    continue

                ct = 0
                for p3 in range(4):
                    if distances[p1, p2, p3] > thres - 1:
                        break
                    texture[p1, texture_map.shape[1]-1-p2, :] = texture[p1, texture_map.shape[1]-1-p2, :] + linp[iuv_indices[knn_indices[p1, p2, 0]], :] * distances[
                        p1, p2, p3]
                    ct = ct + distances[p1, p2, p3]

                if ct > 0:
                    texture[p1, texture_map.shape[1]-1-p2, :] = texture[p1, texture_map.shape[1]-1-p2, :] / ct
                    tex_mask[p1, texture_map.shape[1]-1-p2]=1

        texture = torch.from_numpy(texture).permute(1,0,2).long().cuda()
        tex_mask=np.uint8(np.transpose(tex_mask, (1,0)))
        #texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        #texture_pil = PIL.Image.fromarray(texture)
        return texture,tex_mask


    def repose(self, pose_source, img_source, pose_target):
        # pose_source_pil:vui
        # pose_target:vui
        uv_coor, uv_mask, uv_symm_mask = dp2coor(pose_source) #vui
        rgb_uv = complete_corr(img_source, uv_coor, uv_mask, uv_symm_mask, None,device=self.device)
        warp_im,warp_mask = warp_source(pose_target, rgb_uv,self.dp_uv_lookup_256_np, tex_mask=uv_mask,device=self.device)
        
        return warp_im,warp_mask

    # def repose(self, pose_source, img_source, pose_target):
    #     # pose_source_pil:vui
    #     # pose_target:vui
    #     rgb_uv,tex_mask = self.getUVtextureBackward(img_source, pose_source[..., ::-1], resolution=256)
    #     warp_im,warp_mask = warp_source(pose_target, rgb_uv,tex_mask)
    #     return warp_im,warp_mask

    def mapper(self, iuv, resolution=256):
        H, W, _ = iuv.shape
        iuv_raw = iuv[iuv[:, :, 0] > 0]
        x = np.linspace(0, W - 1, W).astype(np.int)
        y = np.linspace(0, H - 1, H).astype(np.int)
        xx, yy = np.meshgrid(x, y)
        xx_rgb = xx[iuv[:, :, 0] > 0]
        yy_rgb = yy[iuv[:, :, 0] > 0]
        # modify i to start from 0... 0-23
        i = iuv_raw[:, 0] - 1
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]
        uv_smpl = self.dp_uv_lookup_256_np[
            i.astype(np.int),
            v.astype(np.int),
            u.astype(np.int)
        ]
        u_f = uv_smpl[:, 0] * (resolution - 1)
        v_f = (1 - uv_smpl[:, 1]) * (resolution - 1)
        return xx_rgb, yy_rgb, u_f, v_f
