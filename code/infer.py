
import numpy as np
from PIL import Image
import os
import os.path as osp
import json
import cv2
import torchvision.transforms as T
from utils import get_resize_params,get_transform,mask2bbox,_3dts2np,_pil2nts,_np2nts,add_dimension,set_seed
import torch
import math
import argparse
from tqdm import tqdm

from mmpose_utils import tryon_cloth_warp
from unihuman import UniHuman


# Define dino preprocessing 
dino_processor=T.Compose(
               [T.Resize((224,224), interpolation=T.InterpolationMode.BICUBIC),
               T.ToTensor(),
               T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
               )


print('Initializing UniHuman model...')
model=UniHuman()

def predict_parsing(img,parser):
    # accept pil image, return pil image
    parsing,vis=parser.predict_parsing(img)
    if np.sum(np.array(parsing))==0:
        print('Warning: Failed to detect person in the image')
    return parsing,vis

def predict_pose(img,pose_detector):
    # return pil image
    pose=pose_detector.predict_pose(img)
    if pose is None:
        print('Warning: Failed to detect person in the image')
    return pose

def predict_clothes_mmpose(img,mmpose_detector,clothes_cat='Upper Clothing'):
    # return dict 
    result=mmpose_detector.predict_clothes_mmpose(img,clothes_cat=clothes_cat)
    return result

def predict_person_mmpose(img,mmpose_detector):
    # return dict 
    result=mmpose_detector.predict_human_mmpose(img)
    return result

def predict_clothes_mask(img,parser):
    # return pil image ranging in [0,1]
    parsing,vis=parser.predict_clothes_parsing(img)
    if np.sum(np.array(parsing))==0:
        print('Warning: Failed to detect clothing item in the image')
        parsing=Image.fromarray(np.ones_like(parsing))
    return parsing,vis

def get_pwarped_tex_for_pt(src_image,src_pose,tgt_pose,uv_tex_warper):

    source_img_np=np.array(src_image)

    src_pose_cvted=np.array(src_pose)[...,[1,0,2]]#vui
    tgt_pose_cvted=np.array(tgt_pose)[...,[1,0,2]]

    tgt_img,tgt_map=uv_tex_warper.repose(src_pose_cvted,source_img_np,tgt_pose_cvted)
    tgt_img=tgt_img.astype(np.uint8)
    tgt_map=tgt_map.astype(np.uint8)
    return tgt_img,tgt_map

def get_pwarped_tex_for_to(person_img,parsing,target_pose,clothes,clothes_mask,src_mmpose,cl_mmpose,tryon_cat):

    tgt_img,tgt_map=tryon_cloth_warp(person_img,parsing,target_pose,clothes,clothes_mask,cl_mmpose,src_mmpose,tryon_cat)
    
    if np.sum(tgt_map)==0:
        print('Warning: Failed to find matched keypoints between the clothes and the subject')
    
    return tgt_img,tgt_map
    
def get_tgt_pose_mask(tgt_pose_pil):
    
    pose=np.array(tgt_pose_pil)[...,-1]
    pose_mask=np.not_equal(pose,0).astype(np.uint8)
    
    # Expand mask in case of fluffy clothes
    pose_mask=cv2.dilate(pose_mask,np.ones((64*pose_mask.shape[0]//256,64*pose_mask.shape[1]//256), np.uint8))
    pose_mask=mask2bbox(pose_mask)
    
    return pose_mask
    
# Parsing labels    
#LABEL=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat','Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm','Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
 
def find_background(person_img,src_pose,tgt_pose,parsing_pil,task,tryon_cat):
    person_img=np.array(person_img)
    src_pose=np.array(src_pose)
    tgt_pose=np.array(tgt_pose)
    parsing=np.array(parsing_pil)
    
    assert person_img.shape==src_pose.shape==tgt_pose.shape
    assert parsing.shape[:2]==person_img.shape[:2]
    
    
    ksize=person_img.shape[0]//64
    
    # Process parsing
    if task=='Pose Transfer':
        fg_mask=np.not_equal(parsing,0) 
    elif task=='Virtual Try-on' or task=='Text Editing':
        if tryon_cat=='Upper Clothing':
            fg_indices=[5,6,7,10,14,15]
        elif tryon_cat=='Lower Clothing':
            fg_indices=[8,9,12,16,17]
        elif tryon_cat=='Dress':
            fg_indices=[5,6,7,8,9,10,12,14,15,16,17]
        else:
            raise NotImplementedError
        fg_mask=0
        for ind in fg_indices:
            fg_mask+=np.equal(parsing,ind) 
    
    parsing_fg_mask=cv2.dilate(fg_mask.astype(np.uint8),np.ones((ksize,ksize), np.uint8))
    parsing_utch_mask=(1-fg_mask)*np.not_equal(parsing,0).astype(np.uint8)
    
    # Process pose
    if task=='Pose Transfer':
        src_pose=src_pose[...,2] # body part indices
        tgt_pose=tgt_pose[...,2]
        fg_mask=np.logical_or(np.not_equal(src_pose,0),np.not_equal(tgt_pose,0))
    elif task=='Virtual Try-on' or task=='Text Editing':
        pose=np.array(src_pose)[...,2] # body part indices
        if tryon_cat=='Upper Clothing':
            fg_indices=[15,16,17,18,19,20,21,22]#arms
        elif tryon_cat=='Lower Clothing':
            fg_indices=[7,8,9,10,11,12,13,14]#legs
        elif tryon_cat=='Dress':
            fg_indices=[15,16,17,18,19,20,21,22,1,2,7,8,9,10,11,12,13,14]#body
        else:
            raise NotImplementedError
        fg_mask=0
        for ind in fg_indices:
            fg_mask+=np.equal(pose,ind) 
    # Expand pose fg mask
    if task=='Pose Transfer':
        fg_mask+=np.roll(fg_mask,(ksize,ksize),(0,1))
        fg_mask+=np.roll(fg_mask,(ksize,-ksize),(0,1))
        fg_mask+=np.roll(fg_mask,(-ksize,ksize),(0,1))
        fg_mask+=np.roll(fg_mask,(-ksize,-ksize),(0,1))
        
        psize=fg_mask.shape[0]//16
        h,w=fg_mask.shape
        nblocks1=h//psize
        nblocks2=w//psize
        fg_mask=fg_mask.reshape(nblocks1,psize,nblocks2,psize)
        fg_mask+=np.greater(fg_mask.sum((1,3),keepdims=True),0)
        pose_fg_mask=fg_mask.reshape(h,w)
    else:
        pose_fg_mask=cv2.dilate(fg_mask.astype(np.uint8),np.ones((ksize,ksize), np.uint8))
    
    fg_mask=np.greater(pose_fg_mask+parsing_fg_mask,0).astype(np.uint8)#
    utcd_mask=cv2.erode(parsing_utch_mask.astype(np.uint8),np.ones((ksize,ksize), np.uint8))
    bg_mask=1-fg_mask+utcd_mask*fg_mask
 
    bg=person_img*bg_mask[...,None]
    return bg,bg_mask
    
    
            
def get_texture(image,parsing,task=None,clothes=None,tryon_cat=None):

    # Resize image to dino input size
    image=image.resize((224,224),resample=Image.BICUBIC)
    parsing=cv2.resize(parsing,(224,224),interpolation = cv2.INTER_NEAREST)
    
    if task=='Pose Transfer':
         excl_indices=[0]
    elif task=='Virtual Try-on' or task=='Text Editing':
        if tryon_cat=='Upper Clothing':
            excl_indices=[0,5,6,7,10,14,15]
        elif tryon_cat=='Lower Clothing':
            excl_indices=[0,8,9,10,12,16,17]
        elif tryon_cat=='Dress':
            excl_indices=[0,5,6,7,8,9,10,12,14,15,16,17]
        else:
            raise NotImplementedError
    else:
         raise NotImplementedError
        
    excl_mask=0
    for ind in excl_indices:
        excl_mask+=np.equal(parsing,ind) 
    excl_mask=cv2.dilate(excl_mask.astype(np.uint8),np.ones((5,5), np.uint8))
    person_mask=1-excl_mask[...,None]
         
    person_tex=np.array(image)*person_mask+255*(1-person_mask)
    person_tex=Image.fromarray(person_tex)
    person_tex_ts=dino_processor(person_tex)

    part_names = ['face', 'hair', 'headwear', 'top', 'outer', 'bottom', 'shoes', 'accesories','person']
    
    tex_lst=[]
    for name in part_names:
        if task=='Virtual Try-on':
            if tryon_cat=='Upper Clothing' and name=='top':
                 tex=dino_processor(clothes)
            elif tryon_cat=='Dress' and name=='top':
                 tex=dino_processor(clothes)
            elif tryon_cat=='Lower Clothing' and name=='bottom':
                 tex=dino_processor(clothes)
            else:
                tex=person_tex_ts
        else:
            tex=person_tex_ts
        tex_lst.append(tex)
    tex_ts=torch.stack(tex_lst,0)
    return tex_ts
    
def make_prompt(parsing):
    # Auto generated prompt based on parsing result
    parsing=np.array(parsing)
    clothing_dict={
            'hat':[1],
            'sunglasses':[4],
            'mask':[36],
            'eyeglasses':[37],
             'upper clothes':[5],
            'dress':[6],
            'jumpsuite':[10],
            'coat':  [7],
            'pants': [9],
            'skirt':[12],
            'socks':[8],
            'shoes': [18,20],
            'gloves':[3],
             'scarf':[11],
        }
    prompt='a person'
    lst=[]
    for name in clothing_dict.keys():
        inds=clothing_dict[name]
        mask=0
        for ind in inds:
            mask+=np.equal(parsing,ind).sum()
        if mask>parsing.shape[0]*0.05*parsing.shape[1]*0.05:
             lst.append(name)
    if len(lst)>0:
        prompt+=' wearing '+','.join(lst)
    else:
        prompt=''
    return prompt

def take_apart_seg(parsing,task=None,clothes_mask=None,tryon_cat=None):
    # return array
    parsing=cv2.resize(parsing,(224,224),interpolation = cv2.INTER_NEAREST)
    part_dict = {
            #'background':  [0],
            'hair':        [2],
            'face':        [13],
            'headwear':    [1,4],
            'top':         [5, 6],
            'outer':       [7],
            'bottom':      [9, 12],
            'shoes':       [8,18,19],
            'accesories':  [3, 11]
        }
    part_names = ['face', 'hair', 'headwear', 'top', 'outer', 'bottom', 'shoes', 'accesories']
    mask_lst=[]
    for i,name in enumerate(part_names):
        ind_lst=part_dict[name]
        mask=0
        for ind in ind_lst:
            mask+=np.equal(parsing,ind)
        if task=='Virtual Try-on':
            if tryon_cat=='Upper Clothing' and name=='top':
                 mask=np.array(clothes_mask.resize((224,224),resample=Image.NEAREST))
            elif tryon_cat=='Dress' and name=='top':
                 mask=np.array(clothes_mask.resize((224,224),resample=Image.NEAREST))
            elif tryon_cat=='Dress' and name=='bottom':
                 mask=np.zeros_like(mask)
            elif tryon_cat=='Lower Clothing' and name=='bottom':
                 mask=np.array(clothes_mask.resize((224,224),resample=Image.NEAREST))
            else:
                 pass     
        elif task=='Text Editing':
            if tryon_cat=='Upper Clothing' and name=='top':
                 mask=np.zeros_like(mask)
            elif tryon_cat=='Dress' and name=='top':
                 mask=np.zeros_like(mask)
            elif tryon_cat=='Dress' and name=='bottom':
                 mask=np.zeros_like(mask)
            elif tryon_cat=='Lower Clothing' and name=='bottom':
                 mask=np.zeros_like(mask)
            else:
                 pass  
        else:
            pass    
        mask_lst.append(mask.astype(np.float32))

    mask_lst.append(np.not_equal(parsing,0)) # human mask
    mask_lst=np.stack(mask_lst,0)   
    return mask_lst    

def composite_imgs(img1,img2,mask1):
    # return numpy array.
    img1=img1.astype(float)
    img2=img2.astype(float)
    smask1=cv2.erode(mask1,np.ones((3,3), np.uint8))
    dmask=np.clip(mask1-smask1,0,1)
    img=img1*smask1[...,None]+(img1**0.5*img2**0.5)*dmask[...,None]+img2*((1-smask1-dmask)[...,None])
    img=img.clip(0,255).astype(np.uint8)  
    return img

def process(task,seed,src_image,tgt_image,clothes,prompt,tryon_cat,edit_cat,ug_scale):
    
    # Set random seed
    set_seed(seed)
    
    # Set image resolution
    res=512
    
    # Sanity check
    if task=='Pose Transfer' and tgt_image is None:
        raise Exception('Please upload a target image for pose trasnfer!')

    if task=='Virtual Try-on' and clothes is None:
        raise Exception('Please upload a garment image for the virtual try-on task!')

    if task=='Text Editing' and len(prompt)==0:
        raise Exception('Please give a prompt!')
    
    src_image = src_image.convert('RGB')
    
    ### Source image preprocessing begins. ### 
    parsing,vis_parsing=predict_parsing(src_image,model.parser)
    src_pose=predict_pose(src_image,model.pose_detector)
    
        
    src_param = get_resize_params(src_pose.size,intSize=res)
    src_img_trans_to_tensor = get_transform(src_param, normalize=True, toTensor=True,constant=255)
    src_trans_to_image = get_transform(src_param, normalize=False, toTensor=False,constant=0, method=Image.NEAREST)
    src_trans_to_image_fill = get_transform(src_param, normalize=False, toTensor=False,constant=255, method=Image.NEAREST)
    src_pose_resized=src_trans_to_image(src_pose)
    src_pose_ts=_pil2nts(src_pose_resized)
    src_image_resized=src_trans_to_image_fill(src_image)
    src_image_ts=src_img_trans_to_tensor(src_image)
    
    assert src_image_ts.shape==(3,res,res)
    
    param_parsing = get_resize_params(parsing.size,intSize=res)
    trans_parsing = get_transform(param_parsing, normalize=False, toTensor=False,constant=0, method=Image.NEAREST)
    parsing_resized=trans_parsing(parsing)
    
    ### Source image preprocessing ends. ### 


    ### Pose transfer data preprocessing begins. ### 
    if task=="Pose Transfer":
        tgt_pose=predict_pose(tgt_image,model.pose_detector)

        # prepare inputs to the model
        prompt=make_prompt(parsing)
        
        tgt_param = get_resize_params(tgt_pose.size,intSize=res)
        tgt_img_trans_to_tensor = get_transform(tgt_param, normalize=True, toTensor=True,constant=255)
        tgt_trans_to_image = get_transform(tgt_param, normalize=False, toTensor=False,constant=0, method=Image.NEAREST)
        tgt_pose_resized=tgt_trans_to_image(tgt_pose)
        tgt_pose_ts=_pil2nts(tgt_pose_resized)

        pwarped_tex_img_np,pwarped_tex_mask= get_pwarped_tex_for_pt(src_image,src_pose,tgt_pose,model.uv_tex_warper) #h,W,4
        pwarped_tex_img=tgt_img_trans_to_tensor(Image.fromarray(pwarped_tex_img_np)) 
        pwarped_tex_mask=tgt_trans_to_image(Image.fromarray(pwarped_tex_mask))
        pwarped_tex_mask=torch.from_numpy(np.array(pwarped_tex_mask)).float().unsqueeze(0)
    
        pwarped_tex_img=pwarped_tex_img*pwarped_tex_mask
        pwarped_tex_ts=torch.cat([pwarped_tex_img,pwarped_tex_mask],0)

    ### Pose transfer data preprocessing ends. ### 



    ### Virtual Tryon data preprocessing begins. ### 
    clothes_mask=None
    if task=="Virtual Try-on":
        clothes=clothes.convert('RGB')
        
        clothes_mmpose=predict_clothes_mmpose(clothes,model.mmpose_detector,clothes_cat=tryon_cat)
        clothes_mask,_=predict_clothes_mask(clothes,model.parser)

        # prepare inputs to the model
        src_mmpose=predict_person_mmpose(src_image,model.mmpose_detector)
        prompt=make_prompt(parsing)
        tgt_pose=src_pose
        tgt_pose_ts=src_pose_ts
        tgt_pose_resized=src_pose_resized

        pwarped_tex_img_np,pwarped_tex_mask= get_pwarped_tex_for_to(src_image,parsing,tgt_pose,clothes,clothes_mask,src_mmpose,clothes_mmpose,tryon_cat=tryon_cat)
        pwarped_tex_img=src_img_trans_to_tensor(Image.fromarray(pwarped_tex_img_np)) 
        pwarped_tex_mask=src_trans_to_image(Image.fromarray(pwarped_tex_mask))
        pwarped_tex_mask=torch.from_numpy(np.array(pwarped_tex_mask)).float().unsqueeze(0)
        pwarped_tex_img=pwarped_tex_img*pwarped_tex_mask
        pwarped_tex_ts=torch.cat([pwarped_tex_img,pwarped_tex_mask],0)
        
    ### Virtual Tryon data preprocessing ends. ### 

    ### Text Edit data preprocessing ends. ### 
    if task=="Text Editing" :
        prompt=prompt
        tgt_pose=src_pose
        tgt_pose_ts=src_pose_ts
        tgt_pose_resized=src_pose_resized

        pwarped_tex_img,pwarped_tex_mask=torch.zeros_like(tgt_pose_ts),torch.zeros_like(tgt_pose_ts)[:1]
        pwarped_tex_img_np=pwarped_tex_img.permute(1,2,0).numpy()
        pwarped_tex_ts=torch.cat([pwarped_tex_img,pwarped_tex_mask],0)
        
    ### Text Editdata preprocessing ends. ### 

    part_masks=take_apart_seg(np.array(parsing),task=task,clothes_mask=clothes_mask,tryon_cat=tryon_cat if task=='Virtual Try-on' else edit_cat) #9x224x224
    part_masks_ts=torch.from_numpy(part_masks).float()
    src_tex_ts=get_texture(src_image,np.array(parsing),task=task,clothes=clothes,tryon_cat=tryon_cat if task=='Virtual Try-on' else edit_cat) # 9x224x224
    
    bg_img_np,bg_mask=find_background(src_image_resized,src_pose_resized,tgt_pose_resized,parsing_resized,task,tryon_cat if task=='Virtual Try-on' else edit_cat)
    bg_img_ts=_np2nts(bg_img_np)
    bg_mask_ts=torch.from_numpy(bg_mask).unsqueeze(0).float()
    bg_img_ts=bg_img_ts*bg_mask_ts
    bg_ts=torch.cat([bg_img_ts,bg_mask_ts],0)
    
    tgt_pose_mask_np=get_tgt_pose_mask(tgt_pose_resized)
    tgt_pose_mask_ts=torch.from_numpy(tgt_pose_mask_np).unsqueeze(0).float()
    
    
    prompt=prompt+', best quality'
    batch=dict(src=src_image_ts.permute(1,2,0),txt=prompt, hint_pose=tgt_pose_ts,hint_tex=src_tex_ts,pwarped_tex=pwarped_tex_ts,seg=part_masks_ts,bg=bg_ts,human_area_mask=tgt_pose_mask_ts)
    batch=add_dimension(batch,0)
    
    
    results=model.edit_human(batch,ug_scale=ug_scale,task=task,ddim_steps=50)
    result=_3dts2np(results['samples_cfg'][0].cpu()).clip(0,255)
    result=composite_imgs(np.array(src_image_resized),result,bg_mask)
    
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-img-list",     type=str,  default='')
    parser.add_argument("--tgt-img-list",     type=str,  default='')
    parser.add_argument("--tgt-clothes-list",     type=str,  default='')
    parser.add_argument("--prompt-list",     type=str,  default='')
    parser.add_argument("--task",     type=str,  default='')
    parser.add_argument("--out-dir",     type=str,  default='./results')
    parser.add_argument("--tryon-cat",     type=str,  default='upper')
    parser.add_argument("--edit-cat",     type=str,  default='upper')
    parser.add_argument("--seed",     type=int,  default=12345)
    parser.add_argument("--cfg_scale",     type=int,  default=2)
    opt = parser.parse_args()
    
    if opt.task=='reposing':
        src_img_paths=[]
        with open(opt.src_img_list,'r') as src_file:
            for line in src_file.readlines():
                if osp.exists(line.strip()):
                     src_img_paths.append(line.strip())
                else:
                     print('File %s not exists!'%line)
          
        tgt_img_paths=[]
        with open(opt.tgt_img_list,'r') as tgt_file:
            for line in tgt_file.readlines():
                if osp.exists(line.strip()):
                     tgt_img_paths.append(line.strip())
                else:
                     print('File %s not exists!'%line)
                     
        assert len(src_img_paths)>0 and len(tgt_img_paths)>0, print('Empty file list!')
        assert len(src_img_paths)==len(tgt_img_paths), print('The number of source images should be equal to the number of target images!')
        

        os.makedirs(opt.out_dir,exist_ok=True)
        for src_path,tgt_path in tqdm(zip(src_img_paths,tgt_img_paths)):
              src_image=Image.open(src_path)
              tgt_image=Image.open(tgt_path)
              out_name=src_path.split('/')[-1]+'_to_'+tgt_path.split('/')[-1]+'.png'
              
              result=process('Pose Transfer',opt.seed,src_image,tgt_image,clothes=None,prompt=None,tryon_cat=None,edit_cat=None,ug_scale=opt.cfg_scale)
              
              Image.fromarray(result).save(osp.join(opt.out_dir,out_name))
              
    elif opt.task=='tryon':
        src_img_paths=[]
        with open(opt.src_img_list,'r') as src_file:
            for line in src_file.readlines():
                if osp.exists(line.strip()):
                     src_img_paths.append(line.strip())
                else:
                     print('File %s not exists!'%line)
          
        tgt_clothes_paths=[]
        with open(opt.tgt_clothes_list,'r') as tgt_file:
            for line in tgt_file.readlines():
                if osp.exists(line.strip()):
                     tgt_clothes_paths.append(line.strip())
                else:
                     print('File %s not exists!'%line)
                     
        assert len(src_img_paths)>0 and len(tgt_clothes_paths)>0, print('Empty file list!')
        assert len(src_img_paths)==len(tgt_clothes_paths), print('The number of source images should be equal to the number of clothing images!')
        
        category_mapping={'upper':'Upper Clothing','dress':'Dress','lower':'Lower Clothing'}
        
        os.makedirs(opt.out_dir,exist_ok=True)
        
        for src_path,tgt_cpath in tqdm(zip(src_img_paths,tgt_clothes_paths)):
              src_image=Image.open(src_path)
              tgt_cloth_image=Image.open(tgt_cpath)
              out_name=src_path.split('/')[-1]+'_to_'+tgt_cpath.split('/')[-1]+'.png'
              
              result=process('Virtual Try-on',opt.seed,src_image,tgt_image=None,clothes=tgt_cloth_image,prompt=None,tryon_cat=category_mapping[opt.tryon_cat],edit_cat=None,ug_scale=opt.cfg_scale)
              
              Image.fromarray(result).save(osp.join(opt.out_dir,out_name))
              
    elif opt.task=='text_edit':
        src_img_paths=[]
        with open(opt.src_img_list,'r') as src_file:
            for line in src_file.readlines():
                if osp.exists(line.strip()):
                     src_img_paths.append(line.strip())
                else:
                     print('File %s not exists!'%line)
          
        prompt_list=[]
        with open(opt.prompt_list,'r') as tgt_file:
            for line in tgt_file.readlines():
                prompt_list.append(line.strip())
                     
        assert len(src_img_paths)>0 and len(prompt_list)>0, print('Empty file list!')
        assert len(src_img_paths)==len(prompt_list), print('The number of source images should be equal to the number of prompts!')
        
        category_mapping={'upper':'Upper Clothing','dress':'Dress','lower':'Lower Clothing'}
        
        os.makedirs(opt.out_dir,exist_ok=True)
        
        for src_path,prompt in tqdm(zip(src_img_paths,prompt_list)):
              src_image=Image.open(src_path)
              out_name=src_path.split('/')[-1]+'.png'
              
              result=process('Text Editing',opt.seed,src_image,tgt_image=None,clothes=None,prompt=prompt,tryon_cat=None,edit_cat=category_mapping[opt.edit_cat],ug_scale=opt.cfg_scale)
              
              Image.fromarray(result).save(osp.join(opt.out_dir,out_name))
    else:
        raise NotImplementedError

