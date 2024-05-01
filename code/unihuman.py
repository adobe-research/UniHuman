from cldm.model import create_model, load_state_dict
import os
import torch

from reposer import Synthesis
from parser import HumanParser
from pose_predictor import PosePredictor
from mmpose_utils import MMPosePredictor

class UniHuman():
    def __init__(self):
        
        model = create_model('./configs/unihuman512.yaml').cpu()
        ckpt_path = os.path.join('./models/model512_iters=235999.ckpt')
        model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
        self.model=model.cuda()
        
        print('Initializing pose warping module...')
        self.uv_tex_warper=Synthesis(device=torch.device('cpu'))
        
        print('Initializing human parser...')
        self.parser=HumanParser()

        print('Initializing DensePose detector...')
        self.pose_detector=PosePredictor()

        print('Initializing MMPose detector...')
        self.mmpose_detector=MMPosePredictor()
        
    def edit_human(self,batch,ug_scale=2.0,ddim_steps=50,task='Pose Transfer'):
        
        results=self.model.log_images(batch,N=len(batch),sample=False,unconditional_guidance_scale=ug_scale,ddim_steps=ddim_steps,\
                task=task)
        return results
        
if __name__=='__main__':
    
    img_size=512
    src_image_ts=torch.randn((1,3,img_size,img_size))
    prompt=['photo']
    tgt_pose_ts=src_image_ts
    src_tex_ts=torch.randn((1,9,3,224,224))
    pwarped_tex_ts=torch.concat([torch.randn((1,3,img_size,img_size)),torch.randint(0,2,(1,1,img_size,img_size)).float()],1)
    mask=torch.randint(0,18,(1,224,224))
    part_masks_ts=[]
    for i in range(9):
        part_masks_ts.append(torch.eq(mask,i).float())
    part_masks_ts=torch.stack(part_masks_ts,1)
    bg_ts=pwarped_tex_ts
    tgt_pose_mask_ts=torch.randint(0,2,(1,1,224,224)).float()
    batch=dict(src=src_image_ts.permute(0,2,3,1),txt=prompt, hint_pose=tgt_pose_ts,hint_tex=src_tex_ts,pwarped_tex=pwarped_tex_ts,seg=part_masks_ts,bg=bg_ts,human_area_mask=tgt_pose_mask_ts)
    
    model=UniHuman()
    results=model.edit_human(batch)