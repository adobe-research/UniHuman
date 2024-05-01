from mmpose.apis import MMPoseInferencer
import numpy as np
from PIL import Image
import cv2

class MMPosePredictor():
    def __init__(self):
        self.human_inferencer = MMPoseInferencer('human')
        self.clothes_inferencer_upper = MMPoseInferencer(pose2d='configs/my_custom_config_upper.py',pose2d_weights='models/upper_mmpose.pth',)
        self.clothes_inferencer_lower = MMPoseInferencer(pose2d='configs/my_custom_config_lower.py',pose2d_weights='models/lower_mmpose.pth',)
        self.clothes_inferencer_dress = MMPoseInferencer(pose2d='configs/my_custom_config_dress.py',pose2d_weights='models/dress_mmpose.pth',)
        
    def predict_human_mmpose(self,img_pil):
        img=np.array(img_pil.convert('RGB'))
        result_generator = self.human_inferencer(img)
        result=next(result_generator)
        result=result['predictions']
        
        for instance in result[0]:
            instance['bbox']=[float(x) for x in instance['bbox'][0]]
            instance['keypoints']=[[float(x) for x in row] for row in instance['keypoints'] ]
            instance['keypoint_scores']=[float(x) for x in instance['keypoint_scores']]
            instance['bbox_score']=float(instance['bbox_score'])
            
        if len(result):
            return result[0]
        else:
            return None
            
    def predict_clothes_mmpose(self,img_pil,clothes_cat='Upper Clothing'):
        img=np.array(img_pil.convert('RGB'))
        if clothes_cat=='Upper Clothing':
            result_generator = self.clothes_inferencer_upper(img)
        elif clothes_cat=='Lower Clothing':
            result_generator = self.clothes_inferencer_lower(img)
        elif clothes_cat=='Dress':
            result_generator = self.clothes_inferencer_dress(img)
        else:
            raise NotImplementedError
            
        result=next(result_generator)
        result=result['predictions']
        
        for instance in result[0]:
            instance['bbox']=[float(x) for x in instance['bbox'][0]]
            instance['keypoints']=[[float(x) for x in row] for row in instance['keypoints'] ]
            instance['keypoint_scores']=[float(x) for x in instance['keypoint_scores']]
            instance['bbox_score']=float(instance['bbox_score'])

        if len(result):
            return result[0]
        else:
            return None
            
def warp_cloth_torso_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose):

     cloth_img=np.array(cloth_img)
     cloth_mask=np.array(cloth_mask)
     warped_cloth=np.zeros((target_shape[0],target_shape[1],3),dtype=cloth_img.dtype)
     warped_mask=np.zeros((target_shape[0],target_shape[1]),dtype=cloth_img.dtype)
     
     kpts=np.array(cloth_pose[0]['keypoints']) #17 by 2
     scores=np.array(cloth_pose[0]['keypoint_scores'])
     if scores[5]>0.3 and scores[6]>0.3 and scores[11]>0.3 and scores[12]>0.3:
        if kpts[5][0]>=kpts[6][0]:
            lshoulder=np.array(kpts[5])
            rshoulder=np.array(kpts[6])
        else:
            lshoulder=np.array(kpts[6])
            rshoulder=np.array(kpts[5])
        if kpts[11][0]>=kpts[12][0]:
            lhip=np.array(kpts[11])
            rhip=np.array(kpts[12])
        else:
            lhip=np.array(kpts[12])
            rhip=np.array(kpts[11])

        ratio=0.9
        lshoulder=(lshoulder*ratio+lhip*(1-ratio))
        rshoulder=(rshoulder*ratio+rhip*(1-ratio))
        cloth_cor=np.array([lshoulder,rshoulder,lhip,rhip]).astype(np.float32)

        max_x=int(np.max(np.array(cloth_cor)[:,0]))
        min_x=int(np.min(np.array(cloth_cor)[:,0]))
        max_y=int(np.max(np.array(cloth_cor)[:,1]))
        min_y=int(np.min(np.array(cloth_cor)[:,1]))
        mask=np.zeros((cloth_img.shape[0],cloth_img.shape[1]),dtype=np.uint8)
        mask[min_y:max_y,min_x:max_x]=1
        mask=mask*cloth_mask
        kpts=np.array(person_pose[0]['keypoints']) #17 by 2
        scores=np.array(person_pose[0]['keypoint_scores'])
        if scores[5]>0.3 and scores[6]>0.3 and scores[11]>0.3 and scores[12]>0.3:
                lshoulder=np.array(kpts[5])
                rshoulder=np.array(kpts[6])
                lhip=np.array(kpts[11])
                rhip=np.array(kpts[12])

                lshoulder=(lshoulder*ratio+lhip*(1-ratio))
                rshoulder=(rshoulder*ratio+rhip*(1-ratio))
                person_cor=np.array([lshoulder,rshoulder,lhip,rhip]).astype(np.float32)

                M = cv2.getPerspectiveTransform(cloth_cor, person_cor)
                warped_cloth=cv2.warpPerspective(cloth_img, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
                warped_mask=cv2.warpPerspective(mask, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
                warped_cloth=warped_cloth*warped_mask[:,:,None]       
                        
                        
     return warped_cloth,warped_mask

def limp_to_rec(limpA,limpB,alpha=0.25):
    segment = limpB - limpA
    normal = np.array([-segment[1],segment[0]])
    a = limpA + alpha*normal
    b = limpA - alpha*normal
    c = limpB - alpha*normal
    d = limpB + alpha*normal
    rectangle = np.float32([a,b,c,d])
    return rectangle

def get_limps_coords_and_mask(kpts,scores,limp_indices,ratio=0.9,mask=None,adjust_lr=False,alpha=0.25):
    lu,ru,lb,rb=limp_indices
    
    if adjust_lr and scores[lu]>0.3 and scores[ru]>0.3:
        if kpts[lu][0]<=kpts[ru][0]:
                t=kpts[lu].copy()
                kpts[lu]=kpts[ru].copy()
                kpts[ru]=t
    if adjust_lr and scores[lb]>0.3 and scores[rb]>0.3:
        if kpts[lb][0]<=kpts[rb][0]:
                t=kpts[lb].copy()
                kpts[lb]=kpts[rb].copy()
                kpts[rb]=t 

    if scores[lu]>0.3 and scores[lb]>0.3:  
        lhip=np.array(kpts[lu])
        lknee=np.array(kpts[lb])
        lhip=(lhip*ratio+lknee*(1-ratio))
                
        lhip2knee=limp_to_rec(lhip,lknee,alpha=alpha)
               
        if mask is None:
            lhip2knee_mask=None
        else:
            max_x=int(np.max(np.array(lhip2knee)[:,0]))
            min_x=int(np.min(np.array(lhip2knee)[:,0]))
            max_y=int(np.max(np.array(lhip2knee)[:,1]))
            min_y=int(np.min(np.array(lhip2knee)[:,1]))
            lhip2knee_mask=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
            lhip2knee_mask[min_y:max_y,min_x:max_x]=1
            lhip2knee_mask=lhip2knee_mask*mask
    else:
        lhip2knee=None
        lhip2knee_mask=None


    if scores[ru]>0.3 and scores[rb]>0.3:
        rhip=np.array(kpts[ru])
        rknee=np.array(kpts[rb])
        rhip=(rhip*ratio+rknee*(1-ratio))
        rhip2knee=limp_to_rec(rhip,rknee,alpha=alpha)

        if mask is None:
            rhip2knee_mask=None
        else:
            max_x=int(np.max(np.array(rhip2knee)[:,0]))
            min_x=int(np.min(np.array(rhip2knee)[:,0]))
            max_y=int(np.max(np.array(rhip2knee)[:,1]))
            min_y=int(np.min(np.array(rhip2knee)[:,1]))
            rhip2knee_mask=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
            rhip2knee_mask[min_y:max_y,min_x:max_x]=1
            rhip2knee_mask=rhip2knee_mask*mask
    else:
        rhip2knee=None
        rhip2knee_mask=None
    return lhip2knee,lhip2knee_mask,rhip2knee,rhip2knee_mask
        

def warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=0.9,sep_left_right=False,alpha=0.25):
    cloth_img=np.array(cloth_img)
    cloth_mask=np.array(cloth_mask)
    warped_cloth=np.zeros((target_shape[0],target_shape[1],3),dtype=cloth_img.dtype)
    warped_mask=np.zeros((target_shape[0],target_shape[1]),dtype=cloth_img.dtype)
    warped_cloth2=np.zeros_like(warped_cloth)
    warped_mask2=np.zeros_like(warped_mask) 

    kpts=np.array(cloth_pose[0]['keypoints']) #17 by 2
    scores=np.array(cloth_pose[0]['keypoint_scores'])
    cloth_lhip2knee,cloth_lhip2knee_mask,cloth_rhip2knee,cloth_rhip2knee_mask= \
       get_limps_coords_and_mask(kpts,scores,limp_indices,ratio=ratio,mask=cloth_mask,adjust_lr=True,alpha=alpha)
                
    kpts=np.array(person_pose[0]['keypoints']) #17 by 2
    scores=np.array(person_pose[0]['keypoint_scores'])
    person_lhip2knee,_,person_rhip2knee,_= \
         get_limps_coords_and_mask(kpts,scores,limp_indices,ratio=ratio,mask=None,adjust_lr=False,alpha=alpha)

    if cloth_lhip2knee is not None and person_lhip2knee is not None:
            M = cv2.getPerspectiveTransform(cloth_lhip2knee, person_lhip2knee)
            warped_cloth=cv2.warpPerspective(cloth_img, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
            warped_mask=cv2.warpPerspective(cloth_lhip2knee_mask, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
            warped_cloth=warped_cloth*warped_mask[:,:,None]
                        
    if cloth_rhip2knee is not None and person_rhip2knee is not None: 
            M = cv2.getPerspectiveTransform(cloth_rhip2knee, person_rhip2knee)
            warped_cloth2=cv2.warpPerspective(cloth_img, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
            warped_mask2=cv2.warpPerspective(cloth_rhip2knee_mask, M, (target_shape[1],target_shape[0]), borderMode = cv2.BORDER_REPLICATE)
            warped_cloth2=warped_cloth2*warped_mask2[:,:,None]

            if not sep_left_right:
                warped_cloth+=(1-warped_mask[:,:,None])*warped_cloth2
                warped_mask+=(1-warped_mask)*warped_mask2
                
    if not sep_left_right:                   
         return warped_cloth,warped_mask
    else:
         return warped_cloth,warped_mask,warped_cloth2,warped_mask2


def warp_cloth_legs_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,is_dress=False):
    limp_indices=[11,12,13,14]
    warped_cloth,warped_mask=\
    warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=0.9)

    if not is_dress:
        limp_indices=[13,14,15,16]
        warped_cloth2,warped_mask2=\
        warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=1.0)
    
        warped_cloth+=(1-warped_mask[:,:,None])*warped_cloth2
        warped_mask+=(1-warped_mask)*warped_mask2
    
    return warped_cloth,warped_mask

def warp_cloth_legs_to_person_left_right(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose):
    limp_indices=[11,12,13,14]
    warped_clothl,warped_maskl,warped_clothr,warped_maskr=\
    warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=0.9,sep_left_right=True)

   
    limp_indices=[13,14,15,16]
    warped_cloth2l,warped_mask2l,warped_cloth2r,warped_mask2r=\
warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=1.0,sep_left_right=True)

    warped_clothl+=(1-warped_maskl[:,:,None])*warped_cloth2l
    warped_maskl+=(1-warped_maskl)*warped_mask2l
    warped_clothr+=(1-warped_maskr[:,:,None])*warped_cloth2r
    warped_maskr+=(1-warped_maskr)*warped_mask2r
     
    return warped_clothl,warped_maskl,warped_clothr,warped_maskr

def warp_cloth_arms_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose):
    limp_indices=[5,6,7,8]
    warped_clothl,warped_maskl,warped_clothr,warped_maskr=\
    warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=1.0,sep_left_right=True,alpha=0.2)
    
    limp_indices=[7,8,9,10]
    warped_cloth2l,warped_mask2l,warped_cloth2r,warped_mask2r=\
    warp_cloth_limps_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,limp_indices,ratio=1.0,sep_left_right=True,alpha=0.2)

    warped_clothl+=(1-warped_maskl[:,:,None])*warped_cloth2l
    warped_maskl+=(1-warped_maskl)*warped_mask2l
    warped_clothr+=(1-warped_maskr[:,:,None])*warped_cloth2r
    warped_maskr+=(1-warped_maskr)*warped_mask2r
    
    return warped_clothl,warped_maskl,warped_clothr,warped_maskr

def warp_dress_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose):
    warped_cloth,warped_mask=warp_cloth_torso_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)

    warped_cloth2,warped_mask2=warp_cloth_legs_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,is_dress=True)

    warped_cloth+=(1-warped_mask[:,:,None])*warped_cloth2
    warped_mask+=(1-warped_mask)*warped_mask2
    
    return warped_cloth,warped_mask

def warp_cloth_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,cat='upper_body'):
  
    if cat=='Upper Clothing':
        warped_cloth,warped_mask=warp_cloth_torso_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)
    elif cat=='Lower Clothing':
        warped_cloth,warped_mask=warp_cloth_legs_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)
    elif cat=='Dress':
        warped_cloth,warped_mask=warp_dress_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)
    else:
        NotImplementedError
    return warped_cloth,warped_mask

def find_arms_using_warped_clothing(left_arm_mask,right_arm_mask,warped_mask1,warped_mask2):

    #keypoints exists
    ksize=6
    if warped_mask1.sum()>0:
         warped_mask1=np.roll(warped_mask1,(ksize,ksize),(-1,-2))
         warped_mask1+=np.roll(warped_mask1,(-ksize,-ksize),(-1,-2))
         warped_mask1+=np.roll(warped_mask1,(-ksize,ksize),(-1,-2))
         warped_mask1+=np.roll(warped_mask1,(ksize,-ksize),(-1,-2))
         ys,xs=np.nonzero(warped_mask1)
         xmin, ymin, xmax, ymax=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
         xmin, ymin, xmax, ymax=int(xmin), int(ymin), int(xmax), int(ymax)   
         left_arm_mask[ymin:ymax,xmin:xmax]=0
    #else:
    #    left_arm_mask=torch.zeros_like(left_arm_mask)
    if warped_mask2.sum()>0:
         warped_mask2=np.roll(warped_mask2,(ksize,ksize),(-1,-2))
         warped_mask2+=np.roll(warped_mask2,(-ksize,-ksize),(-1,-2))
         warped_mask2+=np.roll(warped_mask2,(-ksize,ksize),(-1,-2))
         warped_mask2+=np.roll(warped_mask2,(ksize,-ksize),(-1,-2))
         ys,xs=np.nonzero(warped_mask2) 
         xmin, ymin, xmax, ymax=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
         xmin, ymin, xmax, ymax=int(xmin), int(ymin), int(xmax), int(ymax)   
         right_arm_mask[ymin:ymax,xmin:xmax]=0
    #else:
    #    right_arm_mask=torch.zeros_like(right_arm_mask)
    return left_arm_mask+right_arm_mask
    
def add_arm_to_warped_clothes(seg_np,pose_np,warped_arm1,warped_arm2,warped_mask_arm1,warped_mask_arm2,source,warped_cloth,warped_mask,arm_inds,arm_seg_inds):
        
        #Warp in-shop clothes arms to person   
        left_arm_part_mask=0
        for ind in arm_inds[0]:
              left_arm_part_mask+=np.equal(pose_np[...,-1],ind)        
        right_arm_part_mask=0
        for ind in arm_inds[1]:
              right_arm_part_mask+=np.equal(pose_np[...,-1],ind)        

        left_arm_skin_part_mask=np.equal(seg_np,arm_seg_inds[0])
        left_arm_skin_part_mask=cv2.erode(left_arm_skin_part_mask.astype(np.uint8),np.ones((5,5), np.uint8))
        right_arm_skin_part_mask=np.equal(seg_np,arm_seg_inds[1])
        right_arm_skin_part_mask=cv2.erode(right_arm_skin_part_mask.astype(np.uint8),np.ones((5,5), np.uint8))
    
        arm_mask=find_arms_using_warped_clothing(left_arm_skin_part_mask,right_arm_skin_part_mask,warped_mask_arm1,warped_mask_arm2)#1,h,w
        res_arm=source*arm_mask[...,None]
        warped_cloth=warped_cloth*warped_mask[...,None]+res_arm*(1-warped_mask[...,None])#+warped_arm*(1-warped_mask)
        warped_mask=warped_mask+(arm_mask)*(1-warped_mask)#warped_arm_mask+

        return warped_cloth,warped_mask
        
def tryon_cloth_warp(person_img,parsing,target_pose,cloth_img,cloth_mask,cloth_pose,person_pose,tryon_cat):  
    target_shape=(person_img.size[1],person_img.size[0])
    assert person_img.size==target_pose.size
    warped_cloth,warped_mask=warp_cloth_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose,tryon_cat)
    
    parsing=parsing.resize(person_img.size,resample=Image.NEAREST)
    seg_np=np.array(parsing)
    if tryon_cat=='Upper Clothing' or tryon_cat=='Dress':
            warped_arm1,warped_mask_arm1,warped_arm2,warped_mask_arm2=warp_cloth_arms_to_person(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)
            arm_inds=[[15,16,19,20],[17,18,21,22]]
            arm_seg_inds=[14,15]
    else:
            warped_arm1,warped_mask_arm1,warped_arm2,warped_mask_arm2=warp_cloth_legs_to_person_left_right(cloth_img,cloth_mask,target_shape,cloth_pose,person_pose)
            arm_inds=[[7,8,11,12],[9,10,13,14]]
            arm_seg_inds=[16,17]
            
    warped_cloth,warped_mask=\
        add_arm_to_warped_clothes(seg_np,np.array(target_pose),warped_arm1,warped_arm2,warped_mask_arm1,warped_mask_arm2,np.array(person_img),warped_cloth,warped_mask,arm_inds,arm_seg_inds)    
    return warped_cloth,warped_mask
            