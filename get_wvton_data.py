from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import urllib.request 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def download_images(download_dir='./downloaded_data'):
    os.makedirs(download_dir,exist_ok=True)
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    with open('wvton/image_urls.txt','r') as f:
        for line in tqdm(f.readlines()):
            url,img_name=line.strip().split('\t')
            write_path=osp.join(download_dir,img_name)
            
            try:
                urllib.request.urlretrieve(url, write_path)
            except:
                print('Failed to download %s'%url)
            
            
                  
def preprocess_human_tryon_data(download_dir='./downloaded_data'):

      out_dir='wvton/images'
      os.makedirs(out_dir,exist_ok=True)
              
      bbox_file=open('wvton/bbox.txt','r')
      bbox_dict={}
      for line in tqdm(bbox_file.readlines()):
          
              name,left_idx,top_idx,right_idx,btm_idx=line.strip().split()

              raw_img_path=os.path.join(download_dir,name)
              image=Image.open(raw_img_path)
              img_np=np.array(image)          
              
              left_idx,top_idx,right_idx,btm_idx=int(left_idx),int(top_idx),int(right_idx),int(btm_idx)
              person_img=img_np[top_idx:btm_idx,left_idx:right_idx]
              
              resize_l=min(1024,max(person_img.shape[0],person_img.shape[1]))
              resize_ratio=resize_l/max(person_img.shape[0],person_img.shape[1])
              new_h=round(person_img.shape[0]*resize_ratio)
              new_w=round(person_img.shape[1]*resize_ratio)
              
              #sanity check
              seg=Image.open('wvton/parsing/'+name)
              assert new_h==seg.size[1] and new_w==seg.size[0],print(name,new_h,new_w,seg.size)
              
              person_pil=Image.fromarray(person_img).resize((new_w,new_h),resample=Image.BICUBIC)     
              person_pil.save(os.path.join(out_dir,name))
              
def preprocess_clothing_tryon_data(cl_download_dir='./downloaded_clothing'):

      out_dir='wvton/clothes'
      os.makedirs(out_dir,exist_ok=True)
              
      bbox_file=open('wvton/clothing_bbox.txt','r')
      bbox_dict={}
      for line in tqdm(bbox_file.readlines()):
          
              name,left_idx,top_idx,right_idx,btm_idx=line.strip().split()

              raw_img_path=os.path.join(cl_download_dir,name)
              
              # special image that has five sweaters
              if name in ['AdobeStock_375973851_1.jpeg','AdobeStock_375973851_2.jpeg','AdobeStock_375973851_3.jpeg','AdobeStock_375973851_4.jpeg']:
                  raw_img_path=os.path.join(cl_download_dir,'AdobeStock_375973851.jpeg')
                  
              if not osp.exists(raw_img_path):
                  print('Can not find clothing image at %s'%raw_img_path)
                  continue
                  
              image=Image.open(raw_img_path)
              img_np=np.array(image)          
              
              left_idx,top_idx,right_idx,btm_idx=int(left_idx),int(top_idx),int(right_idx),int(btm_idx)
              cl_img=img_np[top_idx:btm_idx,left_idx:right_idx]
              
              seg=Image.open('wvton/clothes_mask/'+os.path.splitext(name)[0]+'.png')
              new_h=seg.size[1] 
              new_w=seg.size[0]
              
              cl_pil=Image.fromarray(cl_img).resize((new_w,new_h),resample=Image.BICUBIC)     
              cl_pil.save(os.path.join(out_dir,name))
              
if __name__=='__main__':
    
    download_dir='./downloaded_data'
    
    # Please manually download 18 clothing images to this directory
    # Image urls can be found at wvton/clothing_urls.txt
    cl_download_dir='./cl_downloaded_data'
    
    print('Start downloading images')
    download_images(download_dir=download_dir)
    
    print('Start preprocessing human images')
    preprocess_human_tryon_data(download_dir=download_dir)
    
    if len(os.listdir(cl_download_dir))==0:
        print('Please manually download 18 clothing images to %s'%cl_download_dir)
        print('Image urls can be found at wvton/clothing_urls.txt')
    else:
        print('Start preprocessing clothes images')
        preprocess_clothing_tryon_data(cl_download_dir=cl_download_dir)