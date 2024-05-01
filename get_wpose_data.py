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
    with open('wpose/image_urls.txt','r') as f:
        for line in tqdm(f.readlines()):
            url,img_name=line.strip().split('\t')
            write_path=osp.join(download_dir,img_name)
            
            try:
                urllib.request.urlretrieve(url, write_path)
            except:
                print('Failed to download %s'%url)
            
            
                  
def preprocess_reposing_data(download_dir='./downloaded_data'):

      out_dir='wpose/images'
      os.makedirs(out_dir,exist_ok=True)
              
      bbox_file=open('wpose/bbox.txt','r')
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
              seg=Image.open('wpose/parsing/'+name)
              assert new_h==seg.size[1] and new_w==seg.size[0]
              
              person_pil=Image.fromarray(person_img).resize((new_w,new_h),resample=Image.BICUBIC)     
              person_pil.save(os.path.join(out_dir,name))
              
              
if __name__=='__main__':
    download_dir='./downloaded_data'
    
    print('Start downloading images')
    download_images(download_dir=download_dir)
    
    print('Start preprocessing images')
    preprocess_reposing_data(download_dir=download_dir)