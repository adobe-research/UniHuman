from PIL import Image
from tqdm import tqdm
import os
import os.path as osp
import urllib.request 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_pil(person_img,max_size=512,resample=Image.BICUBIC):
    
    resize_l=min(max_size,max(person_img.size[0],person_img.size[1]))
    resize_ratio=resize_l/max(person_img.size[0],person_img.size[1])
    new_w=round(person_img.size[0]*resize_ratio)
    new_h=round(person_img.size[1]*resize_ratio)
    assert new_w==max_size or new_h==max_size
    
    return person_img.resize((new_w,new_h),resample=resample)

def download_and_resize_images(img_dir='lh-400k/images'):
    os.makedirs(img_dir,exist_ok=True)
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    num_fails=0
    num_success=0
    with open('lh-400k/image_urls.txt','r') as f:
        for line in tqdm(f.readlines()):
            url,img_name=line.strip().split('\t')
            write_path=osp.join(img_dir,img_name)
            
            try:
                urllib.request.urlretrieve(url, write_path)
                # Resize image to save disk space
                img=Image.open(write_path)
                img=resize_pil(img,max_size=512,resample=Image.BICUBIC)
                img.save(write_path)
                num_success+=1
            except:
                print('Failed to download %s'%url)
                num_fails+=1
                
    print('Successfullly downloaded %d samples and failed downloading %d samples'%(num_success,num_fails))
            
if __name__=='__main__':
    
    img_dir='lh-400k/images'
    
    print('Start downloading images')
    
    download_and_resize_images(img_dir=img_dir)
    