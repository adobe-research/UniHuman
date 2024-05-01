import glob
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput,DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import (
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
)


from PIL import Image
import time
import numpy as np
import io

    
class PosePredictor():
   
    def __init__(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file('./configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml')
        cfg.MODEL.WEIGHTS = './models/model_final_844d15.pkl'
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

  
    def predict_pose(self,img_pil):

        
        img=np.array(img_pil)[...,::-1]  # predictor expects BGR image.
        outputs = self.predictor(img)["instances"]
        result={}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
                
        if len(result['pred_boxes_XYXY'])<1:
             print('No person in ',result['file_name'])
             return None

        max_area=0
        for index in range(len(result['pred_boxes_XYXY'])):
           box=result['pred_boxes_XYXY'][index].cpu().numpy().astype(np.int32).tolist()
           if (box[3]-box[1])*(box[2]-box[0])>max_area:
                x1,y1,x2,y2=box
                max_area=(box[3]-box[1])*(box[2]-box[0])
                i=result['pred_densepose'][index].labels.cpu().numpy()
                uv=result['pred_densepose'][index].uv.cpu().numpy() 
               
        H,W=img.shape[:2] 
       
        uv=np.transpose(uv,axes=[1,2,0])
        uv=np.round(255*uv).astype(np.uint8)
        if y2-y1!=i.shape[0]:
             y2=y2-np.sign(y2-y1-i.shape[0])
        if x2-x1!=i.shape[1]:
             x2=x2-np.sign(x2-x1-i.shape[1])
        uvi=np.zeros((H,W,3),dtype=np.uint8)
        uvi[y1:y2,x1:x2,2]=i
        uvi[y1:y2,x1:x2,:2]=uv[:,:,[1,0]]

        uvi=Image.fromarray(uvi)
        return uvi
        
        

   