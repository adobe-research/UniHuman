
import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict

# PyTorch includes
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import cv2

sys.path.append('Graphonomy/')
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

    
    
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)




def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample


class HumanParser():
    def __init__(self):
        
        self.label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        self.adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        self.adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        self.adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()
        
        net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
        x = torch.load('Graphonomy/inference.pth')
        net.load_source_model(x)
        print('loaded parsing model')
        net.eval()
        
        self.clothes_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.clothes_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").cuda()
        
        self.net=net.cuda()
        
        scale_list_dict={1024:[0.375, 0.125, 0.25, 0.5, 0.625, 0.75],
                   512:[0.5, 0.25, 0.375, 0.625, 0.75, 0.875],
                   256: [1, 0.5, 0.75, 1.25, 1.5, 1.75]
                  }
    
        transforms_dict={}
        transforms_flip_dict={}
        for scale,scale_list in scale_list_dict.items():
            transforms_dict[scale]=[]
            transforms_flip_dict[scale]=[]
            for pv in scale_list:
                composed_transforms_ts = transforms.Compose([
                  tr.Scale_only_img(pv),
                  tr.Normalize_xception_tf_only_img(),
                  tr.ToTensor_only_img()])

                composed_transforms_ts_flip = transforms.Compose([
                  tr.Scale_only_img(pv),
                  tr.HorizontalFlip_only_img(),
                  tr.Normalize_xception_tf_only_img(),
                  tr.ToTensor_only_img()])
                  
                transforms_dict[scale].append(composed_transforms_ts)
                transforms_flip_dict[scale].append(composed_transforms_ts_flip)
                
        self.transforms_dict=transforms_dict
        self.transforms_flip_dict=transforms_flip_dict
        
    def decode_labels(self,mask, num_images=1, num_classes=20):
        """Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
        """
        n, h, w = mask.shape
        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = self.label_colours[k]
            outputs[i] = np.array(img)
        return outputs
        
    def predict_clothes_parsing(self,img_pil):
        batch = self.clothes_processor(images=img_pil, return_tensors="pt")['pixel_values'] 
        prediction = self.clothes_model(batch.cuda()).logits
        prediction = nn.functional.interpolate(
         prediction,
         size=img_pil.size[::-1],
         mode="bilinear",
        align_corners=False,
        )
        indices=torch.argmax(prediction[0],dim=0)
        mask=torch.greater(indices,0).long()
        mask=mask.cpu().numpy().astype(np.uint8)
        return Image.fromarray(mask),Image.fromarray(mask*255)
        
    def predict_parsing(self,img_pil):
        W,H=img_pil.size
        if min(W,H)>1024:
            scale=1024
        elif min(W,H)>512:
            scale=512
        else:
             scale=256
    
        testloader_list = []
        testloader_flip_list = []
        for composed_transforms_ts,composed_transforms_ts_flip in zip(self.transforms_dict[scale],self.transforms_flip_dict[scale]):

           testloader_list.append(img_transform(img_pil, composed_transforms_ts))
           # print(img_transform(img_pil, composed_transforms_ts))
           testloader_flip_list.append(img_transform(img_pil, composed_transforms_ts_flip))
           
        for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            inputs = torch.cat((inputs, inputs_f), dim=0)
            if iii == 0:
                _, _, h, w = inputs.size()
            # assert inputs.size() == inputs_f.size()

            # Forward pass of the mini-batch
            inputs = Variable(inputs, requires_grad=False)

            with torch.no_grad():
                inputs = inputs.cuda()
                # outputs = net.forward(inputs)
                outputs = self.net.forward(inputs, self.adj1_test.cuda(), self.adj3_test.cuda(), self.adj2_test.cuda())
                outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii > 0:
                    outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        predictions = torch.max(outputs_final, 1)[1]
        results = predictions.cpu().numpy().astype(np.uint8)
        vis_res = self.decode_labels(results).astype(np.uint8)
        
        return Image.fromarray(results[0]),Image.fromarray(vis_res[0])

