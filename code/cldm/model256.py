"""
This file is modified from https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py which is under Apache 2.0 license.
"""
import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import cv2

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock,CrossAttentionBlock,NewSoftCrossAttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch.nn.functional as Func

from transformers import AutoTokenizer, CLIPTextModel

class ControlledUnetModel(UNetModel):
    def __init__(self,*arg,**kwargs):
        super().__init__(*arg,**kwargs)
        model_channels=self.model_channels
        channel_mult=self.channel_mult
        attention_resolutions=self.attention_resolutions
        time_embed_dim = model_channels * 4
        dropout=self.dropout
        dims=2
        use_checkpoint=self.use_checkpoint
        use_scale_shift_norm=False
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            if level != len(channel_mult) - 1:
                ds *= 2

        self.includes_att_layer=[]
        self.control_block_gatt=nn.ModuleList([])
        self.control_block_latt=nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ch = model_channels * mult
                if level and i == self.num_res_blocks[level]:
                     ds //= 2

                if ds not in [2,1]:#attention_resolutions:
                      self.includes_att_layer.append(False)
                else:
                      self.includes_att_layer.append(True)
                      glayers=[
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                             ),
                            CrossAttentionBlock(channels=ch,channels_kv=768,num_heads=self.num_heads,use_pos=False)]
                      self.control_block_gatt.append(TimestepEmbedSequential(*glayers))
                      llayers=[
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                             ),
                            NewSoftCrossAttentionBlock(channels=ch,channels_kv=768,num_heads=self.num_heads,use_pos=False)]
                      self.control_block_latt.append(TimestepEmbedSequential(*llayers)) 

            
                
            
        
    def forward(self, x, timesteps=None, cond_txt=None, cond_tex=None, human_area_mask=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, cond_txt)
                hs.append(h)
            h = self.middle_block(h, emb, cond_txt)

        tex=cond_tex.pop()
        vis_mask=cond_tex.pop()#N,h,w            
        h += cond_tex.pop()
        tex_pooled=tex[:,:9] # global features
        tex_token=tex[:,9:] # local features
        length_tex=tex_token.shape[1]-vis_mask.sum((1,2)).max()
        assert tex_token.shape[1]<=256*3,print(tex_token.shape)
            
        bg_mask=1-human_area_mask
        return_mask_weight=False

        j=0
        for i, module in enumerate(self.output_blocks):
            # if cond_tex is None:
            #     h = torch.cat([h, hs.pop()], dim=1)                
            # else:
            h = torch.cat([h, hs.pop() + cond_tex.pop()], dim=1)
            h = module(h, emb, cond_txt)
            if self.includes_att_layer[i]:
                tmp_bg_mask=Func.interpolate(bg_mask,size=(h.shape[2],h.shape[3]),mode='nearest').flatten(1)
                h=self.control_block_gatt[j](h,emb,None,tex_pooled,bg_mask=tmp_bg_mask,return_mask_weight=return_mask_weight)                
                       
                tmp_vis_mask=Func.interpolate(vis_mask,size=(h.shape[2],h.shape[3]),mode='nearest').flatten(1)
                h=self.control_block_latt[j](h,emb,None,tex_token,return_mask_weight=return_mask_weight,vis_mask=tmp_vis_mask,bg_mask=tmp_bg_mask,length_context=length_tex)
                      
                    
                j+=1

        h = h.type(x.dtype)
        
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        hint_base_channels=32
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, hint_base_channels, 3, padding=1),
            ResBlock(
                        hint_base_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels, conv_resample, dims=dims, out_channels=hint_base_channels
                        ),
            ResBlock(
                        hint_base_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels*3,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels*3, conv_resample, dims=dims, out_channels=hint_base_channels*3
                        ),
            ResBlock(
                        hint_base_channels*3,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels*8,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels*8, conv_resample, dims=dims, out_channels=hint_base_channels*8
                        )
        )
        self.input_vis_block = TimestepEmbedSequential(
            conv_nd(dims, 3, hint_base_channels, 3, padding=1),
            ResBlock(
                        hint_base_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels, conv_resample, dims=dims, out_channels=hint_base_channels
                        ),
            ResBlock(
                        hint_base_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels*2,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels*2, conv_resample, dims=dims, out_channels=hint_base_channels*2
                        ),
            ResBlock(
                        hint_base_channels*2,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels*4,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels*4, conv_resample, dims=dims, out_channels=hint_base_channels*4
                        ),
            ResBlock(
                        hint_base_channels*4,
                        time_embed_dim,
                        dropout,
                        out_channels=hint_base_channels*8,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
            Downsample(
                            hint_base_channels*8, conv_resample, dims=dims, out_channels=hint_base_channels*8
            ),
            conv_nd(dims, hint_base_channels*8, context_dim, 3, padding=1),
        )
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, hint_base_channels*8, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                        
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
           
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        self.clip_image_encoder=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for name,param in self.clip_image_encoder.named_parameters():
            if 'norm.bias'==name or 'norm.weight'==name or 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name:
                param.requires_grad=True
            else:
                param.requires_grad=False
 
        clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        inputs = tokenizer(['face', 'hair', 'headwear', 'upper clothing', 'coat', 'lower clothing', 'shoes', 'accesories'], padding=True, return_tensors="pt")
        self.style_cls_emd = clip_text_model(**inputs).pooler_output[None].detach()#1,10,512
        self.style_cls_emd.requires_grad=False
        self.clip_global_adaptor=linear(context_dim,context_dim-128)
        self.clip_cls_adaptor=linear(512,128) # clip text encoder output dim is 512
        
        self.clip_adaptor=linear(context_dim,context_dim)
        self.unknown_style=nn.parameter.Parameter(data=torch.zeros(1,context_dim),requires_grad=True)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, p_concat, timesteps, cond_tex,**kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        pwarped_pixels=p_concat[:,3:6]
        pwarped_mask=p_concat[:,6:7]
        pwarped_feats=self.input_vis_block(pwarped_pixels,emb,None).permute(0,2,3,1)#N,C,H,W->N,H,W,C
        pwarped_mask=Func.interpolate(pwarped_mask,size=(pwarped_feats.shape[1],pwarped_feats.shape[2]),mode='nearest')#N,1,H,W

        max_len=int(pwarped_mask.sum((-1,-2)).max().item())
        pwarped_tokens=[]
        for i in range(pwarped_feats.shape[0]):
              t=pwarped_feats[i][pwarped_mask[i,0].bool()]#n,C
              pad=torch.zeros((max_len-len(t),t.shape[-1]),requires_grad=False).float().to(emb).detach()
              t=torch.cat([t,pad],0)#max_len,C
              pwarped_tokens.append(t)
            
        pwarped_tokens=torch.stack(pwarped_tokens,0)#N,max_len,C
        cond_tex=torch.cat([cond_tex,pwarped_tokens],1)

        h = self.input_hint_block(p_concat,emb,None)
                 
        outs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, None)
            outs.append(zero_conv(h, emb, None))

        h = self.middle_block(h, emb, None)
        outs.append(self.middle_block_out(h, emb, None))
        outs.append(pwarped_mask)
        outs.append(cond_tex)
        

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key_tex,control_key_pose, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key_tex = control_key_tex
        self.control_key_pose = control_key_pose

    def get_unkown_style(self,num):
        emd=self.control_model.unknown_style
        return emd.repeat(num,1)


    def encode_tex_images(self,controlt,seg):

        b,num_styles,C,H,W=controlt.shape
        assert H==W==224, print(H,W)
        controlt=controlt.view(b*num_styles,C,H,W)
        controlt=self.control_model.clip_image_encoder.forward_features(controlt)['x_norm_patchtokens']#b*9,256,768
        
        controlt=self.control_model.clip_adaptor(controlt) ##b*9,256,768
        controlt=controlt.view(b,num_styles,controlt.shape[1],controlt.shape[2])#b,9,256,768

        RES=int(np.sqrt(controlt.shape[2]))
        seg=Func.interpolate(seg,(RES,RES),mode='nearest').flatten(2) #b,9,h,w->b,9,256


        global_feats=(controlt*seg.unsqueeze(-1)).sum(2)/(1e-6+seg.sum(2).unsqueeze(-1))#b,9,256,768->b,9,768

        known_mask=torch.clamp(seg.sum(-1,keepdim=True),0,1) #b,9,1
        global_feats=global_feats*known_mask+self.get_unkown_style(b*global_feats.shape[1]).reshape(b,global_feats.shape[1],-1)*(1-known_mask)
        
        global_feats_part1=self.control_model.clip_global_adaptor(global_feats[:,:8])
        cls_emd=self.control_model.style_cls_emd.repeat(b,1,1).to(self.device)
        cls_emd=self.control_model.clip_cls_adaptor(cls_emd)
        cls_emd=torch.gt(seg[:,:8].sum(-1,keepdims=True),0).to(cls_emd)*cls_emd #b,8,128
        global_feats=torch.cat([torch.cat([global_feats_part1,cls_emd],-1),global_feats[:,8:9]],1)


        
        local_feats=[]
        max_len=0
        for i in range(b):
            feats=controlt[i,:8][seg[i,:8].bool()]
            local_feats.append(feats) #n,768
            max_len=max(max_len,len(feats))

        if max_len==0:
            local_feats=self.get_unkown_style(b).reshape(b,1,-1)
        else:
           for i in range(b):
               pad=self.get_unkown_style(max_len-len(local_feats[i]))
               local_feats[i]=torch.cat([local_feats[i],pad],0)

           local_feats=torch.stack(local_feats,0)   

        controlt=torch.cat([global_feats,local_feats],1)
        return controlt

    @torch.no_grad()
    def get_input(self, batch, k, bs=None,*args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        controlt = batch[self.control_key_tex]
        controlp = batch[self.control_key_pose]
        if bs is not None:
            controlt = controlt[:bs]
            controlp = controlp[:bs]
        controlt = controlt.to(self.device)
        controlp = controlp.to(self.device)
        controlt = controlt.to(memory_format=torch.contiguous_format).float()
        controlp = controlp.to(memory_format=torch.contiguous_format).float()

        seg=batch['seg']
        if bs is not None:
            seg = seg[:bs]
        seg=seg.to(self.device)
            
        pwarped_tex=batch['pwarped_tex'][:bs].to(self.device) #N,4,h,w
        controlp=torch.concat([controlp,pwarped_tex],1) 
            
        bg_tex=batch['bg'][:bs].to(self.device) #N,4,h,w
        controlp=torch.concat([controlp,bg_tex],1) 

     
        human_area_mask=batch['human_area_mask']
        if bs is not None:
            human_area_mask=human_area_mask[:bs]
        human_area_mask = human_area_mask.to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(txt_crossattn=[c], p_concat=[controlp],tex_crossattn=[controlt],human_area_mask=[human_area_mask],seg=[seg])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['txt_crossattn'], 1)
        cond_tex = torch.cat(cond['tex_crossattn'], 1)
        seg=torch.cat(cond['seg'], 1)
        
        # If cond_tex includes raw images
        if len(cond_tex.shape)==5:
              cond_tex=self.encode_tex_images(cond_tex,seg)
            
        human_area_mask= torch.cat(cond['human_area_mask'],1)

        cond_tex = self.control_model(x=x_noisy, p_concat=torch.cat(cond['p_concat'], 1), timesteps=t,cond_tex=cond_tex)
        eps = diffusion_model(x=x_noisy, timesteps=t, cond_txt=cond_txt, cond_tex=cond_tex,human_area_mask=human_area_mask)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def get_uncond_style(self, N,num_styles):
        
        white_in_dino=torch.tensor([2.2489083 , 2.42857143, 2.64],requires_grad=False)[:,None,None]
        cond=torch.ones((3,224,224))*white_in_dino
        cond=cond[None,None].repeat(N,num_styles,1,1,1).to(self.device)
        return cond

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=2.0, unconditional_guidance_label=None,
                   use_ema_scope=True,task='Pose Transfer',
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
      
        c_cat, c, c2,seg,human_area_mask = c["p_concat"][0][:N], c["txt_crossattn"][0][:N], c["tex_crossattn"][0][:N],c["seg"][0][:N],c["human_area_mask"][0][:N]
        cond_pixels=c_cat[:,3:]
        
        num_styles=c2.shape[1]
                       
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"p_concat": [c_cat], "txt_crossattn": [c], "tex_crossattn": [c2],"seg":[seg],"human_area_mask":[human_area_mask]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            un_c = self.get_unconditional_conditioning(N)
            un_controlt=self.get_uncond_style(N,num_styles)
            un_seg=torch.zeros_like(seg) 
            
            if task=='Pose Transfer':
                cond={"p_concat": [c_cat], "txt_crossattn": [c], "tex_crossattn": [c2],"seg":[seg],"human_area_mask":[human_area_mask]}
                uc_full = {"p_concat": [torch.zeros_like(c_cat)], "txt_crossattn": [un_c], "tex_crossattn": [un_controlt],"seg":[un_seg],"human_area_mask":[torch.zeros_like(human_area_mask)]}
            elif task=='Virtual Try-on':
                cond={"p_concat": [c_cat], "txt_crossattn": [c], "tex_crossattn": [c2],"seg":[seg],"human_area_mask":[human_area_mask]}
                uc_full = {"p_concat": [torch.zeros_like(c_cat)], "txt_crossattn": [c], "tex_crossattn": [un_controlt],"seg":[un_seg],"human_area_mask":[torch.zeros_like(human_area_mask)]}
            else:
                cond={"p_concat": [c_cat], "txt_crossattn": [c], "tex_crossattn": [c2],"seg":[un_seg],"human_area_mask":[human_area_mask]}
                uc_full = {"p_concat": [c_cat], "txt_crossattn": [un_c], "tex_crossattn": [c2],"seg":[un_seg],"human_area_mask":[human_area_mask]}
                
            samples_cfg, _ = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg).clip(-1,1)
            
            log["samples_cfg"] = x_samples_cfg
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["p_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
