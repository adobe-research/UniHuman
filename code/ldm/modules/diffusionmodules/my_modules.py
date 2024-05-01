
import torch as th
import torch.nn as nn
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
)
import math

def make_grid(iW,iH):
    grid_x = th.arange(0,iW).view(1, iW).expand(iH, -1).reshape(iH*iW).to(th.device('cuda'))
    grid_y = th.arange(0,iH).view(iH, 1).expand(-1, iW).reshape(iH*iW).to(th.device('cuda'))
    return grid_x,grid_y
        
class _PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, n_dim,n_heads=1):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads=n_heads

        inv_freq = 1 / (10000 ** (th.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.pos_netx = linear(embed_dim, n_heads * n_dim)
        self.pos_nety = linear(embed_dim, n_heads * n_dim)
        self.q_bias = nn.Parameter(th.Tensor(1,n_heads * n_dim,1))


    def forward(self, x):
        bs, t, length = x.shape #b,768,170
        assert length==170
        x=x.view(bs,t,10,17)#10=num_styles, 17=spatial_res
        
        n_dim=t//self.n_heads

        res=int(math.sqrt(16))
        pos_seqx,pos_seqy=make_grid(res,res)
        #pos_seq: length
        sinusoid_inpx = th.ger(pos_seqx, self.inv_freq)
        pos_embx = th.cat([sinusoid_inpx.sin(), sinusoid_inpx.cos()], dim=-1)
        
        sinusoid_inpy = th.ger(pos_seqy, self.inv_freq)
        pos_emby = th.cat([sinusoid_inpy.sin(), sinusoid_inpy.cos()], dim=-1)
        pos_emd=(self.pos_netx(pos_embx)+self.pos_nety(pos_emby)).unsqueeze(0).expand(bs,-1,-1)#b,length,n_heads*n_dm.
        pos_emd=pos_emd.permute(0,2,1)#b,t,16#.reshape(bs * self.n_heads,n_dim,length)

        pos_emd=th.cat([pos_emd,self.q_bias.repeat(bs,1,1)],-1).unsqueeze(2)#b,t,1,17
        x=(x+pos_emd).view(bs,t,length)
        
        #pos_emd=th.einsum("bws,bwt->bst", q, pos_emd)
        #return pos_emd
        return x 