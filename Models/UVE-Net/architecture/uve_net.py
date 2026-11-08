"""Original UVE-Net large architecture, with unused training utilities omitted."""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from uve_blocks import ResidualBlockNoBN, ResidualBlockNoBN2D, make_layer
from dynamic_dw import DynamicDWConv

class Deblur(nn.Module):
    def __init__(self, num_feat=32, num_block=15):
        super().__init__()
        # extractor & reconstruction
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        ##
        self.n_feats0=num_feat
        self.n_feats=num_feat
        self.num_feat=num_feat
        self.A1=nn.Sequential(make_layer(ResidualBlockNoBN, 30, num_feat=num_feat))
        self.A2=nn.Sequential(make_layer(ResidualBlockNoBN, 30, num_feat=num_feat))
        self.Block1=nn.Sequential(make_layer(ResidualBlockNoBN2D, 15, num_feat=num_feat*4))
        self.Block2=nn.Sequential(make_layer(ResidualBlockNoBN2D, 5, num_feat=num_feat))
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))
        self.conv_last =  nn.Sequential(nn.Conv2d(self.n_feats0,3, 3, 1, 1),nn.Sigmoid())
        self.dyd= DynamicDWConv(self.num_feat*16, 3, 1, self.num_feat*16)
        self.conv_last2 =  nn.Sequential(nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True),nn.Sigmoid())
        self.bias = nn.Parameter(torch.zeros(self.num_feat*16))
    def stageA00(self,x0):
        x0=self.A1(x0)
        return x0
    def stageA0(self,x0):
        x0=self.A2(x0)
        x_out = self.conv_last(x0)
        return x0,x_out
    def stageB10(self,x0):
        x0=self.Block1(x0)
        return x0
    def stageB11(self,x0):
        x0=self.Block2(x0)
        x_out = self.conv_last2(x0)
        return x_out
    def model_B(self,L,S,b):
         pixdown = torch.nn.PixelUnshuffle(4)
         pixup = torch.nn.PixelShuffle(2)
         L_d=pixdown(L)
         S_d=pixdown(S)
         mid=[]
         for i in range(b):
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))#[1,384]
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         mid=torch.stack(mid)
         out=pixup(mid)
         return out
    def model_B2(self,L,S,b):
         pixup = torch.nn.PixelShuffle(4)
         pixdown = torch.nn.PixelUnshuffle(2)
         pixdown2 = torch.nn.PixelUnshuffle(4)
         L_d=pixdown(L)
         S_d=pixdown2(S)
         mid=[]
         for i in range(b):
             S_d_weight=self.dyd(S_d[i].unsqueeze(0))#[1,384]
             m=F.conv2d(L_d[i].unsqueeze(0), S_d_weight, self.bias.repeat(1), stride=1, padding=1, groups=self.num_feat*16)
             mid.append(m)
         mid=torch.stack(mid)
         out=pixup(mid)
         return out
    def forward(self, lrs):
        b, t, c, h, w = lrs.size()
        lrs_mid=lrs[:,t//2]
        lrs_mid_4 = F.interpolate(lrs_mid, scale_factor=0.25, mode='bilinear', align_corners=False)
        lrs_mid_4_0 = self.feat_extract(lrs_mid_4)
        lrs_mid_4_00=self.stageA00(lrs_mid_4_0)
        lrs3d=self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))
        lrs3d = rearrange(lrs3d,'b c t h w -> b t c h w')
        for i in range(t):
            lrs_=lrs3d[:,i]
            f_ele=self.model_B(lrs_,lrs_mid_4_00,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
                #[2, 3, 96, 512, 512]
        lrs3d_B = rearrange(f_out, 'b t c h w -> b c t h w')
        lrs3d_StageB1=self.stageB10(lrs3d_B) #pass by [1,3,3] conv
        lrs3d_B2 = rearrange(lrs3d_StageB1, 'b c t h w -> b t c h w')
        delivery_4_fea,restore_4_out= self.stageA0(lrs_mid_4_00)
        for i in range(t):
            lrs_=lrs3d_B2[:,i]
            f_ele=self.model_B2(lrs_,delivery_4_fea,b)
            if i==0:
                f_out=f_ele
            else:
                f_out=torch.cat((f_out,f_ele),dim=1)
        lrs3d_B2 = rearrange(f_out, 'b t c h w -> b c t h w')
        lrs3d_StageB1=self.stageB11(lrs3d_B2)#b c t h w'
        output=rearrange(lrs3d_StageB1, 'b c t h w -> b t c h w')
        return output,restore_4_out
