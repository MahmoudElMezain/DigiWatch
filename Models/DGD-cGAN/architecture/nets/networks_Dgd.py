"""
 > Network modules of DGD-cGAN model
   * Paper: 
"""
#Libraries
import torch
from torch import nn
#First GENERATOR G1

class batchnorm_1(nn.Module):
    def __init__(self, in_c):
        super().__init__()       
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.LeakyReLU(0.2, True)
        
    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu1(x)
        return x
    

class residual_unit1(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.batch1 = batchnorm_1(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.batch2 = batchnorm_1(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.batch1(inputs)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.conv2(x)
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.uprelu = nn.ReLU(True)
        self.res = residual_unit1(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = self.uprelu(x)
        x = torch.cat([x, skip], axis=1)
        x = self.res(x)
        return x

    
class ResUnet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch1 = batchnorm_1(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.res2 = residual_unit1(64, 128, stride=2)
        self.res3 = residual_unit1(128, 256, stride=2)
        self.res4 = residual_unit1(256, 512, stride=2)
        self.dec1 = decoder1(512, 256)
        self.dec2 = decoder1(256, 128)
        self.dec3 = decoder1(128, 64)
        self.output = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.conv2(x)
        s = self.conv3(inputs)
        skip1 = x + s
        skip2 = self.res2(skip1)
        skip3 = self.res3(skip2)
        b = self.res4(skip3)
        dec1 = self.dec1(b, skip3)
        dec2 = self.dec2(dec1, skip2)
        dec3 = self.dec3(dec2, skip1)
        output = self.output(dec3)
        output = self.sigmoid(output)
        return output
