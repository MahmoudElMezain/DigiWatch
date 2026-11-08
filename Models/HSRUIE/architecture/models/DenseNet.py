import torch
import torch.nn.functional as F
from torch import nn
from .NEDB_IN import Dense_Block_IN


class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode='bilinear')
		out = self.relu(self.IN1(self.conv0(x)))
		return out

class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.gn1 = nn.GroupNorm(32, out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.gn2 = nn.GroupNorm(32, out_channels)


	def forward(self, x):
		out = torch.relu(self.gn1(self.conv1(x)))
		out = self.gn2(self.conv2(out))
		out += x
		out = torch.relu(out)
		return out

class FE(nn.Module):
	def __init__(self):
		super(FE, self).__init__()

		self.conv11 = nn.Sequential(
			ResBlock(64, 64),  # 256 -> 128
			ResBlock(64, 64),
			ResBlock(64, 64),
			ResBlock(64, 64),
			ResBlock(64, 64),
			ResBlock(64, 64),
		)
		self.conv12 = nn.Sequential(
			ResBlock(128, 128),  # 256 -> 128
			ResBlock(128, 128),
			ResBlock(128, 128),
			ResBlock(128, 128),
			ResBlock(128, 128),
			ResBlock(128, 128),
		)
		self.conv13 = nn.Sequential(
			ResBlock(256, 256),  # 256 -> 128
			ResBlock(256, 256),
			ResBlock(256, 256),
			ResBlock(256, 256),
			ResBlock(256, 256),
			ResBlock(256, 256),
		)
		#
		# 几个 conv, 中间 channel, 输入 channel

	#

	def forward(self, x):  # 1, 3, 256, 256   300
		#
		[d1, d2, d3] = x
		#######################################################
		d1 = self.conv11(d1)   # 64,256,256
		d2 = self.conv12(d2)   # 64, 128, 128
		d3 = self.conv13(d3)    # 64, 64, 64

		return [d1, d2, d3]

class SAD(nn.Module):
	def __init__(self):
		super(SAD, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(256, 64, 3, 1, 1),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		#
		# 几个 conv, 中间 channel, 输入 channel
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 16 -> 8
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.GroupNorm(32,64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, 3, 3, 1, 1),
			nn.Tanh(),
		)

	#

	def forward(self, x):  # 1, 3, 256, 256   300
		#
		[d1, d2, d3] = x
		#######################################################
		d1 = self.conv1(d1)   # 64,256,256
		d2 = self.conv2(d2)   # 64, 128, 128
		d3 = self.conv3(d3)    # 64, 64, 64
		down_11 = self.Dense_Down_1(d1)  # 1, 64, 256, 256

		down_21 = self.Dense_Down_2(d2)  # 1, 64, 128, 128

		down_31 = self.Dense_Down_3(d3)  # 1, 64, 64, 64
		down_3 = self.trans_down_3(down_31)  # 1, 64, 32, 32

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32
		down_4 = self.trans_down_4(down_41)  # 1, 64, 16, 16

		#######################################################
		Latent = self.Latent(down_4)  # 1, 64, 16, 16               19
		#######################################################

		up_4 = self.trans_up_4(Latent)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)  # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)  # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)  # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)  # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)  # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)  # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)  # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)  # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		# feature = feature + feature_neg  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs
