import torch

import torch.nn as nn

from torch.nn import Parameter

import torch.nn.functional as F

from models.NEDB_IN import Dense_Block_IN

import math

def define_G(input_nc, output_nc, which_model_netG):
	netG = None
	#############################################
	# Unet
	#############################################
	if which_model_netG == 'unet':
		netG = NLEDN_IN_32_16_32(input_nc, output_nc)
	return netG

def Conv_Channel(C_Channel, Resized_C):
	_, channel_num, _, _ = C_Channel.shape
	#
	conv_channel = nn.Sequential(nn.Conv2d(channel_num, Resized_C, 1, 1, 0).cuda())
	#
	output = conv_channel(C_Channel)
	return output

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

class DoubleConv(nn.Module):
	"""(convolution => [IN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.InstanceNorm2d(out_channels),
			nn.ReLU()
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.up_conv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.up_conv(x)

class ResBlok(nn.Module):
	def __init__(self, ch_in, ch_out):
		super(ResBlok, self).__init__()
		# TWO LAYERS
		self.conv1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
		self.relu1 = nn.ReLU()
		self.in1 = nn.InstanceNorm2d(ch_out)
		#
		self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
		self.relu2 = nn.ReLU()
		self.in2 = nn.InstanceNorm2d(ch_out)

	def forward(self, x):
		out = self.conv1(x)
		out = self.in1(out)
		out = self.relu1(out)
		#
		out = self.conv2(out)
		out = self.in2(out)
		#
		out = x + out
		out = self.relu2(out)
		#
		return out

def row_normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = torch.sum(mx, dim=1)
	r_inv = 1 / (rowsum + 1e-10)
	r_mat_inv = torch.diag(r_inv)
	mx = torch.mm(r_mat_inv, mx)
	return mx

class GraphConvolution(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):  #
		support = torch.mm(input, self.weight)  # (8192,64),  (64, 64)
		output = torch.mm(adj, support)  # (8192,8192), (8192,64)
		#
		if self.bias is not None:
			return output + self.bias
		else:
			return output

class Location_Mask(nn.Module):
	def __init__(self):
		super(Location_Mask, self).__init__()
		#
		self.mask_3_64 = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU())
		#
		self.mask_res = ResBlok(ch_in=64, ch_out=64)
		#
		self.gc0_1 = GraphConvolution(128, 64)
		self.gc0_2 = GraphConvolution(64, 64)

	def forward(self, feats_input):  # [N, 64, H, W], [3N, 1, 1, 1] --> [N, 3N, H, W]
		feats = self.mask_3_64(feats_input)  # [1, 64, 256, 256]
		feats_content = self.mask_res(feats)
		#
		feature_mask_0_Max_image = F.max_pool2d(feats, 8)  # [1, 64, 32, 32]
		feature_mask_0_Avg_image = F.avg_pool2d(feats, 8)  # [1, 64, 32, 32]
		feature_mask_0_all_image = torch.cat([feature_mask_0_Max_image, feature_mask_0_Avg_image],
		                                     dim=1)  # [1, 128, 64, 64]
		#
		feat0_image1 = feature_mask_0_all_image.permute(0, 2, 3, 1).reshape(-1, feature_mask_0_all_image.shape[
			1])  # (N*H*W,C)  (4096, 128)
		adj0_image1 = torch.mm(feat0_image1, torch.t(feat0_image1))  # (N*H*W, N*H*W)   (4096, 4096)
		adj0_image1 = row_normalize(adj0_image1)  # (4096, 4096)
		gc0_image1 = F.relu(self.gc0_1(feat0_image1, adj0_image1))  # (4096, 64)
		gc0_image1 = F.relu(self.gc0_2(gc0_image1, adj0_image1))  # (4096, 64)  # 下一个 [1, 64, 64, 64]
		gc0_image1 = gc0_image1.reshape(feature_mask_0_all_image.shape[0], feature_mask_0_all_image.shape[2],
										feature_mask_0_all_image.shape[3], 64)
		output = gc0_image1.permute(0, 3, 1, 2)  # (N,64,H,W)  [1, 64, 64, 64]
		#
		output = F.upsample(output, scale_factor=8, mode='bilinear')  # [N 64, 256, 256]
		#
		output_res = output + feats_content
		#
		return output_res

class CALayer(nn.Module):
	def __init__(self, channel):
		super(CALayer, self).__init__()
		out_channel = channel
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
		#
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
				nn.Conv2d(out_channel, out_channel // 8, 1, padding=0, bias=True),
				nn.ReLU(),
				nn.Conv2d(out_channel // 8, channel, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		t1 = self.conv1(x)  # in
		t2 = self.relu(t1)  # in, 64
		y = self.avg_pool(t2)  # torch.Size([1, in, 1, 1])
		y = self.ca(y)  # torch.Size([1, in, 1, 1])
		m = t2 * y      # torch.Size([1, in, 64, 64]) * torch.Size([1, in, 1, 1])
		return x + m

class PALayer(nn.Module):
	def __init__(self, channel):
		super(PALayer, self).__init__()
		self.pa = nn.Sequential(
				nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y

class Inter_Assist_Intra(nn.Module):  # Inter: (1, 64, H, W)---- Intra: (N, 64, H, W)
	def __init__(self):
		super(Inter_Assist_Intra, self).__init__()

		self.dim_redunce = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU())

		self.CA = CALayer(channel=64)
		self.PA = PALayer(channel=64)

	def forward(self, Intra, Inter):   # Intra, Inter;;; Inter: (1, 64, H, W)---- Intra: (N, 64, H, W)
		# Expand Inter
		Inter_Expand = Inter
		for ii in range(1, Intra.size(0)):
			Inter_Expand = torch.cat([Inter_Expand, Inter], dim=0)
		#
		Intra_Inter = torch.cat([Intra, Inter_Expand], dim=1)  # [N, 128, 256, 256]
		Intra_Inter_64 = self.dim_redunce(Intra_Inter)
		#
		Intra_Inter = self.CA(Intra_Inter_64)
		Intra_Inter = self.PA(Intra_Inter) + Intra_Inter_64
		#
		return Intra_Inter

class Intra_Assist_Inter(nn.Module):  # Intra: (N, 64, H, W)---- Inter: (1, 64, H, W)
	def __init__(self):
		super(Intra_Assist_Inter, self).__init__()
		self.dim_redunce = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU())

		self.CA = CALayer(channel=64)
		self.PA = PALayer(channel=64)

	def forward(self, Inter, Intra):   # Inter, Intra;;; Intra: (N, 64, H, W)---- Inter: (1, 64, H, W)
		feature_Intra = Intra[0].unsqueeze(0)
		#
		for jj in range(1, Intra.size(0)):
			feature_Intra = torch.cat([feature_Intra, Intra[jj].unsqueeze(0)], dim=1)
			#
		feature_Intra= Conv_Channel(feature_Intra, Resized_C=64)  # (1, 64, 256, 256)
		#
		Inter_Intra = torch.cat([feature_Intra, Inter], dim=1)
		#
		Inter_Intra_64 = self.dim_redunce(Inter_Intra)
		#
		Inter_Intra = self.CA(Inter_Intra_64)
		Inter_Intra = self.PA(Inter_Intra) + Inter_Intra_64

		return Inter_Intra

class NLEDN_IN_32_16_32(nn.Module):
	def __init__(self, input_nc, output_nc):
		super(NLEDN_IN_32_16_32, self).__init__()
		#
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU())
		#
		self.conv1_res = ResBlok(ch_in=64, ch_out=64)
		#
		self.conv_res_region = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU())
		#
		self.CA = CALayer(channel=64)
		'''编码器'''
		self.Ecoder_RDB_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_1_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_2_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_3_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_4_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		#
		self.Ecoder_RDB_encoder = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Ecoder_RDB_encoder_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		#
		'''解码器'''
		self.Decoder_RDB_5 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_6 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_7 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_8 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_5_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_6_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_7_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		self.Decoder_RDB_8_Groups = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)
		#
		self.trans_down1 = Down(64, 64)
		self.trans_down2 = Down(64, 64)
		self.trans_down3 = Down(64, 64)
		self.trans_down4 = Down(64, 64)
		self.trans_down1_Groups = Down(64, 64)
		self.trans_down2_Groups = Down(64, 64)
		self.trans_down3_Groups = Down(64, 64)
		self.trans_down4_Groups = Down(64, 64)
		#
		self.trans_up1 = Up(64, 64)
		self.trans_up2 = Up(64, 64)
		self.trans_up3 = Up(64, 64)
		self.trans_up4 = Up(64, 64)
		self.trans_up1_Groups = Up(64, 64)
		self.trans_up2_Groups = Up(64, 64)
		self.trans_up3_Groups = Up(64, 64)
		self.trans_up4_Groups = Up(64, 64)
		# ################## 转通道数
		self.down_4_fusion = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		self.down_4_fusion_Groups = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		#
		self.down_3_fusion = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		self.down_3_fusion_Groups = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		#
		self.down_2_fusion = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		self.down_2_fusion_Groups = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		#
		self.down_1_fusion = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		self.down_1_fusion_Groups = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU())
		####################
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, output_nc, 3, 1, 1),
			nn.Tanh())
		#
		self.M1 = Location_Mask()
		self.M2 = Location_Mask()
		self.M3 = Location_Mask()
		# ###############################
		self.Inter_Assist_Intra_256_Encoder = Inter_Assist_Intra()
		self.Inter_Assist_Intra_64_Encoder = Inter_Assist_Intra()
		self.Inter_Assist_Intra_16 = Inter_Assist_Intra()
		self.Inter_Assist_Intra_64_Decoder = Inter_Assist_Intra()
		self.Inter_Assist_Intra_256_Decoder = Inter_Assist_Intra()
		#
		self.Intra_Assist_Inter_128_Encoder = Intra_Assist_Inter()
		self.Intra_Assist_Inter_32_Encoder = Intra_Assist_Inter()
		self.Intra_Assist_Inter_32_Decoder = Intra_Assist_Inter()
		self.Intra_Assist_Inter_128_Decoder = Intra_Assist_Inter()
		# ###############################

	def forward(self, X_INPUT,  Depths_Mask01, X_Depths):   # 1, 3, 256, 256
		# ######################### Single-Image ############################
		# #########################  Region_Mask Intra-Image  #########################
		# X_Depths  [[Depth_map*input], [], []],
		for i in range(len(X_Depths)):  # N
			feature_sub = self.M1(X_Depths[i][0]) * Depths_Mask01[i][0] \
						  + self.M2(X_Depths[i][1]) * Depths_Mask01[i][1] \
						  + self.M3(X_Depths[i][2]) * Depths_Mask01[i][2]  # [1, 64, H, W]
			if i == 0:
				feature_region = feature_sub
			else:
				feature_region = torch.cat([feature_region, feature_sub], dim=0)
		# ########################### global features
		feature_0 = self.conv1(X_INPUT)  # 1, 3, 256, 256 -->  1, 64, 256, 256
		feature_res = self.conv1_res(feature_0)
		#
		feature_res_region = torch.cat([feature_res, feature_region], dim=1)
		feature_res_region = self.conv_res_region(feature_res_region)
		#
		feature_res_region = self.CA(feature_res_region)  # (2, 64, 256, 256)
		# ######################## Group-Image #############################
		feature_group = feature_res_region[0].unsqueeze(0)
		#
		for jj in range(1, feature_res_region.size(0)):
			feature_group = torch.cat([feature_group, feature_res_region[jj].unsqueeze(0)], dim=1)
			#
		feature_group = Conv_Channel(feature_group, Resized_C=64)  # (1, 64, 256, 256)
		#######################################################
		# Group-Image help Single-Image
		up_11 = self.Ecoder_RDB_1(feature_res_region)  # N, 64, 256, 256  -->  N, 64, 256, 256
		up_11_Groups = self.Ecoder_RDB_1_Groups(feature_group)  # 1, 64, 256, 256  -->  1, 64, 256, 256
		#
		up_11 = self.Inter_Assist_Intra_256_Encoder(up_11, up_11_Groups)  # N, 64, 256, 256
		#
		up_1 = self.trans_down1(up_11)   # N, 64, 256, 256  -->  N, 64, 128, 128
		up_1_Groups = self.trans_down1_Groups(up_11_Groups)   # 1, 64, 256, 256  -->  1, 64, 128, 128

		# Single-Image help Group-Image
		up_21 = self.Ecoder_RDB_2(up_1)  # N, 64, 128, 128  -->  N, 64, 128, 128
		up_21_Groups = self.Ecoder_RDB_2_Groups(up_1_Groups)  # 1, 64, 128, 128  -->  1, 64, 128, 128
		#
		up_21_Groups = self.Intra_Assist_Inter_128_Encoder(up_21_Groups, up_21)  # 1, 64, 128, 128
		#
		up_2 = self.trans_down2(up_21)    # 1, 128, 128, 128  -->  1, 256, 64, 64
		up_2_Groups = self.trans_down2_Groups(up_21_Groups)    # 1, 128, 128, 128  -->  1, 256, 64, 64

		# Group-Image help Single-Image
		up_31 = self.Ecoder_RDB_3(up_2)  # 1, 256, 64, 64  -->  1, 256, 64, 64
		up_31_Groups = self.Ecoder_RDB_3_Groups(up_2_Groups)  # 1, 256, 64, 64  -->  1, 256, 64, 64
		#
		up_31 = self.Inter_Assist_Intra_64_Encoder(up_31, up_31_Groups)
		#
		up_3 = self.trans_down3(up_31)    # 1, 256, 64, 64  -->  1, 512, 32, 32
		up_3_Groups = self.trans_down3_Groups(up_31_Groups)    # 1, 256, 64, 64  -->  1, 512, 32, 32

		# Single-Image help Group-Image
		up_41 = self.Ecoder_RDB_4(up_3)  # 1, 512, 32, 32  -->  1, 512, 32, 32
		up_41_Groups = self.Ecoder_RDB_4_Groups(up_3_Groups)  # 1, 512, 32, 32  -->  1, 512, 32, 32
		#
		up_41_Groups = self.Intra_Assist_Inter_32_Encoder(up_41_Groups, up_41)
		#
		up_4 = self.trans_down4(up_41)    # 1, 512, 32, 32  -->  1, 1024, 16, 16
		up_4_Groups = self.trans_down4_Groups(up_41_Groups)    # 1, 512, 32, 32  -->  1, 1024, 16, 16

		#######################################################
		#  Group-Image help Single-Image
		Latent_encoder = self.Ecoder_RDB_encoder(up_4)  # 1, 1024,8, 8  -->  1, 1024, 8, 8
		Latent_encoder_Groups = self.Ecoder_RDB_encoder_Groups(up_4_Groups)  # 1, 1024,8, 8  -->  1, 1024, 8, 8
		#
		Latent_encoder = self.Inter_Assist_Intra_16(Latent_encoder, Latent_encoder_Groups)
		#######################################################

		# Single-Image help Group-Image
		down_1 = self.trans_up1(Latent_encoder)  # 1, 1024, 16, 16 -->  1, 512, 32, 32
		down_1 = torch.cat([up_41, down_1], dim=1)  # cat [1, 512, 32, 32; 1, 512, 32, 32] --1024
		down_1 = self.down_1_fusion(down_1)     # 1, 1024, 32, 32 -->  1, 512, 32, 32
		down_1 = self.Decoder_RDB_5(down_1)       # 1, 512, 32, 32 -->  1, 512, 32, 32
		# ### Group-Image ###
		down_1_Groups = self.trans_up1_Groups(Latent_encoder_Groups)  # 1, 1024, 16, 16 -->  1, 512, 32, 32
		down_1_Groups = torch.cat([up_41_Groups, down_1_Groups], dim=1)  # cat [1, 512, 32, 32; 1, 512, 32, 32] --1024
		down_1_Groups = self.down_1_fusion_Groups(down_1_Groups)  # 1, 1024, 32, 32 -->  1, 512, 32, 32
		down_1_Groups = self.Decoder_RDB_5_Groups(down_1_Groups)  # 1, 512, 32, 32 -->  1, 512, 32, 32
		#
		down_1_Groups = self.Intra_Assist_Inter_32_Decoder(down_1_Groups, down_1)

		#   Group-Image help Single-Image
		down_2 = self.trans_up2(down_1)  # 1, 512, 32, 32 -->1, 256, 64, 64
		down_2 = torch.cat([up_31, down_2], dim=1)  # cat [1, 256, 32, 32; 1, 256, 32, 32] --512
		down_2 = self.down_2_fusion(down_2)     # 1, 512, 64, 64 -->1, 256, 64, 64
		down_2 = self.Decoder_RDB_6(down_2)       # 1, 256, 64, 64 -->1, 256, 64, 64
		# ### Group-Image ###
		down_2_Groups = self.trans_up2_Groups(down_1_Groups)  # 1, 512, 32, 32 -->1, 256, 64, 64
		down_2_Groups = torch.cat([up_31_Groups, down_2_Groups], dim=1)  # cat [1, 256, 32, 32; 1, 256, 32, 32] --512
		down_2_Groups = self.down_2_fusion_Groups(down_2_Groups)  # 1, 512, 64, 64 -->1, 256, 64, 64
		down_2_Groups = self.Decoder_RDB_6_Groups(down_2_Groups)  # 1, 256, 64, 64 -->1, 256, 64, 64
		#
		down_2 = self.Inter_Assist_Intra_64_Decoder(down_2, down_2_Groups)

		# Single-Image help Group-Image
		down_3 = self.trans_up3(down_2)  # 1, 256, 64, 64 --> 1, 128, 128, 128
		down_3 = torch.cat([up_21, down_3], dim=1)  # cat [1, 128, 128, 128; 1, 128, 128, 128] --256
		down_3 = self.down_3_fusion(down_3)     # 1, 256, 128,128 --> 1, 128, 128, 128
		down_3 = self.Decoder_RDB_7(down_3)     # 1, 128, 128,128 --> 1, 128, 128, 128
		# ### Group-Image ###
		down_3_Groups = self.trans_up3_Groups(down_2_Groups)  # 1, 256, 64, 64 --> 1, 128, 128, 128
		down_3_Groups = torch.cat([up_21_Groups, down_3_Groups], dim=1)  # cat [1, 128, 128, 128; 1, 128, 128, 128]--256
		down_3_Groups = self.down_3_fusion_Groups(down_3_Groups)     # 1, 256, 128,128 --> 1, 128, 128, 128
		down_3_Groups = self.Decoder_RDB_7_Groups(down_3_Groups)     # 1, 128, 128,128 --> 1, 128, 128, 128
		#
		down_3_Groups = self.Intra_Assist_Inter_128_Decoder(down_3_Groups, down_3)

		#   Group-Image help Single-Image
		down_4 = self.trans_up4(down_3)     # 1, 128, 128,128 --> 1, 64, 256, 256
		down_4 = torch.cat([up_11, down_4], dim=1)   # cat [1, 64, 256, 256; 1, 64, 256, 256] --128
		down_4 = self.down_4_fusion(down_4)  # 1, 128, 256, 256  -->  1, 64, 256, 256
		down_4 = self.Decoder_RDB_8(down_4)  # 1, 64, 256, 256  -->  1, 64, 256, 256
		# ### Group-Image ###
		down_4_Groups = self.trans_up4_Groups(down_3_Groups)     # 1, 128, 128,128 --> 1, 64, 256, 256
		down_4_Groups = torch.cat([up_11_Groups, down_4_Groups], dim=1)   # cat [1, 64, 256, 256; 1, 64, 256, 256] --128
		down_4_Groups = self.down_4_fusion_Groups(down_4_Groups)  # 1, 128, 256, 256  -->  1, 64, 256, 256
		down_4_Groups = self.Decoder_RDB_8_Groups(down_4_Groups)  # 1, 64, 256, 256  -->  1, 64, 256, 256
		#
		down_4 = self.Inter_Assist_Intra_256_Decoder(down_4, down_4_Groups)
		#
		feature = down_4 + feature_res_region  # 1, 64, 256, 256
		outputs = self.fusion(feature)
		return outputs
