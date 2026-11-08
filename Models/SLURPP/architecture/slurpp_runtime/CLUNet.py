# Authors: Mingyang Xie, Tianfu Wang
from .myvae import Encoder, Decoder
import torch.nn as nn
import copy
import json
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, DecoderOutput
from diffusers.utils import BaseOutput
import torch

class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821
    composite_latents: list = None



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = zero_module(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
    def forward(self, x):
        return self.conv(x)


class CrossLatentUNet(nn.Module):
    def __init__(self,skip_connection=True,mid_control=False,residual=False,config_path=""):
        super(CrossLatentUNet,self).__init__()

        with open(config_path) as f:
            config = json.load(f)

        self.scale = 0.18215

        self.quant_conv = nn.Conv2d( 2 * config["latent_channels"], 2 * config["latent_channels"], 1)
        self.post_quant_conv = nn.Conv2d(config["latent_channels"], config["latent_channels"], 1)
        self.Decoder = Decoder(in_channels=config["latent_channels"],
            out_channels=config["out_channels"],
            up_block_types=config["up_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"])
        self.Encoder = Encoder(in_channels=config["in_channels"],
            out_channels=config["latent_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"],
            double_z=True,)
        self.residual = residual
        self.skip_connection = skip_connection
        self.mid_control = mid_control
        self.dtype = torch.float32

        self.zero_conv_0 = ZeroConv(128, 256) # 修改这里的channel数量
        self.zero_conv_1 = ZeroConv(128, 512) # 修改这里的channel数量
        self.zero_conv_2 = ZeroConv(256, 512) # 修改这里的channel数量
        self.zero_conv_3 = ZeroConv(512, 512) # 修改这里的channel数量

        if self.mid_control:
            self.zero_conv_4 = ZeroConv(8, 4) # 修改这里的channel数量

    def load_vae(self, autoencoder, encoder_only=False):
        print("load encoder and decoder from autoencoder")
        self.quant_conv.load_state_dict(copy.deepcopy(autoencoder.quant_conv.state_dict()), strict=True)
        self.post_quant_conv.load_state_dict(copy.deepcopy(autoencoder.post_quant_conv.state_dict()), strict=True)
        self.Encoder.load_state_dict(copy.deepcopy(autoencoder.encoder.state_dict()), strict=True)
        if not encoder_only:
            self.Decoder.load_state_dict(copy.deepcopy(autoencoder.decoder.state_dict()), strict=True)

    def encode(self, x, skip_connection=False, return_dict=True):
        # print(f"x.shape: {x.shape}")
        # print(f"skip_connection: {skip_connection}")

        out = self.Encoder(x, skip_connection=skip_connection, mid_control=skip_connection)

        if skip_connection:
            enc = out[-1]
            composite_latents = out[:-1]
        else:
            enc = out

        if self.quant_conv is not None:
            enc = self.quant_conv(enc)
        # print(f"enc.shape: {enc.shape}")

        posterior = DiagonalGaussianDistribution(enc)

        if not skip_connection:
            if not return_dict:
                return (posterior, )
            else:
                return AutoencoderKLOutput(latent_dist=posterior)
        else:
            return AutoencoderKLOutput(latent_dist=posterior, composite_latents=composite_latents)

    def pass_zero_conv(self, composite_latents):
        composite_latents[0] = self.zero_conv_0(composite_latents[0])
        composite_latents[1] = self.zero_conv_1(composite_latents[1])
        composite_latents[2] = self.zero_conv_2(composite_latents[2])
        composite_latents[3] = self.zero_conv_3(composite_latents[3])
        return composite_latents

    def decode(self, x, composite_latents=None, return_dict=True):

        if self.post_quant_conv is not None:
            x = self.post_quant_conv(x)
        dec =  self.Decoder(x, composite_latents=composite_latents, mid_control=self.mid_control)

        if not return_dict:
            return (dec, )
        else:
            return DecoderOutput(sample=dec)


    def forward(self, denoised_latents, composite_img):
        denoised_latents  = denoised_latents / self.scale

        denoised_latents = self.post_quant_conv(denoised_latents)

        composite_latent_list = self.Encoder(composite_img,
                                             mid_control=self.mid_control)

        '''
        0: [1,128, 768, 768], encoder: after ConvIn; decoder: after 3rd UpBlock, [1, 256, 768, 768]
        1: [1,128, 384, 384], encoder: after 1st DownBlock; decoder: after 2nd UpBlocks, [1, 512, 384, 384]
        2: [1,256, 192, 192], encoder: after 2nd DownBlock; decoder: after 1st UpBlock, [1, 512, 192, 192]
        3: [1,512, 96, 96], encoder: after 3rd DownBlock; decoder: after MidBlock, [1, 512, 96, 96]
        4. [1,8, 96, 96], encoder: final output; decoder: input
        '''

        if self.skip_connection:
            composite_latent_list[0] = self.zero_conv_0(composite_latent_list[0])
            composite_latent_list[1] = self.zero_conv_1(composite_latent_list[1])
            composite_latent_list[2] = self.zero_conv_2(composite_latent_list[2])
            composite_latent_list[3] = self.zero_conv_3(composite_latent_list[3])

            if self.mid_control:
                composite_latent_list[4] = self.quant_conv(composite_latent_list[4])
                composite_latent_list[4] = self.zero_conv_4(composite_latent_list[4])

        restored_img = self.Decoder(denoised_latents,
                                    composite_latents=composite_latent_list,
                                    mid_control=self.mid_control)

        if self.residual:
            return composite_img - restored_img

        return restored_img
