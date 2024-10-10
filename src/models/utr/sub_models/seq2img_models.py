import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_

from utils.layer_helper import compute_layer_channels_exp, compute_layer_channels_linear

class Seq2ImgEncoder(nn.Module):
    """
    Encoder use several consecutive conv&maxpool layers to convert 1D sequence tensor to 2D image tensor.
    increase its channels from 5 -> seq2img_num_channels, and create a W dimension to 1.
        note: we regard the nucleotides dimension as channel.
    input: [B, 5, L] represent: [batch_size, channel, length]
    output: [B, K, M, 1] represent: [batch_size, channel, height, width]
        unsqueeze the width dimension to 1, for Encoder_2d to process.
    """
    def __init__(self, in_channel, target_channel, num_feature_layers, num_layers):
        super().__init__()

        # ====FeatureLayerBlock====
        # only increase the channel to target_channel//2, not decrease the sequence length.
        feature_channel_list = compute_layer_channels_linear(in_channel, target_channel//2, num_feature_layers)
        feature_layers = []
        for i, j in zip(feature_channel_list[:-1], feature_channel_list[1:]):
            feature_layers.append(Conv1DBlock(i, j, kernel_sizes=[15]))  # can be multikernel
        self.feature_layer = nn.Sequential(*feature_layers)

        # ====LayerBlock====
        # increase the channel to target_channel and maxpool the sequence length to in_length/2^num_layers
        # increase the channel layer by layer.
        channel_list = compute_layer_channels_exp(target_channel//2, target_channel, num_layers, divide=2)
        layers = []
        # maxpool at the third layer
        layer_count = 0
        for i, j in zip(channel_list[:-1], channel_list[1:]):
            if layer_count == 2:
                layers.append(Conv1DBlock(i, j, kernel_sizes=[5], sample_type='maxpool'))
            else:
                layers.append(Conv1DBlock(i, j, kernel_sizes=[5]))
            layer_count += 1
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x

class Img2SeqDecoder(nn.Module):
    """
    input: [B, C, L]
    output: [B, 5, L]
    """
    def __init__(self, in_channel, target_channel, num_feature_layers, num_layers):
        super().__init__()

        # ====LayerBlock====
        channel_list = compute_layer_channels_exp(target_channel//2, target_channel, num_layers, divide=2)
        channel_list.reverse()
        layers = []
        count = 0
        for i, j in zip(channel_list[:-1], channel_list[1:]):
            if count == 1:
                layers.append(Conv1DBlock(i, j, kernel_sizes=[5], sample_type='upsample'))
            else:
                layers.append(Conv1DBlock(i, j, kernel_sizes=[5]))
            count += 1
        self.layers = nn.Sequential(*layers)

        # ====FeatureLayerBlock====
        feature_channel_list = compute_layer_channels_linear(in_channel, target_channel//2, num_feature_layers)
        feature_channel_list.reverse()
        feature_layers = []
        for i, j in zip(feature_channel_list[:-1], feature_channel_list[1:]):
            feature_layers.append(Conv1DBlock(i, j, kernel_sizes=[15]))
        self.feature_layer = nn.Sequential(*feature_layers)

    def forward(self, x):
        x = self.layers(x)
        x = self.feature_layer(x)
        return x

class Img2LatentEncoder(nn.Module):
    """
    take in 2D image tensor and output latent variable z
    input: [B, 1, L, C]
    output: [B, W, L, C], where W = 2 * L, contained both mu and logvar
    """
    def __init__(self, hidden_width):
        super().__init__()
        modules = []
        in_width = 1
        for out_width in hidden_width:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_width, out_width, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_width),
                    nn.LeakyReLU()
                )
            )
            in_width = out_width
        self.layers = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.layers(x)

class Latent2ImgDecoder(nn.Module):
    """
    input: [B, W, L, C], here W = 2 * L, same shape as output of Img2LatentEncoder
    output: [B, 1, L, C]
    var:
        hidden_width: a list of int, the width of the latent variable z
    """
    def __init__(self, hidden_width):
        super().__init__()
        modules = []
        for i in range(len(hidden_width)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_width[i],
                                       hidden_width[i+1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_width[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.layers = nn.Sequential(*modules)
        # last conv2d layer: hidden_width[-1] -> 1, activation function is tanh
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_width[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
#             nn.BatchNorm2d(1),
#             nn.LeakyReLU()
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = self.final_layer(x)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes, sample_type=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_sizes = kernel_sizes
        self.sample_type = sample_type
        self.transpose = False
        if self.sample_type == 'upsample':
            self.transpose = True

        self.multikernel_conv = MultiKernelConv1DBlock(in_channel, out_channel, kernel_sizes, self.transpose)
        self.conv_residual = MultiKernelConv1DBlock(out_channel, out_channel, [1], self.transpose)

        if self.sample_type == 'maxpool':
            self.sample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.sample_type == 'upsample':
            self.sample_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # upsample
        if self.sample_type == 'upsample':
            x = self.sample_layer(x)
        x = self.multikernel_conv(x)  # (out_channel, length)
        # residual
        add_ = x
        x = self.conv_residual(x)
#         x = self.dropout(x)
        x = x + add_

        # maxpool
        if self.sample_type == 'maxpool':
            x = self.sample_layer(x)

        return x

class MultiKernelConv1DBlock(nn.Module):
    """
    if kernel_sizes contains more than one kernel size, then use different kernel size to do conv.
    if kernel_sizes contains only one kernel size, then use the normal conv1D.
    """
    def __init__(self, in_channel, out_channel, kernel_sizes, transpose=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_sizes = kernel_sizes
        self.transpose = transpose

        if in_channel > out_channel:
            self.transpose = True

        # decoder blcok
        if self.transpose:
            self.conv_layers = nn.ModuleList([
                nn.ConvTranspose1d(in_channel, 
                                   out_channel,
                                   kernel_size=kernel_size,
                                   padding=(kernel_size-1)//2)  # not change the length
                                   for kernel_size in self.kernel_sizes])
        else:  # encoder block
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(in_channel, 
                          out_channel,
                          kernel_size=kernel_size,
                          padding=(kernel_size-1)//2)  # not change the length
                          for kernel_size in self.kernel_sizes])

        self.conv_down = nn.Conv1d(len(self.kernel_sizes)*out_channel, out_channel, kernel_size=1)

        self.norm = nn.GroupNorm(in_channel, in_channel)
        self.gelu = nn.GELU()

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.gelu(x)

        x = torch.cat([conv(x) for conv in self.conv_layers], dim=1)  # channel to len(kernel_sizes)*out_channel
        if len(self.kernel_sizes) > 1:
            x = self.conv_down(x)  # reshape channel back to out_channel
        return x
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)


class Conv1DAttnBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes, attn_embed_dim, sample_type=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_sizes = kernel_sizes
        self.sample_type = sample_type
        self.attn_embed_dim = attn_embed_dim
        self.transpose = False
        if self.sample_type == 'upsample':
            self.transpose = True

        self.multikernel_conv = MultiKernelConv1DBlock(in_channel, out_channel, kernel_sizes, self.transpose)
        self.conv1 = nn.Conv1d(len(kernel_sizes)*out_channel, out_channel, kernel_size=1)

        self.conv_up= nn.Conv1d(out_channel, attn_embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(attn_embed_dim, num_heads=4)
        self.conv_down = nn.Conv1d(attn_embed_dim, out_channel, kernel_size=1)

        self.Norm = nn.GroupNorm(num_groups=out_channel, num_channels=out_channel)

        if self.sample_type == 'maxpool':
            self.sample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.sample_type == 'upsample':
            self.sample_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.multikernel_conv(x)  # (3*inchannel, length)
        x = self.conv1(x)  # (outchannel, length)

        # attention
        add_ = x
        x = self.conv_up(x)
#         x = self.gelu(x)
        x = x.permute(2, 0, 1)
        x = self.attn(x, x, x)[0]
        x = x.permute(1, 2, 0)
#         x = self.gelu(x)
        x = self.conv_down(x)
        x = x + add_

        # maxpool or upsample
        if self.sample_type:
            x = self.sample_layer(x)

        x = self.Norm(x)
        return self.gelu(x)


