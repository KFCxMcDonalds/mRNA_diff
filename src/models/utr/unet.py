import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.unets.unet_1d import UNet1DModel


class UNet1DWithSoftmax(nn.Module):
    def __init__(self, config):
        super(UNet1DWithSoftmax, self).__init__()
        self.unet = UNet1DModel(
            sample_size = config.input_length,  # the input length of data
            in_channels = config.in_channels,  # the one-hot encoded data
            out_channels = config.out_channels,  # reconstructed channel of data (also 5, cuz we need gain a sequence)
            layers_per_block = config.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels = config.block_out_channels,  # block output channels on each side
            down_block_types = config.down_block_types,
            up_block_types = config.up_block_types
        )
        self.softmax = nn.Softmax(dim=1)  # apply to channels (batch, =>5, 512)

    def forward(self, x, timesteps, return_dict=False):
        x = self.unet(x, timesteps, return_dict=return_dict)[0]
        x = self.softmax(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, config):
        super(UNet1D, self).__init__()
        self.unet = UNet1DModel(
            sample_size = config.input_length,  # the input length of data
            in_channels = config.in_channels,  # the one-hot encoded data
            out_channels = config.out_channels,  # reconstructed channel of data (also 5, cuz we need gain a sequence)
            layers_per_block = config.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels = config.block_out_channels,  # block output channels on each side
            down_block_types = config.down_block_types,
            up_block_types = config.up_block_types
        )

    def forward(self, x, timesteps, return_dict=False):
        x = self.unet(x, timesteps, return_dict=return_dict)[0]
        return x

    def loss_function(self, model_output, target_oh, reduction='sum'):
        output_reshape = model_output.reshape(-1, 5)
        labels = target_oh.permute(0, 2, 1)
        label_indices = torch.argmax(labels, dim=2).view(-1)

        loss = F.cross_entropy(output_reshape,label_indices, reduction=reduction)

        return loss


