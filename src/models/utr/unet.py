import torch.nn as nn
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
        self.softmax = nn.Softmax(dim=1)  # apply to channels (=>5, 512)

    def forward(self, x, timesteps, return_dict=False):
        x = self.unet(x, timesteps, return_dict=return_dict)[0]
        x = self.softmax(x)
        return x

