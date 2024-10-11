# this file is only for inference convenience
from diffusers.models.unets.unet_2d import UNet2DModel


class UNet2D(UNet2DModel):
    def __init__(self, config, **kwargs):
        super().__init__(sample_size= config.sample_size,
                        in_channels= config.in_channels,
                        out_channels= config.out_channels,
                        layers_per_block= config.layers_per_block,
                        block_out_channels= config.block_out_channels,
                        down_block_types= config.down_block_types,
                        up_block_types= config.up_block_types,
                        **kwargs)