import torch.nn as nn

class Seq2ImgEncoder(nn.Module):
    """
    Encoder use several consecutive conv&maxpool layers to convert 1D sequence tensor to 2D image tensor.
    increase its channels from 5 -> seq2img_num_channels, and create a W dimension to 1.
        note: we regard the nucleotides dimension as channel.
    input: [B, 5, L] represent: [batch_size, channel, length]
    output: [B, K, M, 1] represent: [batch_size, channel, height, width]
        unsqueeze the width dimension to 1, for Encoder_2d to process.
    """
    def __init__(self, in_channel, target_channel, num_layers):
        super().__init__()

        # ====FeatureLayerBlock====
        # only increase the channel to target_channel//2, not decrease the sequence length.
        self.feature_layer = ConvBlock(in_channel, target_channel//2, kernel_size=15)

        # ====LayerBlock====
        # increase the channel to target_channel and maxpool the sequence length to in_length/2^num_layers
        # increase the channel layer by layer.
        channel_list = compute_layer_channels(target_channel//2, target_channel, num_layers)
        layers = []
        for i, j in zip(channel_list[:-1], channel_list[1:]):
            layers.append(ConvBlock(i, j, kernel_size=3, sample_type='maxpool'))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x

def compute_layer_channels(in_channel, trg_channel, num_layers):
    """
    compute the input and output channels after feature layers (Layer Block), details should look at the paper figure. 
    """
    current_channel = in_channel
    channel_list = [current_channel]
    step = (trg_channel - in_channel) / num_layers
    for _ in range(num_layers):
        current_channel += step
        channel_list.append(int(current_channel))
    return channel_list


class Img2SeqDecoder(nn.Module):
    """
    input: [B, C, L]
    output: [B, 5, L]
    """
    def __init__(self, in_channel, target_channel, num_layers):
        super().__init__()

        # ====LayerBlock====
        channel_list = compute_layer_channels(target_channel//2, target_channel, num_layers)
        channel_list.reverse()
        layers = []
        for i, j in zip(channel_list[:-1], channel_list[1:]):
            layers.append(ConvBlock(i, j, kernel_size=3, sample_type='upsample'))
        self.layers = nn.Sequential(*layers)

        # ====FeatureLayerBlock====
        self.feature_layer = ConvBlock(target_channel//2, in_channel, kernel_size=9)

    def forward(self, x):
        x = self.layers(x)
        x = self.feature_layer(x)
        return x

class Img2LatentEncoder(nn.Module):
    """
    take in 2D image tensor and output latent variable z
    input: [B, 1, L, C]
    output: [B, W, L, C], where W = 2 * L
    """
    def __init__(self, in_width, hidden_width):
        super().__init__()
        modules = []
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
    
    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    """
    Convolution Block use maxPool + CNN to encoder DNA sequence to a two dimension vector.
        1. reduce the length dimension of DNA sequence by half
        2. use CNN to encode DNA sequence to increase the dimension of DNA sequence.
    Args:
        filter_list: [input channels, output channels]
        kernel_size: default [5], is a list because it's multikernel block.
        sample_type: 'maxpool' or 'upsample'
    """
    def __init__(self, in_channel, out_channel, kernel_size, sample_type=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.sample_type = sample_type
        self.transpose = False
        if self.sample_type == 'upsample':
            self.transpose = True

        # conv layer
        if self.transpose:
            self.conv_layer = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, padding=(kernel_size-1)//2)
        else:
            self.conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size, padding=(kernel_size-1)//2)
        # residual layer
        self.residual_conv_layer = nn.Conv1d(out_channel, out_channel, 1)
        # conv block
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=in_channel, num_channels=in_channel),
            nn.GELU(),
            self.conv_layer
        )

        # maxpool or upsample
        if self.sample_type == 'maxpool':
            self.sample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.sample_type == 'upsample':
            self.sample_layer = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv_block(x)
        add = x
        x = self.residual_conv_layer(x)
        x = x + add

        if self.sample_type != None:
            x = self.sample_layer(x)
        return x 

# TODO: multikernel
# TODO: add residual connection in latent encoder/decoder
