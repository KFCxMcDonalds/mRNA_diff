import torch
import torch.nn as nn
import torch.nn.functional as F

from .sub_models.seq2img_models_old import Seq2ImgEncoder, Img2SeqDecoder, Img2LatentEncoder, Latent2ImgDecoder

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for UTR sequences

    Input: [Bx128x5] tensor
    Output: [Bx128x5] tensor
    """
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config 

        # ============================
        # ========= Encoder ==========
        # ============================
        # sequence to image
        # 1D Encoder: takes in sequence data and reformat representation to 2D representation.
        self.Encoder_1d = Seq2ImgEncoder(self.config.in_channel, self.config.seq2img_img_channels, self.config.seq2img_num_layers)

        # 2D Encoder: takes in 2D representation and output latent variable z
        self.Encoder_2d = Img2LatentEncoder(self.config.in_width, self.config.hidden_width)
        
        self.latent_encoder = nn.Conv2d(self.config.hidden_width[-1], self.config.hidden_width[-1], kernel_size=1)

        # ============================
        # ========= Decoder ==========
        # ============================
        self.latent_decoder = nn.Conv2d(self.config.hidden_width[-1]//2, self.config.hidden_width[-1], kernel_size=1)
        # 2D Decoder: takes in latent 2D representation and output sequence 2D representaion 
        self.config.hidden_width.reverse()
        self.config.hidden_width.append(1)
        self.Decoder_2d = Latent2ImgDecoder(self.config.hidden_width)
        # 1D Decoder: takes in sequence 2D representation and output sequence 1D representation.
        self.Decoder_1d = Img2SeqDecoder(self.config.in_channel, self.config.seq2img_img_channels, self.config.seq2img_num_layers)

        # ============================
        # ====== final layer =========
        # ============================
        self.tanh = nn.Tanh()
    
    def encoder(self, x):
        # 1d encoder input: [B, 5, L]
        x = self.Encoder_1d(x)
        # print("after encoder 1d:", x.shape)
        # 1d encoder output: [B, C, L], and C=L=128

        # 2d encoder input: [B, 1, L, C]
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        z = self.Encoder_2d(x)
        # print("after encoder 2d:", z.shape)
        
        # intermediate
        z = self.latent_encoder(z)
        # print("after latent encoder:", z.shape)
        # 2d encoder output: [B, W, L, C] = [B, ]

        # split z to mu and logvar
        mean, logvar = torch.chunk(z, 2, dim=1)  # dim W
        logvar = torch.clamp(logvar, -30.0, 20.0)

        return mean, logvar


    def decoder(self, z):
        # convert z to same shape as output of encoder2d, which contains both mu and logvar
        z = self.latent_decoder(z)
        # print("after latent decoder:", z.shape)

        # 2d decoder
        y  = self.Decoder_2d(z)
        # print("after decoder 2d:", y.shape)
        y = self.tanh(y)
        # print("after tanh:", y.shape)

        # 1d decoder
        y = y.squeeze(1).permute(0, 2, 1)  # to [B, C, L]
        y = self.Decoder_1d(y)
        # print("after decoder 1d:", y.shape)

        return y 

    def reparametrize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar).to(self.config.device)
        eps = torch.randn_like(std).to(self.config.device)
        return z_mean + std * eps
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return [self.decoder(z), mu, logvar]

    def generate(self, x):
        return self.forward(x)[0]

    def loss_function(self, x, x_recon, z_mean, z_logvar, kld_weight):
        # KL + CrossEntropy
        # KL divergence: KL(N(mu, std) || N(0, 1))
        kld = 0.5 * torch.sum(torch.pow(z_mean, 2) + torch.exp(z_logvar) - z_logvar - 1, dim=[1, 2, 3])
        kld_loss = kld.sum()

        # CrossEntropy: reconstruction loss
        ce = F.cross_entropy(x_recon.reshape(-1, 5), x.reshape(-1, 5), reduction='sum')

        loss = kld_weight * kld_loss + ce
        return loss, kld_loss, ce


