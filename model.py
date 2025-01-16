import torch
import torch.nn as nn
import torch.nn.functional as F
# Denoise Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super(DenoisingAutoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            # use same padding to keep the input size
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # record the input size
        orig_size = x.shape[-2:]

        # encoder - decoder
        x = self.encoder(x)
        x = self.decoder(x)

        if x.shape[-2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)

        return x

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1):
        super(UNetGenerator, self).__init__()

        # Down sampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Up sampling path
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               output_padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):

        d1 = self.down1(x)

        d2 = self.down2(d1)

        d3 = self.down3(d2)

        bottle = self.bottleneck(d3)

        if bottle.shape[2:] != d3.shape[2:]:
            bottle = F.interpolate(bottle, size=d3.shape[2:], mode='bilinear', align_corners=False)

        u3 = self.up3(torch.cat([bottle, d3], dim=1))

        if u3.shape[2:] != d2.shape[2:]:
            u3 = F.interpolate(u3, size=d2.shape[2:], mode='bilinear', align_corners=False)

        u2 = self.up2(torch.cat([u3, d2], dim=1))
        if u2.shape[2:] != d1.shape[2:]:

            u2 = F.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=False)

        u1 = self.up1(torch.cat([u2, d1], dim=1))

        if u1.shape[2:] != x.shape[2:]:
            u1 = F.interpolate(u1, size=x.shape[2:], mode='bilinear', align_corners=False)

        return u1

# discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize=True):
            """conns a discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters,
                                kernel_size=(7, 7),
                                stride=(stride, stride),
                                padding=(3, 3))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            # (B, 1, H, W) -> (B, 64, H/2, W/2)
            *discriminator_block(in_channels, 64, 2, normalize=False),

            # (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
            *discriminator_block(64, 128, 2),

            # (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
            *discriminator_block(128, 256, 2),

            # (B, 256, H/8, W/8) -> (B, 512, H/16, W/16)
            *discriminator_block(256, 512, 2),

            # (B, 512, H/16, W/16) -> (B, 1, H/16, W/16)
            nn.Conv2d(512, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

