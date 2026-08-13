import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_batchnorm=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_ch))

        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(-1, nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HidingUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_residual = config["use_residual"]
        base = config["base_channels"]
        depth = config["depth"]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        use_bn = config["use_batchnorm"]

        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        chs = base
        self.downs.append(DoubleConv(in_channels, chs, use_bn))

        for _ in range(depth - 1):
            self.pools.append(nn.MaxPool2d(2))
            self.downs.append(DoubleConv(chs, chs * 2, use_bn))
            chs *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(chs, chs * 2, use_bn)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(chs * 2, chs, 2, stride=2))
            self.up_convs.append(DoubleConv(chs * 2, chs, use_bn))
            chs //= 2

        self.final = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, cover, secret):

        x = torch.cat([cover, secret], dim=1)

        skip_connections = []

        for i in range(len(self.downs)):
            x = self.downs[i](x)
            skip_connections.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[i]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.up_convs[i](x)

        out = self.final(x)

        if self.use_residual:
            return cover + out

        return out