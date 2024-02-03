import torch
import torch.nn as nn

class ResiduleBlock(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False)
        )

        if in_c != out_c:
            self.origin_conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        else:
            self.origin_conv = nn.Sequential()

        self.prelu3 = nn.PReLU()


    def forward(self, x):
        origin = x
        x = self.seq(x)
        origin = self.origin_conv(origin)
        x = torch.add(x, origin)

        x = self.prelu3(x)

        return x
    
class ResidualGroup(nn.Module):
    def __init__(self, inc, outc, length) -> None:
        super().__init__()

        self.first = ResiduleBlock(inc, outc)

        self.group = nn.Sequential(
            *[ResiduleBlock(outc, outc) for _ in range(length - 1)]
        )

    def forward(self, x):
        x = self.first(x)
        x = self.group(x)
        return x
    

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cmip = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding='same', bias=False),
            nn.PReLU(),
            ResidualGroup(64, 128, 8),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.PixelShuffle(4)  # channel=8
        )

        self.cmip_lulc = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding='same', bias=False),
            nn.PReLU(),
            ResiduleBlock(32, 32),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.PixelShuffle(4)  # channel=2
        )

        self.dem = nn.Sequential(
            nn.Conv2d(1, 2, 1, padding='same', bias=False),
            nn.PReLU()
        )

        self.combine = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding='same', bias=False),
            nn.PReLU(),
            ResiduleBlock(8, 8),
            nn.Conv2d(8, 1, 1, bias=False)
        )

        self.end = nn.Conv2d(1, 1, 1, bias=True)

    def forward(self, x):
        x, DEM = x     # LULC shape: (12, 240, 364)
        x = self.cmip(x)
        x = self.cmip_lulc(x)
        DEM = self.dem(DEM)
        x = torch.concatenate((x, DEM), axis=1)
        x = self.combine(x)
        x = self.end(x)
        return x


if __name__ == '__main__':
    net = NeuralNetwork()
    print(net)
