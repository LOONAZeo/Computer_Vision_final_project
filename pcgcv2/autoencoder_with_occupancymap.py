import torch
import MinkowskiEngine as ME

from data_utils import isin, istopk

class get_coordinate(torch.nn.Module):
    def __init__(self):
        super(get_coordinate, self).__init__()
        self.SumPooling = ME.MinkowskiSumPooling(kernel_size=3, stride=2, dilation=1, dimension=2)

    def forward(self, input):
        coordinate1 = self.SumPooling(input)
        coordinate2 = self.SumPooling(coordinate1)
        coordinate3 = self.SumPooling(coordinate2)
        return coordinate2, coordinate3


class Encoder(torch.nn.Module):
    def __init__(self, channels=[1, 64, 128]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)

        self.relu = ME.MinkowskiReLU(inplace=True)

        # self.NLAM = NLAM(channels[2], channels[2])

    def forward(self, x):
        # no IRN
        out0 = self.relu(self.down0(self.conv0(x)))
        out1 = self.relu(self.down1(self.conv1(out0)))
        out2 = self.down2(self.conv2(out1))
        out_cls_list = [out2, out1, out0]

        # # no IRN NALM
        # out0 = self.NLAM1(self.relu(self.down0(self.conv0(x))))
        # out1 = self.relu(self.down1(self.conv1(out0)))
        # out2 = self.NLAM2(self.down2(self.conv2(out1)))
        # out_cls_list = [out2, out1, out0]

        return out_cls_list


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[1, 64, 128]):
        super().__init__()
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=2)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=2)

        self.relu = ME.MinkowskiReLU(inplace=True)

        # self.NLAM = NLAM(channels[2], channels[2])

    def forward(self, x):
        # no IRN
        out2 = self.relu(self.conv0(self.up0((x))))
        out1 = self.relu(self.conv1(self.up1(out2)))
        out0 = self.conv2(self.up2(out1))

        # # no IRN NALM
        # out2 = self.relu(self.conv0(self.up0(self.NLAM1(x))))
        # out1 = self.relu(self.conv1(self.up1(out2)))
        # out0 = self.conv2(self.up2(self.NLAM2(out1)))

        out_cls_list = [out2, out1, out0]

        # return out_cls_list, out0
        return out0
