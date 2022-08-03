import torch
import torch.nn as nn
import torchvision.models as tvm
from collections import namedtuple


VGG_Outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])


class VGG16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        weights = tvm.VGG16_Weights.DEFAULT if pretrained else None
        vgg_pretrained_features = tvm.vgg16(weights=weights).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h

        h = self.slice2(h)
        h_relu2_2 = h

        h = self.slice3(h)
        h_relu3_3 = h

        h = self.slice4(h)
        h_relu4_3 = h

        h = self.slice5(h)
        h_relu5_3 = h

        res = VGG_Outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return res


if __name__ == "__main__":
    ipt = torch.randn((16, 3, 64, 64))

    vgg = VGG16()
    out = vgg(ipt)

    print("Input:", ipt.shape)
    print("Output:")
    for o in out:
        print("\t", o.shape)
