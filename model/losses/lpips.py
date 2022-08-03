"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import os
import torch
import torch.nn as nn
from model.losses.vgg16 import VGG16

CUR_DIR = os.path.dirname(__file__)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        # set pretrained False, as we will load our own pretrained model
        self.net = VGG16(pretrained=False, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        ckpt = os.path.join(CUR_DIR, 'vgg.pth')
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("Loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]

        loss = torch.mean(val)

        return loss


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x: torch.Tensor):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


if __name__ == "__main__":
    ipt = torch.rand((8, 3, 128, 128))
    tgt = torch.rand((8, 3, 128, 128))

    lpips = LPIPS().eval()
    l_diff = lpips(ipt, tgt)
    l_same = lpips(ipt, ipt)

    print("Loss different images:", l_diff.item())
    print("Loss same images:", l_same.item())
