# Reference from https://github.com/huiyang865/plugnet/blob/master/models/rcan.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

__all__ = ['RCAN']

def make_model(args, parent=False):
    return RCAN(args)

def default_conv(in_channels, out_channels, kernel_size):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2))

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2D(channel, channel // reduction, 1, padding=0),
                nn.ReLU(),
                nn.Conv2D(channel // reduction, channel, 1, padding=0),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size))
            if bn: modules_body.append(nn.BatchNorm(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Layer):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Layer):
    def __init__(self, in_channels, conv=default_conv):
        super(RCAN, self).__init__()
        
        # n_resgroups = 10
        # n_resblocks = 20
        n_resgroups = 2
        n_resblocks = 2
        n_feats = 64
        kernel_size = 3
        reduction = 16
        act = nn.ReLU(True)

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        scale = (32,4)
        modules_tail = [
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            conv(n_feats, n_feats, kernel_size),
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            conv(n_feats, n_feats, kernel_size),
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            conv(n_feats, n_feats, kernel_size),
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            conv(n_feats, n_feats, kernel_size),
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            conv(n_feats, 3, kernel_size)]

        self.conv1 = conv(in_channels, n_feats, kernel_size=1)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.conv1(x)
        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    rcan = RCAN(in_channels=3)
    paddle.summary(rcan, (1,3,32,320))
    pass