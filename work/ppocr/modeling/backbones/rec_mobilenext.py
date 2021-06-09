# Code reference from https://github.com/zhoudaquan/rethinking_bottleneck_design/blob/master/mobilenext/codebase/models/mnext.py

import paddle
import paddle.nn as nn
import math

__all__ = ['MobileNeXt',]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

def group_conv_1x1_bn(inp, oup, expand_ratio):
    hidden_dim = oup // expand_ratio
    return nn.Sequential(
        nn.Conv2D(inp, hidden_dim, 1, 1, 0, groups=hidden_dim, bias_attr=False),
        nn.BatchNorm2D(hidden_dim),
        nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

class SGBlock(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2, (2,1)]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)# + 16

        #self.relu = nn.ReLU6()
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                nn.BatchNorm2D(inp),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                nn.Conv2D(oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                nn.BatchNorm2D(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU6(),
            )
        elif inp != oup and stride != 1 and keep_3x3==False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                nn.Conv2D(oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                nn.BatchNorm2D(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                nn.BatchNorm2D(inp),
                nn.ReLU6(),
                # pw
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                #nn.ReLU6(),
                # pw
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                nn.Conv2D(oup, oup, 3, 1, 1, groups=oup, bias_attr=False),
                nn.BatchNorm2D(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            shape = x.shape
            id_tensor = x[:,:shape[1]//self.identity_div,:,:]
            # id_tensor = torch.cat([x[:,:shape[1]//self.identity_div,:,:],torch.zeros(shape)[:,shape[1]//self.identity_div:,:,:].cuda()],dim=1)
            # import pdb; pdb.set_trace()
            out[:,:shape[1]//self.identity_div,:,:] = out[:,:shape[1]//self.identity_div,:,:] + id_tensor
            return out #+ x
        else:
            return out

class MobileNeXt(nn.Layer):
    def __init__(self, in_channels=3, scale=1.):
        super(MobileNeXt, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [2,  96, 1, 1],
            [6, 144, 1, 1],
            [6, 192, 3, (2,1)],
            [6, 288, 3, (2,1)],
            [6, 384, 4, 1],
            [6, 576, 4, (2,1)],
            [6, 960, 3, 1],
            [6,1280, 1, 1],
        ]
        #self.cfgs = [
        #    # t, c, n, s
        #    [1,  16, 1, 1],
        #    [4,  24, 2, 2],
        #    [4,  32, 3, 2],
        #    [4,  64, 3, 2],
        #    [4,  96, 4, 1],
        #    [4, 160, 3, 2],
        #    [4, 320, 1, 1],
        #]

        # building first layer
        input_channel = _make_divisible(32 * scale, 4 if scale == 0.1 else 8)
        layers = [conv_3x3_bn(in_channels, input_channel, 2)]
        # building inverted residual blocks
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * scale, 4 if scale == 0.1 else 8)
            if c == 1280 and scale < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n==1 and s==1))
            input_channel = output_channel
            for i in range(n-1):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel
        self.out_channels = _make_divisible(input_channel, 4) # if scale == 0.1 else 8) if scale > 1.0 else input_channel
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x
