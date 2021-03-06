# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
__all__ = ["RedNet"]


class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = involution(out_channels, 7, stride, name=name + "_invo_branch2b")
        if name == "conv1":
            bn_name = "bn_" + name + "_invo_branch2b"
        else:
            bn_name = "bn" + name[3:] + "_invo_branch2b"
        self.bn1 = nn.BatchNorm(
            out_channels,
            act='relu',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        # self.conv1 = ConvBNLayer(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     stride=stride,
        #     act='relu',
        #     name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)

        conv1 = self.conv1(y)
        conv1 = self.bn1(conv1)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y

class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.invo = in_channels == out_channels
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")

        self.conv1 = involution(out_channels, 7, (1, 1), name=name + "_invo_branch2b")
        if name == "conv1":
            bn_name = "bn_" + name + "_invo_branch2b"
        else:
            bn_name = "bn" + name[3:] + "_invo_branch2b"
        self.bn1 = nn.BatchNorm(
            out_channels,
            act=None,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        # self.conv1 = ConvBNLayer(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     act=None,
        #     name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv1 = self.bn1(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y

class RedNet(nn.Layer):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(RedNet, self).__init__()

        self.layers = layers
        supported_layers = [26, 38, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 26:
            depth = [1, 2, 4, 1]
        elif layers == 38:
            depth = [2, 3, 5, 2]
        elif layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_1")

        self.conv1_2 = involution(32, 3, (1, 1), name="conv1_2")
        self.bn1_2 = nn.BatchNorm(
            32,
            act='relu',
            param_attr=ParamAttr(name='conv1_2_bn_scale'),
            bias_attr=ParamAttr('conv1_2_bn_offset'),
            moving_mean_name='conv1_2_bn_mean',
            moving_variance_name='conv1_2_bn_variance')

        # self.conv1_2 = ConvBNLayer(
        #     in_channels=32,
        #     out_channels=32,
        #     kernel_size=3,
        #     stride=1,
        #     act='relu',
        #     name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152, 200] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    self.block_list.append(bottleneck_block)
                self.out_channels = num_filters[block] * 4
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    self.block_list.append(basic_block)
                self.out_channels = num_filters[block]
        self.out_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.bn1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.out_pool(y)
        return y

class involution(nn.Layer):

    def __init__(self,
                 channels,
                 kernel_size=3,
                 stride=1,
                 reduction_ratio=4,
                 group_channels=16,
                 name=None):
        super(involution, self).__init__()
        # Save parameters
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        # Init model
        self.reduce = nn.Sequential(
            nn.Conv2D(channels, channels // reduction_ratio, 1, bias_attr=False),
            nn.BatchNorm(channels // reduction_ratio),
            nn.ReLU())
        self.span = nn.Conv2D(channels // reduction_ratio, kernel_size * kernel_size * self.groups, 1)
        if stride != (1,1):
            self.avgpool = nn.AvgPool2D(stride)

    def forward(self, x):
        weight = self.span(self.reduce(x if not hasattr(self, 'avgpool') else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = paddle.reshape(weight, shape=[b, self.groups, self.kernel_size**2, h, w]).unsqueeze(2)
        out = F.unfold(x, self.kernel_size, dilations=1, paddings=(self.kernel_size-1)//2, strides=list(self.stride))
        out = paddle.reshape(out, shape=[b, self.groups, self.group_channels, self.kernel_size**2, h, w])
        out = paddle.sum((weight * out), axis=3)
        out = paddle.reshape(out, shape=[b, self.channels, h, w])
        return out

if __name__ == "__main__":
    # Test involution model
    inv = involution(128, 7, (2,1))

    paddle.summary(inv, (1, 128, 64, 64))

    out = inv(paddle.randn((1, 128, 64, 64)))

    print(out.shape)

    # Test RedNet
    model = RedNet(layers=26)

    paddle.summary(model, (1, 3, 32, 160))

    out = model(paddle.randn((1, 3, 32, 160)))

    print(out.shape)