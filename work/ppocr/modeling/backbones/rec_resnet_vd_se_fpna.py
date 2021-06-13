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
from .det_mobilenet_v3 import SEModule
import numpy as np
__all__ = ["ResNetSEFPNA"]


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
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")
        self.se = SEModule(
            in_channels=out_channels * 4,
            reduction=4,
            name=name + "_se"
        )
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
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        
        conv3 = self.se(conv2)
        y = paddle.add(x=short, y=conv3)
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
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")
        self.se = SEModule(
            in_channels=out_channels,
            reduction=4,
            name=name + "_se"
        )
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
        conv2 = self.se(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class ResNetSEFPNA(nn.Layer):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNetSEFPNA, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            self.depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            self.depth = [3, 4, 6, 3]
        elif layers == 101:
            self.depth = [3, 4, 23, 3]
        elif layers == 152:
            self.depth = [3, 8, 36, 3]
        elif layers == 200:
            self.depth = [3, 12, 48, 3]
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
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
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
            for block in range(len(self.depth)):
                shortcut = False
                for i in range(self.depth[block]):
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
            for block in range(len(self.depth)):
                shortcut = False
                for i in range(self.depth[block]):
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
        self.fpn1 = ConvBNLayer(
            in_channels=768,
            out_channels=256,
            kernel_size=3,
            stride=(2,1),
            act='relu',
            name="fpn1")
        self.fpn2 = ConvBNLayer(
            in_channels=384,
            out_channels=128,
            kernel_size=3,
            stride=(4,1),
            act='relu',
            name="fpn2")
        self.out_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = 896

    def forward(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool2d_max(x)
        fpn_list = []
        fe = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))

        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    fe.append(x)
        agg_fe1 = (paddle.concat(                                                    # 768,4,160
                [F.upsample(fe[-1], scale_factor=(2,1),mode='nearest',align_mode=1), # 512,4,160
                fe[-2]],                                                             # 256,4,160
                axis=1)) 
        agg_fe2 = (paddle.concat(                                                    # 384,8,160
                [F.upsample(fe[-2], scale_factor=(2,1),mode='nearest',align_mode=1), # 256,8,160
                fe[-3]],                                                             # 128,8,160
                axis=1))
        agg_fe1 = self.fpn1(agg_fe1)                                                 # 768,4,160 -> 256,2,160
        agg_fe2 = self.fpn2(agg_fe2)                                                 # 384,8,160 -> 128,2,160
        out = paddle.concat([fe[-1],agg_fe1,agg_fe2],axis=1)                         # 896,2,160
        out = self.out_pool(out)                                                     # 896,1,80
        return out