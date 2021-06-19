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

import paddle
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible
import math
__all__ = ['MobileNetV3_FPND']


class MobileNetV3_FPND(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 fe_idx=None,
                 **kwargs):
        super(MobileNetV3_FPND, self).__init__()
        if small_stride is None:
            small_stride = [1, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 1],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            if fe_idx is None:
                fe_idx = [0,1,3,12]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0])],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            if fe_idx is None:
                fe_idx = [0,1,3,8]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        self.fe_idx = fe_idx
        fea_channels =[]
        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')
        i = 0
        self.block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            self.block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name='conv' + str(i + 2)))
            if i in self.fe_idx:
                fea_channels.append(make_divisible(scale * c))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*self.block_list)

        self.fpn_list = []
        in_channels = fea_channels[-1]
        for i in range(len(self.fe_idx) - 1):
            self.fpn_list.append(
                InceptionBlock(
                    in_channels=fea_channels[-1-i] + fea_channels[-2-i],
                    out_channels=fea_channels[-2-i],
                    kernel_size_list=[1,3,5],
                    stride=1,
                    if_act=True,
                    act='relu',
                    name="fpn_"+str(i)+"_1")
            )
            self.fpn_list.append(
                ConvBNLayer(
                    in_channels=fea_channels[-2-i],
                    out_channels=fea_channels[-2-i],
                    kernel_size=3,
                    stride=(pow(2,i+1),1),
                    padding=1,
                    if_act=True,
                    act='relu',
                    name="fpn_"+str(i)+"_2")
            )
            in_channels += fea_channels[-2-i]
        self.fpns = nn.Sequential(*self.fpn_list)
        self.conv2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        fe = []
        for idx, block in enumerate(self.block_list):
            x = block(x)
            if idx in self.fe_idx:
                fe.append(x)

        agg_fe_list = []
        agg_fe_list.append(fe[-1])
        for i in range(len(self.fpn_list)//2):
            agg_fe_list.append(
                self.fpn_list[i*2](
                    paddle.concat(                                                    
                    [F.upsample(agg_fe_list[-1-i], scale_factor=(2,1),mode='nearest',align_mode=1), 
                    fe[-2-i]],                                                             
                    axis=1)
                )
            )
            agg_fe_list.append(
                self.fpn_list[i*2+1](
                    agg_fe_list[-1]
                )
            )

        x = paddle.concat(
            [agg_fe_list[i*2] for i in range(math.ceil(len(agg_fe_list)/2))],axis=1)                           

        x = self.conv2(x)
        x = self.pool(x)
        return x

class InceptionBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size_list,
                 stride,
                 if_act=True,
                 act=None,
                 name=None):
        super(InceptionBlock, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv_list = []
        for kernel_size in kernel_size_list:
            self.conv_list.append(
                ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                if_act=if_act,
                act=act,
                name="{}_{}x{}".format(name,kernel_size,kernel_size))
            )
        self.convs = nn.Sequential(*self.conv_list)

    def forward(self, x):
        y = paddle.add_n([block(x) for block in self.conv_list])
        return y