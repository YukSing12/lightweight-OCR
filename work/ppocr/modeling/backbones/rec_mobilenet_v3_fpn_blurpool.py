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
import paddle.nn.functional as F
# from det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible
# from blurpool import BlurPool
from ppocr.modeling.backbones.blurpool import BlurPool
from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible

__all__ = ['MobileNetV3_FPN_BP']


class MobileNetV3_FPN_BP(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(MobileNetV3_FPN_BP, self).__init__()
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
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
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
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

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
        self.fe_idx = []
        fea_channels =[]
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
            if s != 1:
                self.fe_idx.append(i)
                fea_channels.append(make_divisible(scale * c))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*self.block_list)

        self.fpn1 = ConvBNLayer(
            in_channels=fea_channels[-1] + fea_channels[-2],
            out_channels=fea_channels[-2],
            kernel_size=3,
            stride=(2,1),
            padding=1,
            if_act=True,
            act='relu',
            name="fpn1")
        self.fpn2 = ConvBNLayer(
            in_channels=fea_channels[-2] + fea_channels[-3],
            out_channels=fea_channels[-3],
            kernel_size=3,
            stride=(4,1),
            padding=1,
            if_act=True,
            act='relu',
            name="fpn2")

        self.conv2 = ConvBNLayer(
            in_channels=fea_channels[-1] + fea_channels[-2] + fea_channels[-3],
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')

        self.out_channels = make_divisible(scale * cls_ch_squeeze)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=1, padding=0)
        self.blurpool = BlurPool(self.out_channels, stride=2, pad_type='replicate')
        

    def forward(self, x):
        x = self.conv1(x)
        fe = []
        for idx, block in enumerate(self.block_list):
            x = block(x)
            if idx in self.fe_idx:
                fe.append(x)

        agg_fe1 = (paddle.concat(                                                    # 136,4,160
                [F.upsample(fe[-1], scale_factor=(2,1),mode='nearest',align_mode=1), # 96,4,160
                fe[-2]],                                                             # 40,4,160
                axis=1)) 
        agg_fe2 = (paddle.concat(                                                    # 64,8,160
                [F.upsample(fe[-2], scale_factor=(2,1),mode='nearest',align_mode=1), # 40,8,160
                fe[-3]],                                                             # 24,8,160
                axis=1))
        agg_fe1 = self.fpn1(agg_fe1)                                                 # 136,4,160 -> 40,2,160
        agg_fe2 = self.fpn2(agg_fe2)                                                 # 64 ,8,160 -> 24,2,160
        x = paddle.concat([fe[-1],agg_fe1,agg_fe2],axis=1)                           # 160,2,160

        x = self.conv2(x)
        x = self.pool(x)
        x = self.blurpool(x)
        return x

if __name__ == "__main__":

    model = MobileNetV3_FPN_BP(model_name='large', scale=0.5)
    import paddle
    paddle.summary(model, (1, 3, 32, 320))

    out = model(paddle.randn((1, 3, 32, 160)))

    print(out.shape)