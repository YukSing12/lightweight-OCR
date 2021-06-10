# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

__all__ = ['PlugNet']


class PlugNet(nn.Layer):
    def __init__(self, config):
        """
        PlugNet module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(PlugNet, self).__init__()

        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']

        # build backbone, backbone is need for del, rec and cls
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"], model_type)
        in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        # pluggable super-resolution unit (PSU)
        if 'PSU' not in config or config['PSU'] is None:
            self.use_psu = False
        else:
            self.use_psu = True
            config['PSU']['in_channels'] = self.backbone.mid_fea_channels
            self.psu = build_neck(config['PSU'])

        # # build head, head is need for det, rec and cls
        config["Head"]['in_channels'] = in_channels
        self.head = build_head(config["Head"])

    def forward(self, x, data=None):
        x,y = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)

        x = self.head(x)

        if self.use_psu:
            y = self.psu(y)
            return x, y
        else:
            return x
