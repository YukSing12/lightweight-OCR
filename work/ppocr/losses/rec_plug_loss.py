# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from paddle import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer
class PlugLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(PlugLoss, self).__init__()
        self.rec_loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.sr_loss_func = nn.L1Loss(reduction='mean')
        self.lambda_factor = 0.01 # balance the weight in two different tasks and keep the recognition performance 
        self.alpha = 0.5 # probability parameter alpha

        # build gaussian_blur by convolution
        gaussian_blur_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        gaussian_blur_filter = np.repeat(gaussian_blur_filter.reshape([1, 1, 3, 3]), 3, axis=1)
        self.gaussian_blur = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3],padding='SAME',
                  weight_attr=ParamAttr(
                      name='gaussian_blur',
                      initializer=NumpyArrayInitializer(value=gaussian_blur_filter),
                      trainable=False
                  ))

    def __call__(self, predicts, batch, frn_blur_I):
        frn_I = predicts[1]
        predicts = predicts[0]

        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        rec_loss = self.rec_loss_func(predicts, labels, preds_lengths, label_lengths).mean()

        sr_loss = self.sr_loss_func(frn_I, frn_blur_I)

        loss = rec_loss + self.lambda_factor * sr_loss  # sum
        return {'loss': loss, 'rec_loss': rec_loss, 'sr_loss': sr_loss}

    def get_blur(self,I,p1,p2):
        if p1 >= self.alpha and p2 >= self.alpha:
            return F.upsample(F.upsample(self.gaussian_blur(I),scale_factor=1/4,mode='nearest'),scale_factor=4,mode='nearest')
        elif p1 >= self.alpha and p2 < self.alpha:
            return self.gaussian_blur(I)
        elif p1 < self.alpha and p2 >= self.alpha:
            return F.upsample(F.upsample(I,scale_factor=1/4,mode='nearest'),scale_factor=4,mode='nearest')
        else:
            return I
