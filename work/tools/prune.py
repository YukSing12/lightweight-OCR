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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', 'PaddleSlim')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', 'PaddleOCR')))

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import print_dict
import tools.program as program

import paddle
import numpy as np
from paddleslim.dygraph import FPGMFilterPruner

def main():
    global_config = config['Global']
    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    if len(valid_dataloader) == 0:
        logger.error(
            "No Images in eval dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])
    use_srn = config['Architecture']['algorithm'] == "SRN"

    if (not global_config.get('checkpoints')) and (not global_config.get('pretrained_model')):
        logger.error(
            "No checkpoints or pretrained_model found.\n"
        )
        return
        
    best_model_dict = init_model(config, model, logger)
    if len(best_model_dict):
        logger.info('metric in ckpt ***************')
        for k, v in best_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # build metric
    eval_class = build_metric(config['Metric'])

    # start eval
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, use_srn)
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))

    # baseline
    baseline = metric['acc']
    logger.info('baseline is {}'.format(baseline))

    # pruner
    shape = config['Train']['dataset']['transforms'][3]['RecResizeImg']['image_shape']
    pruner = FPGMFilterPruner(model, [1, shape[0], shape[1], shape[2]])

    # analyse sensitivity
    def eval_fn():
        # Adaptive-BN
        model.train()
        max_iter = int(train_dataloader.batch_sampler.total_size / 30 / train_dataloader.batch_sampler.batch_size)
        with paddle.no_grad():
            for idx, batch in enumerate(train_dataloader):
                model.forward(batch[0])
                if idx > max_iter:
                    break

        # Eval
        metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, use_srn)
        return metric['acc']
    
    sen = pruner.sensitive(eval_func=eval_fn, baseline=baseline, sen_file="./output/fpgm_sen_bn.pickle")
    

if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
