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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', 'work')))

import yaml
import paddle
import paddle.distributed as dist

paddle.seed(2)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.modeling.architectures.plugnet import PlugNet
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program
from model_summary import summary

dist.get_world_size()


def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

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

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    if 'PSU' not in config['Architecture'] or config['Architecture']['PSU'] is None:
        model = build_model(config['Architecture'])
    else:
        model = PlugNet(config['Architecture'])
    if config['Global']['distributed']:
        model = paddle.DataParallel(model)

    # summary
    use_srn = config['Architecture']['algorithm'] == "SRN"
    if use_srn:
        from ppocr.data.imaug.rec_img_aug import srn_other_inputs
        for opp in config['Train']['dataset']['transforms']:
            if 'SRNRecResizeImg' in opp:
                shape = opp['SRNRecResizeImg']['image_shape']
                break
            else:
                shape = None
        num_heads = config['Architecture']['Head']['num_heads']
        max_text_length = config['Architecture']['Head']['max_text_length']
        others = srn_other_inputs(shape, num_heads, max_text_length)
        input_size = []
        input_size.append((1, shape[0], shape[1], shape[2]))
        input_dtype = ['float32','int64','int64''float32','float32']
        for item in others:
            input_size.append((1,) + item.shape)
        summary(model,input_size,input_dtype,logger)
    else:
        for opp in config['Train']['dataset']['transforms']:
            if 'RecResizeImg' in opp:
                shape = opp['RecResizeImg']['image_shape']
                break
            else:
                shape = None
        summary(model, (1, shape[0], shape[1], shape[2]),logger=logger)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)


def test_reader(config, device, logger):
    loader = build_dataloader(config, 'Train', device, logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
    # test_reader(config, device, logger)
