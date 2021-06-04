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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', 'PaddleSlim')))

from ppocr.data import build_dataloader
from ppocr.data.imaug.rec_img_aug import srn_other_inputs
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import print_dict
import tools.program as program
from model_summary import summary
import paddle
from paddle.jit import to_static
import numpy as np
from paddleslim.dygraph import FPGMFilterPruner

def main():
    global_config = config['Global']
    # Build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    config['Train']['loader']['num_workers'] = 1
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

    # Build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # Build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])
    if config['Global']['distributed']:
        model = paddle.DataParallel(model)
        
    # Build loss
    loss_class = build_loss(config['Loss'])

    # Build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # Build metric
    eval_class = build_metric(config['Metric'])

    # Load pretrain model
    if (not global_config.get('pretrained_model')):
        logger.error(
            "No pretrained_model found.\n"
        )
        return
    checkpoints = global_config.get('checkpoints')
    config['Global']['checkpoints'] = None
    pretrained_model_dict = init_model(config, model, logger)
    config['Global']['checkpoints'] = checkpoints

    # summary
    logger.info("Model before pruning: ")
    use_srn = config['Architecture']['algorithm'] == "SRN"
    if use_srn:
        shape = config['Train']['dataset']['transforms'][2]['SRNRecResizeImg']['image_shape']
        num_heads = config['Architecture']['Head']['num_heads']
        max_text_length = config['Architecture']['Head']['max_text_length']
        others = srn_other_inputs(shape, num_heads, max_text_length)
        input_size = []
        input_size.append((1, shape[0], shape[1], shape[2]))
        input_dtype = ['float32','int64','int64','float32','float32']
        for item in others:
            input_size.append((1,) + item.shape)
        summary(model,input_size,input_dtype,logger,use_srn)
    else:
        shape = config['Train']['dataset']['transforms'][3]['RecResizeImg']['image_shape']
        summary(model, (1, shape[0], shape[1], shape[2]), logger=logger, use_srn=use_srn)

    if len(pretrained_model_dict):
        logger.info('metric in pretrained_model ***************')
        for k, v in pretrained_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # Init pruner
    pruner = FPGMFilterPruner(model.backbone, [1, shape[0], shape[1], shape[2]])

    # Analyse sensitivity
    # TODO: Fix memory leak.
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
        metric = metric['acc']

        return metric
    
    sen_file = global_config.get('sen_file')
    if not sen_file:
        logger.error(
            "No sen_file found.\n"
        )
        return
    pruned_flops = 0.5 if not global_config.get('pruned_flops') else global_config.get('pruned_flops')
    skip_vars = [] if not global_config.get('skip_vars') else global_config.get('skip_vars')
    sen = pruner.sensitive(eval_func=eval_fn, sen_file=sen_file,skip_vars=skip_vars)
    for k, v in sen.items():
        tmp = dict()
        for ratio in v:
            tmp[ratio] = round(v[ratio],3)
        logger.info("{:<30}:{}".format(k,tmp))


    logger.info('pruning {}% FLOPs and skip {} layer'.format(pruned_flops*100, skip_vars))
    plan = pruner.sensitive_prune(pruned_flops, skip_vars=skip_vars)
    
    # Finetune
    if global_config.get('checkpoints'):
        logger.info("Load model in checkpoints: ")
        checkpoints_model_dict = init_model(config, model, logger, optimizer)
        if len(checkpoints_model_dict):
            logger.info('metric in ckpt ***************')
            for k, v in checkpoints_model_dict.items():
                logger.info('{}:{}'.format(k, v))
    else:
        pretrained_pruned_model = global_config.get('pretrained_pruned_model')
        if pretrained_pruned_model:
            logger.info("load pretrained model from {}".format(
                pretrained_pruned_model))
            pre_dict = paddle.load(pretrained_pruned_model + '.pdparams')
            model.set_state_dict(pre_dict)
        checkpoints_model_dict = {}
        
    # summary
    logger.info("Model after pruning: ")
    if use_srn:
        summary(model,input_size,input_dtype,logger,use_srn)
    else:
        summary(model, (1, shape[0], shape[1], shape[2]), logger=logger, use_srn=use_srn)

    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, checkpoints_model_dict, logger, vdl_writer)

    # Eval after finetune
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, use_srn)
    logger.info('metric finetune ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))

if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main()
    