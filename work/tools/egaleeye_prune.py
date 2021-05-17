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
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import print_dict
import tools.program as program

import paddle
from paddle.jit import to_static
import numpy as np
from paddleslim.dygraph import FPGMFilterPruner
from paddleslim.analysis import dygraph_flops as flops
from paddleslim.analysis import model_size
from tqdm import tqdm

def get_size(file_path):
    """ Get size of file or directory.

    Args:
        file_path(str): Path of file or directory.

    Returns: 
        size(int): Size of file or directory in bits.
    """
    size = 0
    if os.path.isdir(file_path):
        for root, dirs, files in os.walk(file_path):
            for f in files:
                size += os.path.getsize(os.path.join(root, f))
    elif os.path.isfile(file_path):
        size = (os.path.getsize(file_path))
    return size
    
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
    # build metric
    eval_class = build_metric(config['Metric'])
    
    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])
    shape = config['Train']['dataset']['transforms'][3]['RecResizeImg']['image_shape']
    use_srn = config['Architecture']['algorithm'] == "SRN"

    if (not global_config.get('checkpoints')) and (not global_config.get('pretrained_model')):
        logger.error(
            "No checkpoints or pretrained_model found.\n"
        )
        return
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
    checkpoints = global_config.get('checkpoints')
    config['Global']['checkpoints'] = None
    best_model_dict = init_model(config, model, logger)
    config['Global']['checkpoints'] = checkpoints

    logger.info("Model before pruning: ")
    summary_dict = paddle.summary(model, (1, shape[0], shape[1], shape[2]))
    baseline_metric = {}
    baseline_metric['acc'] = best_model_dict.get('acc')          
    baseline_metric['flops'] = flops(model, [1, shape[0], shape[1], shape[2]])
    baseline_metric['params'] = summary_dict['total_params']

    logger.info('baseline metric:')
    for k, v in baseline_metric.items():
        logger.info('{}:{}'.format(k, v))

    # pruner
    pruner = FPGMFilterPruner(model, [1, shape[0], shape[1], shape[2]])
    FILTER_DIM = [0]

    # condition
    condition = "flops"
    assert condition in ["acc", "flops", "params"]
    target = 0.2

    while(True):
        # random pruning startegy
        max_rate = 0.95
        min_rate = 0
        ratios = {}
        skip_vars = ['conv_last_weights']
        # skip_vars = ['res5b_branch2b_weights','res5b_branch2a_weights','res5a_branch2b_weights','res5a_branch1_weights']
        # skip_vars = ['res5a_branch2b_weights','res5a_branch1_weights']
        pruner.skip_vars = skip_vars
        for group in pruner.var_group.groups:
            var_name = group[0][0]
            if var_name in skip_vars:
                continue
            ratios[var_name] = float(np.random.rand(1) * max_rate + min_rate)

        plan = pruner.prune_vars(ratios, FILTER_DIM)
        logger.info("Model after pruning: ")
        summary_dict = paddle.summary(model, (1, shape[0], shape[1], shape[2]))

        # Adaptive-BN
        model.train()
        max_iter = int(train_dataloader.batch_sampler.total_size / 30 / train_dataloader.batch_sampler.batch_size)
        with paddle.no_grad():
            pbar = tqdm(total=max_iter, desc='adaptiveBN model:')
            for idx, batch in enumerate(train_dataloader):
                if idx > max_iter:
                    break
                model.forward(batch[0])
                pbar.update(1)
        pbar.close()
        
        # Eval
        eval_metric = program.eval(model, valid_dataloader, post_process_class,
                            eval_class, use_srn)

        
        pruned_metric = {}
        pruned_metric['acc'] = eval_metric['acc']
        pruned_metric['flops'] = flops(model, [1, shape[0], shape[1], shape[2]])
        pruned_metric['params'] = summary_dict['total_params']
        logger.info('pruned metric:')
        for k, v in pruned_metric.items():
            logger.info('{}:{}'.format(k, v))

        ratio = (baseline_metric[condition] - pruned_metric[condition]) / baseline_metric[condition]
        logger.info('ratio:{}'.format(ratio))
        if ratio > target: # For acc
            logger.info('Save model')
            break
        else:
            logger.info('Restore model')
            plan.restore(model)

    # Finetune
    if global_config.get('checkpoints'):
        checkpoints_model_dict = init_model(config, model, logger, optimizer)
        if len(checkpoints_model_dict):
            logger.info('metric in ckpt ***************')
            for k, v in checkpoints_model_dict.items():
                logger.info('{}:{}'.format(k, v))
    else:
        checkpoints_model_dict = {}

    logger.info("Model after pruning: ")
    summary_dict = paddle.summary(model, (1, shape[0], shape[1], shape[2]))

    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, checkpoints_model_dict, logger, vdl_writer)

    # Save
    model.eval()
    save_path = '{}/inference'.format(config['Global']['save_inference_dir'])

    infer_shape = [3, 32, -1]  # for rec model, H must be 32
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype='float32')
        ])
    paddle.jit.save(model, save_path)
    logger.info('inference model is saved to {}'.format(save_path))

    # Calculate model size
    model_size = get_size(os.path.join(save_path + '.pdiparams')) + get_size(os.path.join(save_path + '.pdmodel'))
    logger.info('pruned model size is {}'.format(model_size))    
        

if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
