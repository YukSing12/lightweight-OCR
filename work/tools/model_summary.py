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

import warnings
import numpy as np
import numbers

import paddle
import paddle.nn as nn
from paddle.static import InputSpec

from collections import OrderedDict

__all__ = ['summary']


def summary(net, input_size, dtypes=None, logger=None):
    """Prints a string summary of the network.

    Args:
        net (Layer): the network which must be a subinstance of Layer.
        input_size (tuple|InputSpec|list[tuple|InputSpec]): size of input tensor. if model only 
                    have one input, input_size can be tuple or InputSpec. if model
                    have multiple input, input_size must be a list which contain 
                    every input's shape. Note that input_size only dim of
                    batch_size can be None or -1.
        dtypes (str, optional): if dtypes is None, 'float32' will be used, Default: None.

    Returns:
        Dict: a summary of the network including total params and total trainable params.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            class LeNet(nn.Layer):
                def __init__(self, num_classes=10):
                    super(LeNet, self).__init__()
                    self.num_classes = num_classes
                    self.features = nn.Sequential(
                        nn.Conv2D(
                            1, 6, 3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2),
                        nn.Conv2D(
                            6, 16, 5, stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2))

                    if num_classes > 0:
                        self.fc = nn.Sequential(
                            nn.Linear(400, 120),
                            nn.Linear(120, 84),
                            nn.Linear(
                                84, 10))

                def forward(self, inputs):
                    x = self.features(inputs)

                    if self.num_classes > 0:
                        x = paddle.flatten(x, 1)
                        x = self.fc(x)
                    return x

            lenet = LeNet()

            params_info = paddle.summary(lenet, (1, 1, 28, 28))
            print(params_info)

    """
    if isinstance(input_size, InputSpec):
        _input_size = tuple(input_size.shape)
    elif isinstance(input_size, list):
        _input_size = []
        for item in input_size:
            if isinstance(item, int):
                item = (item, )
            assert isinstance(item,
                              (tuple, InputSpec)), 'When input_size is list, \
            expect item in input_size is a tuple or InputSpec, but got {}'.format(
                                  type(item))

            if isinstance(item, InputSpec):
                _input_size.append(tuple(item.shape))
            else:
                _input_size.append(item)
    elif isinstance(input_size, int):
        _input_size = (input_size, )
    else:
        _input_size = input_size

    if not paddle.in_dynamic_mode():
        warnings.warn(
            "Your model was created in static mode, this may not get correct summary information!"
        )
        in_train_mode = False
    else:
        in_train_mode = net.training

    if in_train_mode:
        net.eval()

    def _is_shape(shape):
        for item in shape:
            if isinstance(item, (list, tuple)):
                return False
        return True

    def _check_shape(shape):
        num_unknown = 0
        new_shape = []
        for i in range(len(shape)):
            item = shape[i]
            if item is None or item == -1:
                num_unknown += 1
                if num_unknown > 1:
                    raise ValueError(
                        'Option input_size only the dim of batch_size can be None or -1.'
                    )
                item = 1
            elif isinstance(item, numbers.Number):
                if item <= 0:
                    raise ValueError(
                        "Expected element in input size greater than zero, but got {}".
                        format(item))
            new_shape.append(item)
        return tuple(new_shape)

    def _check_input(input_size):
        if isinstance(input_size, (list, tuple)) and _is_shape(input_size):
            return _check_shape(input_size)
        else:
            return [_check_input(i) for i in input_size]

    _input_size = _check_input(_input_size)
    result, params_info = summary_string(net, _input_size, dtypes)
    if logger:
        logger.info(result)
    else:
        print(result)

    if in_train_mode:
        net.train()

    return params_info


@paddle.no_grad()
def summary_string(model, input_size, dtypes=None):
    def _all_is_numper(items):
        for item in items:
            if not isinstance(item, numbers.Number):
                return False
        return True

    def _build_dtypes(input_size, dtype):
        if dtype is None:
            dtype = 'float32'

        if isinstance(input_size, (list, tuple)) and _all_is_numper(input_size):
            return [dtype]
        else:
            return [_build_dtypes(i, dtype) for i in input_size]

    if not isinstance(dtypes, (list, tuple)):
        dtypes = _build_dtypes(input_size, dtypes)

    batch_size = 1

    summary_str = ''

    depth = len(list(model.sublayers()))

    def _get_shape_from_tensor(x):
        if isinstance(x, (paddle.fluid.Variable, paddle.fluid.core.VarBase)):
            return list(x.shape)
        elif isinstance(x, (list, tuple)):
            return [_get_shape_from_tensor(xx) for xx in x]

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = [_get_output_shape(o) for o in output]
        else:
            output_shape = list(output.shape)
        return output_shape

    def register_hook(layer):
        def hook(layer, input, output):
            class_name = str(layer.__class__).split(".")[-1].split("'")[0]

            try:
                layer_idx = int(layer._full_name.split('_')[-1])
            except:
                layer_idx = len(summary)

            m_key = "%s-%i" % (class_name, layer_idx + 1)
            summary[m_key] = OrderedDict()

            try:
                summary[m_key]["input_shape"] = _get_shape_from_tensor(input)
            except:
                warnings.warn('Get layer {} input shape failed!')
                summary[m_key]["input_shape"] = []

            try:
                summary[m_key]["output_shape"] = _get_output_shape(output)
            except:
                warnings.warn('Get layer {} output shape failed!')
                summary[m_key]["output_shape"] = []

            params = 0

            if paddle.in_dynamic_mode():
                layer_state_dict = layer._parameters
            else:
                layer_state_dict = layer.state_dict()

            for k, v in layer_state_dict.items():
                params += np.prod(v.shape)

                try:
                    if (getattr(getattr(layer, k), 'trainable')) and (
                            not getattr(getattr(layer, k), 'stop_gradient')):
                        summary[m_key]["trainable"] = True
                    else:
                        summary[m_key]["trainable"] = False
                except:
                    summary[m_key]["trainable"] = True

            summary[m_key]["nb_params"] = params

        if (not isinstance(layer, nn.Sequential) and
                not isinstance(layer, nn.LayerList) and
            (not (layer == model) or depth < 1)):

            hooks.append(layer.register_forward_post_hook(hook))
        # For rnn, gru and lstm layer
        elif hasattr(layer, 'could_use_cudnn') and layer.could_use_cudnn:
            hooks.append(layer.register_forward_post_hook(hook))

    if isinstance(input_size, tuple):
        input_size = [input_size]

    def build_input(input_size, dtypes):
        if isinstance(input_size, (list, tuple)) and _all_is_numper(input_size):
            if isinstance(dtypes, (list, tuple)):
                dtype = dtypes[0]
            else:
                dtype = dtypes
            return paddle.cast(paddle.rand(list(input_size)), dtype)
        else:
            return [
                build_input(i, dtype) for i, dtype in zip(input_size, dtypes)
            ]

    x = build_input(input_size, dtypes)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x[0],x[-4:])

    # remove these hooks
    for h in hooks:
        h.remove()

    def _get_str_length(summary):
        head_length = {
            'layer_width': 15,
            'input_shape_width': 20,
            'output_shape_width': 20,
            'params_width': 15,
            'table_width': 75
        }

        for layer in summary:
            if head_length['output_shape_width'] < len(
                    str(summary[layer]["output_shape"])):
                head_length['output_shape_width'] = len(
                    str(summary[layer]["output_shape"]))
            if head_length['input_shape_width'] < len(
                    str(summary[layer]["input_shape"])):
                head_length['input_shape_width'] = len(
                    str(summary[layer]["input_shape"]))
            if head_length['layer_width'] < len(str(layer)):
                head_length['layer_width'] = len(str(layer))
            if head_length['params_width'] < len(
                    str(summary[layer]["nb_params"])):
                head_length['params_width'] = len(
                    str(summary[layer]["nb_params"]))

        _temp_width = 0
        for k, v in head_length.items():
            if k != 'table_width':
                _temp_width += v

        if head_length['table_width'] < _temp_width + 5:
            head_length['table_width'] = _temp_width + 5

        return head_length

    table_width = _get_str_length(summary)

    summary_str += "-" * table_width['table_width'] + "\n"
    line_new = "{:^{}} {:^{}} {:^{}} {:^{}}".format(
        "Layer (type)", table_width['layer_width'], "Input Shape",
        table_width['input_shape_width'], "Output Shape",
        table_width['output_shape_width'], "Param #",
        table_width['params_width'])
    summary_str += line_new + "\n"
    summary_str += "=" * table_width['table_width'] + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    max_length = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:^{}} {:^{}} {:^{}} {:^{}}".format(
            layer, table_width['layer_width'],
            str(summary[layer]["input_shape"]),
            table_width['input_shape_width'],
            str(summary[layer]["output_shape"]),
            table_width['output_shape_width'],
            "{0:,}".format(summary[layer]["nb_params"]),
            table_width['params_width'])
        total_params += summary[layer]["nb_params"]

        try:
            total_output += np.prod(summary[layer]["output_shape"])
        except:
            for output_shape in summary[layer]["output_shape"]:
                total_output += np.prod(output_shape)

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    def _get_input_size(input_size, size):
        if isinstance(input_size, (list, tuple)) and _all_is_numper(input_size):
            size = abs(np.prod(input_size) * 4. / (1024**2.))
        else:
            size = sum([_get_input_size(i, size) for i in input_size])
        return size

    total_input_size = _get_input_size(input_size, 0)

    total_output_size = abs(2. * total_output * 4. /
                            (1024**2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024**2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * table_width['table_width'] + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "-" * table_width['table_width'] + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * table_width['table_width'] + "\n"

    # return summary
    return summary_str, {
        'total_params': total_params,
        'trainable_params': trainable_params
    }