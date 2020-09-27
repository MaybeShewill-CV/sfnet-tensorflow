#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午4:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV
# @File    : resnet.py
# @IDE: PyCharm
"""
Resnet for image classification
"""
import collections
import time

import tensorflow as tf

from sfnet_model import resnet_utils
from local_utils.config_utils import parse_config_utils


class ResNet(resnet_utils.ResnetBase):
    """
    Dialated Resnet Backbone for semantic segmentation
    """

    def __init__(self, phase, cfg):
        """

        :param phase: phase of training or testing
        """
        super(ResNet, self).__init__(phase=phase)

        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()
        self._resnet_size = self._cfg.MODEL.RESNET.NET_SIZE
        self._block_sizes = self._get_block_sizes()
        self._block_strides = [1, 2, 2, 2]
        self._use_dilation = self._cfg.MODEL.RESNET.USE_DILATION
        self._dilation_rates = [1, 1, 2, 4]
        self._classes_nums = self._cfg.DATASET.NUM_CLASSES
        self._separate_index = 2
        if self._resnet_size < 50:
            self._block_func = self._bottleneck_block_v1
            self._block_fn_with_dilation = self._bottleneck_block_v1_with_dilation
        else:
            self._block_func = self._bottleneck_block_v1
            self._block_fn_with_dilation = self._bottleneck_block_v1_with_dilation

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _get_block_sizes(self):
        """

        :return:
        """
        resnet_name = 'resnet-{:d}'.format(self._resnet_size)
        block_sizes = {
            'resnet-18': [2, 2, 2, 2],
            'resnet-34': [3, 4, 6, 3],
            'resnet-50': [3, 4, 6, 3],
            'resnet-101': [3, 4, 23, 3],
            'resnet-152': [3, 8, 36, 3]
        }
        try:
            return block_sizes[resnet_name]
        except KeyError:
            raise RuntimeError('Wrong resnet name, only '
                               '[resnet-18, resnet-34, resnet-50, '
                               'resnet-101, resnet-152] supported')

    def _process_image_input_tensor(self, input_image_tensor, kernel_size,
                                    conv_stride, output_dims, pool_size,
                                    pool_stride):
        """
        Resnet entry
        :param input_image_tensor: input tensor [batch_size, h, w, c]
        :param kernel_size: kernel size
        :param conv_stride: stride of conv op
        :param output_dims: output dims of conv op
        :param pool_size: pooling window size
        :param pool_stride: pooling window stride
        :return:
        """
        inputs = self._conv2d_fixed_padding(
            inputs=input_image_tensor, kernel_size=kernel_size,
            strides=conv_stride, output_dims=output_dims, name='initial_conv_pad')
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = self.maxpooling(inputdata=inputs, kernel_size=pool_size,
                                 stride=pool_stride, padding='SAME',
                                 name='initial_max_pool')

        return inputs

    def _resnet_block_layer(self, input_tensor, stride, block_nums, output_dims, name,
                            use_dilation=False, dilation_rate=2):
        """
        resnet single block.Details can be found in origin paper table 1
        :param input_tensor: input tensor [batch_size, h, w, c]
        :param stride: the conv stride in bottleneck conv_2 op
        :param block_nums: block repeat nums
        :param name: layer name
        :param use_dilation:
        :param dilation_rate:
        :return:
        """
        def projection_shortcut(_inputs):
            """
            shortcut projection to align the feature maps
            :param _inputs:
            :return:
            """
            return self._conv2d_fixed_padding(
                inputs=_inputs, output_dims=output_dims * 4, kernel_size=1,
                strides=stride, name='projection_shortcut')

        vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            inputs = self._block_func(
                input_tensor=input_tensor,
                output_dims=output_dims,
                projection_shortcut=projection_shortcut,
                stride=stride,
                name='init_block_fn'
            )
            if not use_dilation:
                for index in range(1, block_nums):
                    inputs = self._block_func(
                        input_tensor=inputs,
                        output_dims=output_dims,
                        projection_shortcut=None,
                        stride=1,
                        name='block_fn_{:d}'.format(index)
                    )
            else:
                for index in range(1, block_nums):
                    inputs = self._block_fn_with_dilation(
                        input_tensor=inputs,
                        output_dims=output_dims,
                        projection_shortcut=None,
                        stride=1,
                        name='block_fn_{:d}'.format(index),
                        dilation_rate=dilation_rate
                    )

        return inputs

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        intermedia_result = collections.OrderedDict()
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first layer process
            inputs = self._process_image_input_tensor(
                input_image_tensor=input_tensor,
                kernel_size=7,
                conv_stride=2,
                output_dims=64,
                pool_size=3,
                pool_stride=2
            )

            # The first two layers doesn't not need apply dilation
            for index, block_nums in enumerate(self._block_sizes):
                output_dims = 32 * (2 ** index)
                if self._use_dilation:
                    if index < self._separate_index:
                        use_dilation = False
                    else:
                        use_dilation = True
                else:
                    use_dilation = False

                dilation_rate = self._dilation_rates[index]
                inputs = self._resnet_block_layer(
                    input_tensor=inputs,
                    stride=self._block_strides[index],
                    block_nums=block_nums,
                    output_dims=output_dims,
                    name='residual_block_{:d}'.format(index + 1),
                    use_dilation=use_dilation,
                    dilation_rate=dilation_rate
                )
                intermedia_result['stage_{:d}'.format(index + 1)] = inputs

        return intermedia_result


def main():
    """test code
    """
    image_tensor = tf.random.uniform([1, 720, 720, 3], name='input_tensor')
    net = ResNet(phase='train', cfg=parse_config_utils.RESNET_FCN_CITYSCAPES_CFG)

    result = net.inference(
        input_tensor=image_tensor,
        name='test',
        reuse=False
    )
    for stage_name, stage_output in result.items():
        print('Stage name: {:s}, output: {}'.format(stage_name, stage_output))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loop_times = 500

        t_start = time.time()
        for _ in range(loop_times):
            preds = sess.run(result)
        t_cost = (time.time() - t_start) / loop_times
        print('Net inference cost time: {:.5f}'.format(t_cost))
        print('Net inference can reach: {:.5f} fps'.format(1.0 / t_cost))


if __name__ == '__main__':
    main()
