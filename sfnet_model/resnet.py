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

CFG = parse_config_utils.CITYSCAPES_CFG


class ResNet(resnet_utils.ResnetBase):
    """
    Dialated Resnet Backbone for semantic segmentation
    """

    def __init__(self, phase, cfg=CFG):
        """

        :param phase: phase of training or testing
        """
        super(ResNet, self).__init__(phase=phase)
        if phase.lower() == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()
        self._resnet_size = cfg.MODEL.RESNET.NET_SIZE
        self._block_sizes = self._get_block_sizes()
        self._block_strides = [1, 2, 2, 2]
        self._classes_nums = cfg.DATASET.NUM_CLASSES
        if self._resnet_size < 50:
            self._block_func = self._building_block_v2
        else:
            self._block_func = self._bottleneck_block_v2

    def _init_phase(self):
        """
        init tensorflow bool flag
        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

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

    def _resnet_block_layer(self, input_tensor, stride, block_nums, output_dims, name):
        """
        resnet single block.Details can be found in origin paper table 1
        :param input_tensor: input tensor [batch_size, h, w, c]
        :param stride: the conv stride in bottleneck conv_2 op
        :param block_nums: block repeat nums
        :param name: layer name
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

        with tf.variable_scope(name):
            inputs = self._block_func(
                input_tensor=input_tensor,
                output_dims=output_dims,
                projection_shortcut=projection_shortcut,
                stride=stride,
                name='init_block_fn'
            )
            for index in range(1, block_nums):
                inputs = self._block_func(
                    input_tensor=inputs,
                    output_dims=output_dims,
                    projection_shortcut=None,
                    stride=1,
                    name='block_fn_{:d}'.format(index)
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
        if tf.__version__ == '1.15.0':
            with tf.compat.v1.variable_scope(name_or_scope=name, reuse=reuse):
                # first layer process
                inputs = self._process_image_input_tensor(
                    input_image_tensor=input_tensor,
                    kernel_size=7,
                    conv_stride=2,
                    output_dims=32,
                    pool_size=3,
                    pool_stride=2
                )

                # The first two layers doesn't not need apply dilation
                for index, block_nums in enumerate(self._block_sizes):
                    output_dims = 32 * (2 ** index)
                    inputs = self._resnet_block_layer(
                        input_tensor=inputs,
                        stride=self._block_strides[index],
                        block_nums=block_nums,
                        output_dims=output_dims,
                        name='residual_block_{:d}'.format(index + 1)
                    )
                    intermedia_result['stage_{:d}'.format(index + 1)] = inputs
        else:
            with tf.variable_scope(name_or_scope=name, reuse=reuse):
                # first layer process
                inputs = self._process_image_input_tensor(
                    input_image_tensor=input_tensor,
                    kernel_size=7,
                    conv_stride=2,
                    output_dims=32,
                    pool_size=3,
                    pool_stride=2
                )

                # The first two layers doesn't not need apply dilation
                for index, block_nums in enumerate(self._block_sizes):
                    output_dims = 32 * (2 ** index)
                    inputs = self._resnet_block_layer(
                        input_tensor=inputs,
                        stride=self._block_strides[index],
                        block_nums=block_nums,
                        output_dims=output_dims,
                        name='residual_block_{:d}'.format(index + 1)
                    )
                    intermedia_result['stage_{:d}'.format(index + 1)] = inputs

        return intermedia_result


def main():
    """test code
    """
    if tf.__version__ == '1.15.0':
        image_tensor = tf.compat.v1.random.uniform(
            [1, 720, 720, 3], name='input_tensor')
    else:
        image_tensor = tf.random.uniform([1, 720, 720, 3], name='input_tensor')

    net = ResNet(phase='train', cfg=CFG)

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
