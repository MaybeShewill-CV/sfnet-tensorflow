#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午3:59
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : cnn_basenet.py
# @IDE: PyCharm Community Edition
"""
The base convolution neural networks mainly implement some useful cnn functions
"""
import math

import tensorflow as tf
import numpy as np

TF_VERSION = tf.__version__


class CNNBaseModel(object):
    """
    Base model for other specific cnn models
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]
                                ] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + \
                    [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                if TF_VERSION == '1.15.0':
                    w_init = tf.variance_scaling_initializer()
                else:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
                # w_init = initializers.xavier_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            if TF_VERSION == '1.15.0':
                w = tf.compat.v1.get_variable(
                    'W', filter_shape, initializer=w_init)
            else:
                w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides,
                                    padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def separate_conv(input_tensor, output_channels, kernel_size, name, depth_multiplier=1,
                      padding='SAME', stride=1):
        """

        :param input_tensor:
        :param output_channels:
        :param kernel_size:
        :param name:
        :param depth_multiplier:
        :param padding:
        :param stride:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            padding = padding.upper()

            depthwise_filter_shape = [
                kernel_size, kernel_size] + [in_channel, depth_multiplier]
            pointwise_filter_shape = [
                1, 1, in_channel * depth_multiplier, output_channels]
            if TF_VERSION == '1.15.0':
                w_init = tf.variance_scaling_initializer()
            else:
                w_init = tf.contrib.layers.variance_scaling_initializer()

            depthwise_filter = tf.get_variable(
                name='depthwise_filter_w', shape=depthwise_filter_shape,
                initializer=w_init
            )
            pointwise_filter = tf.get_variable(
                name='pointwise_filter_w', shape=pointwise_filter_shape,
                initializer=w_init
            )

            result = tf.nn.separable_conv2d(
                input=input_tensor,
                depthwise_filter=depthwise_filter,
                pointwise_filter=pointwise_filter,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='separate_conv_output'
            )

        return result

    @staticmethod
    def depthwise_conv(input_tensor, kernel_size, name, depth_multiplier=1,
                       padding='SAME', stride=1):
        """

        :param input_tensor:
        :param kernel_size:
        :param name:
        :param depth_multiplier:
        :param padding:
        :param stride:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            padding = padding.upper()

            depthwise_filter_shape = [
                kernel_size, kernel_size] + [in_channel, depth_multiplier]
            if TF_VERSION == '1.15.0':
                w_init = tf.variance_scaling_initializer()
            else:
                w_init = tf.contrib.layers.variance_scaling_initializer()

            depthwise_filter = tf.get_variable(
                name='depthwise_filter_w', shape=depthwise_filter_shape,
                initializer=w_init
            )

            result = tf.nn.depthwise_conv2d(
                input=input_tensor,
                filter=depthwise_filter,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='depthwise_conv_output'
            )
        return result

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        if TF_VERSION == '1.15.0':
            return tf.nn.max_pool2d(
                input=inputdata, ksize=kernel, strides=strides, padding=padding,
                data_format=data_format, name=name
            )
        else:
            return tf.nn.max_pool(
                value=inputdata, ksize=kernel, strides=strides, padding=padding,
                data_format=data_format, name=name
            )

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [
            1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', keepdims=False, name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :param keepdims:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name, keepdims=keepdims)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(
            range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable(
                'beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable(
                'gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError(
                "Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable(
            'beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable(
            'gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, is_trainning, noise_shape=None, name=None):
        """

        :param name:
        :param inputdata:
        :param keep_prob:
        :param is_trainning:
        :param noise_shape:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            output_tensor = tf.cond(
                pred=is_trainning,
                true_fn=lambda: tf.nn.dropout(
                    inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name),
                false_fn=lambda: inputdata,
                name='dropout_output'
            )
        return output_tensor

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack(
                [tf.shape(inputdata)[0], -1]))

        if w_init is None:
            if TF_VERSION == '1.15.0':
                w_init = tf.variance_scaling_initializer()
            else:
                w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: tf.identity(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training, name, scale=True):
        """

        :param inputdata:
        :param is_training:
        :param name:
        :param scale:
        :return:
        """
        return tf.layers.batch_normalization(
            inputs=inputdata,
            training=is_training,
            scale=scale,
            name=name
        )

    @staticmethod
    def layerfrn(input_tensor, name, eps=1e-6, learn_eps=True, scale=True):
        """

        :param input_tensor:
        :param name:
        :param eps:
        :param learn_eps:
        :param scale:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            input_channels = input_tensor.get_shape().as_list()[-1]

            # compute norm
            norm_square = tf.pow(input_tensor, 2, name='power')
            norm_square = tf.reduce_mean(
                input_tensor=norm_square, axis=[1, 2], keepdims=True)
            if scale:
                gamma = tf.get_variable(
                    name='gamma',
                    shape=[1, 1, 1, input_channels],
                    dtype=tf.float32,
                    initializer=tf.ones_initializer(),
                    trainable=True
                )
            else:
                gamma = tf.get_variable(
                    name='gamma',
                    shape=[1, 1, 1, input_channels],
                    dtype=tf.float32,
                    initializer=tf.ones_initializer(),
                    trainable=False
                )
            beta = tf.get_variable(
                name='beta',
                shape=[1, 1, 1, input_channels],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=True
            )

            # apply frn
            if learn_eps:
                eps_ = tf.get_variable(
                    name='eps',
                    shape=[1, 1, 1, input_channels],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(eps),
                    trainable=True
                )
            else:
                eps_ = tf.get_variable(
                    name='eps',
                    shape=[1, 1, 1, input_channels],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(eps),
                    trainable=False
                )
            frn = input_tensor * tf.rsqrt(norm_square + tf.abs(eps_))
            frn = gamma * frn + beta

            # apply tlu
            t_thresh = tf.get_variable(
                name='t_thresh',
                shape=[1, 1, 1, input_channels],
                dtype=tf.float32,
                initializer=tf.constant_initializer(eps),
                trainable=True
            )
            frn_output = tf.maximum(frn, t_thresh, 'frn_output')
        return frn_output

    @staticmethod
    def layergn(inputdata, name, group_size=32, esp=1e-5):
        """

        :param inputdata:
        :param name:
        :param group_size:
        :param esp:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            inputdata = tf.transpose(inputdata, [0, 3, 1, 2])
            n, c, h, w = inputdata.get_shape().as_list()
            group_size = min(group_size, c)
            inputdata = tf.reshape(
                inputdata, [-1, group_size, c // group_size, h, w])
            mean, var = tf.nn.moments(inputdata, [2, 3, 4], keep_dims=True)
            inputdata = (inputdata - mean) / tf.sqrt(var + esp)

            # 每个通道的gamma和beta
            gamma = tf.Variable(tf.constant(
                1.0, shape=[c]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(
                0.0, shape=[c]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, c, 1, 1])
            beta = tf.reshape(beta, [1, c, 1, 1])

            # 根据论文进行转换 [n, c, h, w, c] 到 [n, h, w, c]
            output = tf.reshape(inputdata, [-1, c, h, w])
            output = output * gamma + beta
            output = tf.transpose(output, [0, 2, 3, 1])

        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param activation: whether to apply a activation func to deconv result
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                if TF_VERSION == '1.15.0':
                    w_init = tf.variance_scaling_initializer()
                else:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_tensor, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                if TF_VERSION == '1.15.0':
                    w_init = tf.variance_scaling_initializer()
                else:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret

    @staticmethod
    def spatial_dropout(input_tensor, keep_prob, is_training, name, seed=1234):
        """
        空间dropout实现
        :param input_tensor:
        :param keep_prob:
        :param is_training:
        :param name:
        :param seed:
        :return:
        """

        def f1():
            input_shape = input_tensor.get_shape().as_list()
            noise_shape = tf.constant(
                value=[input_shape[0], 1, 1, input_shape[3]])
            return tf.nn.dropout(input_tensor, keep_prob, noise_shape, seed=seed, name="spatial_dropout")

        def f2():
            return input_tensor

        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            output = tf.cond(is_training, f1, f2)
            return output

    @staticmethod
    def lrelu(inputdata, name, alpha=0.2):
        """

        :param inputdata:
        :param alpha:
        :param name:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            return tf.nn.relu(inputdata) - alpha * tf.nn.relu(-inputdata)

    @staticmethod
    def weighted_bce_loss(y_true, y_pred, weight, name):
        """

        :param y_true:
        :param y_pred:
        :param weight:
        :param name:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            logit_y_pred = tf.math.log(y_pred / (1. - y_pred))

            loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                   (tf.math.log(1. + tf.math.exp(-tf.math.abs(logit_y_pred))
                                ) + tf.math.maximum(-logit_y_pred, 0.))
            total_loss = tf.identity(tf.reduce_sum(
                loss) / tf.reduce_sum(weight), name='bce_loss')
        return total_loss

    @staticmethod
    def weighted_dice_loss(y_true, y_pred, weight, name):
        """

        :param y_true:
        :param y_pred:
        :param weight:
        :param name:
        :return:
        """
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            smooth = 1.
            w, m1, m2 = weight * weight, y_true, y_pred
            intersection = (m1 * m2)
            score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
                    (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
            loss = 1. - tf.reduce_sum(score)
            loss = tf.identity(loss, name='dice_loss')
        return loss

    @staticmethod
    def pad(inputdata, paddings, name):
        """
        :param inputdata:
        :param paddings:
        :return:
        """
        if TF_VERSION == '1.15.0':
            with tf.compat.v1.variable_scope(name_or_scope=name):
                return tf.pad(tensor=inputdata, paddings=paddings)
        else:
            with tf.variable_scope(name_or_scope=name):
                return tf.pad(tensor=inputdata, paddings=paddings)

    @staticmethod
    def spatial_pyramid_pool(input_tensor, out_pool_size, name, mode='avg_pool'):
        """Spatial pyramid pooling

        Parameters
        ----------
        input_tensor : tensor
            input tensor data
        out_pool_size : [type]
            [description]
        name : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        [_, in_height, in_width, _] = input_tensor.get_shape().as_list()
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name)
        with vars_scope:
            h_strd = h_size = math.ceil(float(in_height) / out_pool_size)
            w_strd = w_size = math.ceil(float(in_width) / out_pool_size)
            pad_h = int(out_pool_size * h_size - in_height)
            pad_w = int(out_pool_size * w_size - in_width)
            new_input_tensor = tf.pad(input_tensor, tf.constant(
                [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
            if mode == 'max_pool':
                if TF_VERSION == '1.15.0':
                    output_tensor = tf.nn.max_pool2d(
                        input=new_input_tensor,
                        ksize=[1, h_size, h_size, 1],
                        strides=[1, h_strd, w_strd, 1],
                        padding='SAME',
                        name='max_pool2d'
                    )
                else:
                    output_tensor = tf.nn.max_pool(
                        new_input_tensor,
                        ksize=[1, h_size, h_size, 1],
                        strides=[1, h_strd, w_strd, 1],
                        padding='SAME',
                        name='max_pool2d'
                    )
            elif mode == 'avg_pool':
                if TF_VERSION == '1.15.0':
                    output_tensor = tf.nn.avg_pool2d(
                        input=new_input_tensor,
                        ksize=[1, h_size, h_size, 1],
                        strides=[1, h_strd, w_strd, 1],
                        padding='SAME',
                        name='avg_pool2d'
                    )
                else:
                    output_tensor = tf.nn.avg_pool(
                        value=new_input_tensor,
                        ksize=[1, h_size, h_size, 1],
                        strides=[1, h_strd, w_strd, 1],
                        padding='SAME',
                        name='avg_pool2d'
                    )
            else:
                raise ValueError('Not support pool mode')
        return output_tensor
