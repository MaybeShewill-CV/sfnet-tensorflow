#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-29 下午1:34
# @Author  : LuoYao
# @Site    : ICode
# @File    : resnet_utils.py
# @IDE: PyCharm
"""
Resnet local_utils
"""
import tensorflow as tf

from dual_attention_model import cnn_basenet


class ResnetBase(cnn_basenet.CNNBaseModel):
    """
    Resnet base local_utils
    """
    def __init__(self, phase):
        """

        """
        super(ResnetBase, self).__init__()
        if phase.lower() == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()

    def _init_phase(self):
        """
        init tensorflow bool flag
        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def _fixed_padding(self, inputs, kernel_size, name):
        """
        Pads the input along the spatial dimensions independently of input size.
        :param inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        :param kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                       Should be a positive integer.
        :param name: layer name
        :return: A tensor with the same format as the input with the data either intact
          (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        with tf.variable_scope(name_or_scope=name):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            padded_inputs = self.pad(
                inputdata=inputs,
                paddings=[[0, 0], [pad_beg, pad_end],
                          [pad_beg, pad_end], [0, 0]],
                name='pad')

        return padded_inputs

    def _conv2d_fixed_padding(self, inputs, kernel_size, output_dims, strides, name):
        """
        convolution op with fixed pad if stride greater than 1
        :param inputs: input tensor [batch, h, w, c]
        :param kernel_size: kernel size
        :param output_dims: output dims of conv op
        :param strides: stride of conv op
        :param name: layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            if strides > 1:
                inputs = self._fixed_padding(inputs, kernel_size, name='fix_padding')

            result = self.conv2d(
                inputdata=inputs,
                out_channel=output_dims,
                kernel_size=kernel_size,
                stride=strides,
                padding=('SAME' if strides == 1 else 'VALID'),
                use_bias=False,
                name='conv'
            )

        return result

    def _dilated_conv2d_fixed_padding(self, inputs, kernel_size, output_dims, strides, dilation_rate, name):
        """
        convolution op with fixed pad if stride greater than 1
        :param inputs: input tensor [batch, h, w, c]
        :param kernel_size: kernel size
        :param output_dims: output dims of conv op
        :param strides: stride of conv op
        :param name: layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            if strides > 1:
                inputs = self._fixed_padding(inputs, kernel_size, name='fix_padding')

            result = self.dilation_conv(
                input_tensor=inputs,
                k_size=kernel_size,
                out_dims=output_dims,
                rate=dilation_rate,
                padding=('SAME' if strides == 1 else 'VALID'),
                use_bias=False,
                name='dilated_conv'
            )

        return result

    def _bottleneck_block_v1(self, input_tensor, stride,
                             output_dims, projection_shortcut,
                             name):
        """
        A single bottleneck block for ResNet v2.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor: input tensor [batch, h, w, c]
        :param stride: the stride in the middle conv op
        :param output_dims: the output dims the final output dims will be output_dims * 4
        :param projection_shortcut: the project func and could be set to None if not needed
        :param name: the layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor

            if projection_shortcut is not None:
                shortcut = projection_shortcut(input_tensor)
                shortcut = self.layerbn(
                    inputdata=shortcut,
                    is_training=self._is_training,
                    name='project_bn'
                )

            # bottleneck part1
            inputs = self._conv2d_fixed_padding(
                inputs=input_tensor,
                kernel_size=1,
                output_dims=output_dims,
                strides=1,
                name='conv_pad_1'
            )
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            # bottleneck part2 repalce origin conv with dilation convolution op
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=3,
                output_dims=output_dims,
                strides=stride,
                name='conv_pad_2'
            )
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')

            # bottleneck part3
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims * 4,
                strides=1,
                name='conv_pad_3'
            )
            inputs = self.layerbn(
                inputdata=inputs,
                is_training=self._is_training,
                name='bn_3'
            )

            inputs = tf.add(inputs, shortcut, name='residual_add')
            inputs = self.relu(inputdata=inputs, name='residual_relu')

        return inputs

    def _bottleneck_block_v1_with_dilation(
            self, input_tensor, stride, output_dims, projection_shortcut, name, dilation_rate=2):
        """
        A single bottleneck block for ResNet v2.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor: input tensor [batch, h, w, c]
        :param stride: the stride in the middle conv op
        :param output_dims: the output dims the final output dims will be output_dims * 4
        :param projection_shortcut: the project func and could be set to None if not needed
        :param name: the layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor

            if projection_shortcut is not None:
                shortcut = projection_shortcut(input_tensor)
                shortcut = self.layerbn(
                    inputdata=shortcut,
                    is_training=self._is_training,
                    name='project_bn'
                )

            # bottleneck part1
            inputs = self._conv2d_fixed_padding(
                inputs=input_tensor,
                kernel_size=1,
                output_dims=output_dims,
                strides=1,
                name='conv_pad_1'
            )
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            # bottleneck part2 repalce origin conv with dilation convolution op
            inputs = self._dilated_conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=3,
                output_dims=output_dims,
                strides=stride,
                dilation_rate=dilation_rate,
                name='dilated_conv_pad_2'
            )
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')

            # bottleneck part3
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims * 4,
                strides=1,
                name='conv_pad_3'
            )
            inputs = self.layerbn(
                inputdata=inputs,
                is_training=self._is_training,
                name='bn_3'
            )

            inputs = tf.add(inputs, shortcut, name='residual_add')
            inputs = self.relu(inputdata=inputs, name='residual_relu')

        return inputs

    def _bottleneck_block_v2(self, input_tensor, stride,
                             output_dims, projection_shortcut,
                             name):
        """
        A single bottleneck block for ResNet v2.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor: input tensor [batch, h, w, c]
        :param stride: the stride in the middle conv op
        :param output_dims: the output dims the final output dims will be output_dims * 4
        :param projection_shortcut: the project func and could be set to None if not needed
        :param name: the layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            inputs = self.layerbn(inputdata=shortcut, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            # bottleneck part1
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims,
                strides=1,
                name='conv_pad_1'
            )
            # bottleneck part2 repalce origin conv with dilation convolution op
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=3,
                output_dims=output_dims,
                strides=stride,
                name='conv_pad_2'
            )
            # bottleneck part3
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_3')
            inputs = self.relu(inputdata=inputs, name='relu_3')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims * 4,
                strides=1,
                name='conv_pad_3'
            )

            inputs = tf.add(inputs, shortcut, name='residual_add')

        return inputs

    def _bottleneck_block_v2_with_dilation(
            self, input_tensor, stride, output_dims, projection_shortcut, name, dilation_rate=2):
        """
        A single bottleneck block for ResNet v2.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor: input tensor [batch, h, w, c]
        :param stride: the stride in the middle conv op
        :param output_dims: the output dims the final output dims will be output_dims * 4
        :param projection_shortcut: the project func and could be set to None if not needed
        :param name: the layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            inputs = self.layerbn(inputdata=shortcut, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            # bottleneck part1
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims,
                strides=1,
                name='conv_pad_1'
            )
            # bottleneck part2 repalce origin conv with dilation convolution op
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')
            inputs = self._dilated_conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=3,
                output_dims=output_dims,
                strides=stride,
                dilation_rate=dilation_rate,
                name='dilated_conv_pad_2'
            )
            # bottleneck part3
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_3')
            inputs = self.relu(inputdata=inputs, name='relu_3')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims * 4,
                strides=1,
                name='conv_pad_3'
            )

            inputs = tf.add(inputs, shortcut, name='residual_add')

        return inputs

    def _building_block_v1(self, input_tensor, strides, output_dims, name, projection_shortcut=None):
        """A single block for ResNet v1, without a bottleneck.
        Convolution then batch normalization then ReLU as described by:
          Deep Residual Learning for Image Recognition
          https://arxiv.org/pdf/1512.03385.pdf
          by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
        Args:
          input_tensor: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          output_dims: The number of filters for the convolutions.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
          name: the layer op name
        Returns:
          The output tensor of the block; shape should match inputs.
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor

            if projection_shortcut is not None:
                shortcut = projection_shortcut(input_tensor)
                shortcut = self.layerbn(
                    inputdata=shortcut,
                    is_training=self._is_training,
                    name='bn'
                )

            inputs = self._conv2d_fixed_padding(
                inputs=input_tensor, output_dims=output_dims, kernel_size=3, strides=strides,
                name='conv_pad_1')
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputs, name='relu_1')

            inputs = self._conv2d_fixed_padding(
                inputs=inputs, output_dims=output_dims, kernel_size=3, strides=1,
                name='conv_pad_2')
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs += shortcut
            inputs = self.relu(inputs, name='relu_2')

        return inputs

    def _building_block_v2(self, input_tensor, stride,
                           output_dims, name, projection_shortcut=None):
        """
        A single block for ResNet v2, without a bottleneck.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor:
        :param stride:
        :param output_dims:
        :param name:
        :param projection_shortcut:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            inputs = self.layerbn(
                inputdata=input_tensor,
                is_training=self._is_training,
                name='bn_1'
            )
            inputs = self.relu(
                inputdata=inputs,
                name='relu_1'
            )

            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)
            # building block part 1
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                output_dims=output_dims * 4,
                kernel_size=3,
                strides=stride,
                name='conv_pad_1'
            )
            # building block part 2
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                output_dims=output_dims * 4,
                kernel_size=3,
                strides=1,
                name='conv_pad_2'
            )

            inputs = tf.add(inputs, shortcut, name='residual_add')

        return inputs

    def _aspp_module(self, input_tensor, output_stride,
                     name, depth=256):
        """
        Implement the atrous spatial pyramid pooling
        module more details can be found in
        "Rethinking Atrous Convolution for Semantic Image Segmentation"
        :param input_tensor: input tensor [batch_size, h, w, c]
        :param output_stride: output stride
        :param name:
        :param depth: output feature dims
        :return:
        """
        def _conv_bn_relu(_input_tensor, _k_size,
                          _output_dims, _stride,
                          _padding, _is_training, _name):
            """
            conv-bn-relu stage
            :param _input_tensor: [batch_size, h, w, c]
            :param _k_size: kernel size
            :param _output_dims: the output feature dimensions
            :param _stride: conv stride
            :param _padding: conv padding
            :param _is_training: whether is training
            :param _name: conv stage name
            :return:
            """
            with tf.variable_scope(name_or_scope=_name):
                _conv = self.conv2d(inputdata=_input_tensor, out_channel=_output_dims,
                                    kernel_size=_k_size, padding=_padding, stride=_stride,
                                    use_bias=False, name='conv')
                _bn = self.layerbn(inputdata=_conv, is_training=_is_training, name='bn')
                _relu = self.relu(inputdata=_bn, name='relu')
                return _relu

        def _atrous_conv_bn_relu(_input_tensor, _k_size, _output_dims,
                                 _rate, _padding, _is_training, _name):
            """
            atrous_conv-bn-relu stage
            :param _input_tensor: [batch_size, h, w, c]
            :param _k_size: conv kernel_size
            :param _output_dims: output feature dimensions
            :param _rate: dilation rate
            :param _padding: conv padding
            :param _is_training: whether is training
            :param _name: conv stage name
            :return:
            """
            with tf.variable_scope(name_or_scope=_name):
                _atrous_conv = self.dilation_conv(
                    input_tensor=_input_tensor, k_size=_k_size,
                    out_dims=_output_dims, rate=_rate, padding=_padding,
                    use_bias=False, name='atrous_conv'
                )
                _bn = self.layerbn(inputdata=_atrous_conv, is_training=_is_training, name='bn')
                _relu = self.relu(inputdata=_bn, name='relu')
                return _relu

        with tf.variable_scope(name_or_scope=name):
            if output_stride not in [8, 16]:
                raise RuntimeError('Only support output of 8 or 16 for aspp module so far')

            if output_stride == 16:
                rates = [6, 12, 18]
            else:
                rates = [12, 24, 36]

            [_, image_h, image_w, _] = input_tensor.get_shape().as_list()

            # apply part a according to the origin paper
            conv_1x1 = _conv_bn_relu(
                _input_tensor=input_tensor, _k_size=1, _output_dims=depth,
                _stride=1, _padding='SAME', _is_training=self._is_training, _name='conv_1x1'
            )
            atrous_conv_6 = _atrous_conv_bn_relu(
                _input_tensor=input_tensor, _k_size=3, _output_dims=depth,
                _rate=rates[0], _padding='SAME', _is_training=self._is_training, _name='atrous_conv_3x3_6'
            )
            atrous_conv_12 = _atrous_conv_bn_relu(
                _input_tensor=input_tensor, _k_size=3, _output_dims=depth,
                _rate=rates[1], _padding='SAME', _is_training=self._is_training, _name='atrous_conv_3x3_12'
            )
            atrous_conv_18 = _atrous_conv_bn_relu(
                _input_tensor=input_tensor, _k_size=3, _output_dims=depth,
                _rate=rates[2], _padding='SAME', _is_training=self._is_training, _name='atrous_conv_3x3_18'
            )
            # apply part 2 to extract image level features
            image_level_feats = self.globalavgpooling(
                inputdata=input_tensor, name='global_avg_pooling',
                keepdims=True
            )
            image_level_feats = _conv_bn_relu(
                _input_tensor=image_level_feats, _k_size=1, _output_dims=depth,
                _stride=1, _padding='SAME', _is_training=self._is_training, _name='conv_1x1_image_level_feats'
            )
            image_level_feats = tf.image.resize_bilinear(
                image_level_feats, [image_h, image_w],
                name='upsampled_image_level_feats'
            )
            # fuse all feats
            concat_feats = tf.concat(
                [conv_1x1, atrous_conv_6, atrous_conv_12,
                 atrous_conv_18, image_level_feats],
                axis=3,
                name='aspp_concat_feats'
            )
            fused_feats = _conv_bn_relu(
                _input_tensor=concat_feats,
                _k_size=1,
                _output_dims=depth,
                _stride=1,
                _padding='SAME',
                _is_training=self._is_training,
                _name='aspp_fused_feats'
            )

            return fused_feats
