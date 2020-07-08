#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午4:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV
# @File    : sfnet.py
# @IDE: PyCharm
"""
semantic flow model for image scene segmentation
"""
import tensorflow as tf

from sfnet_model import cnn_basenet


class _FAMModule(cnn_basenet.CNNBaseModel):
    """Flow alignment module for sfnet mode

    Parameters
    ----------
    cnn_basenet : [type]
        [description]
    """

    def __init__(self, phase) -> None:
        """

        """
        super(_FAMModule, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(
                    inputdata=result, is_training=self._is_training, name='bn', scale=True)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(
                    inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    @classmethod
    def _get_pixel_value(cls, input_tensor, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - input_tensor: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(input_tensor, indices)

    def _bilinear_sampler(self, input_tensor, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.
        Input[summary]
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.
        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.spatial_transformer_network
        """
        height = tf.shape(input_tensor)[1]
        width = tf.shape(input_tensor)[2]
        max_y = tf.cast(height - 1, 'int32')
        max_x = tf.cast(width - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        i_a = self._get_pixel_value(input_tensor, x0, y0)
        i_b = self._get_pixel_value(input_tensor, x0, y1)
        i_c = self._get_pixel_value(input_tensor, x1, y0)
        i_d = self._get_pixel_value(input_tensor, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa * i_a, wb * i_b, wc * i_c, wd * i_d],
                       name='bilinear_sample_output')

        return out

    def __call__(self, *args, **kwargs):
        """fam module function
        """
        input_tensor_low = kwargs['input_tensor_low']
        input_tensor_high = kwargs['input_tensor_high']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        [_, low_height, low_width, _] = input_tensor_low.get_shape().as_list()
        if tf.__version__ == '1.15.0':
            vars_scope = tf.compat.v1.variable_scope(name_or_scope=name_scope)
        else:
            vars_scope = tf.variable_scope(name_or_scope=name_scope)
        with vars_scope:
            # project input_tensor_low
            input_tensor_low = self.conv2d(
                inputdata=input_tensor_low,
                out_channel=output_channels,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='input_tensor_low_align_channels'
            )
            # upsample high features
            tensor_upsample = self.conv2d(
                inputdata=input_tensor_high,
                out_channel=output_channels,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='upsample_conv_1x1'
            )
            if tf.__version__ == '1.15.0':
                tensor_upsample = tf.compat.v1.image.resize_bilinear(
                    images=tensor_upsample,
                    size=(low_height, low_width),
                    align_corners=True,
                    name='upsampled_high_features'
                )
            else:
                tensor_upsample = tf.image.resize_bilinear(
                    images=tensor_upsample,
                    size=(low_height, low_width),
                    align_corners=True,
                    name='upsampled_high_features'
                )
            # generate grid
            input_tensor_low_project = self.conv2d(
                inputdata=input_tensor_low,
                out_channel=output_channels,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='input_tensor_low_project_conv_1x1'
            )
            sf_fileld_input_tensor = tf.concat(
                [input_tensor_low_project, tensor_upsample], axis=-1)
            sf_field_x = self.conv2d(
                inputdata=sf_fileld_input_tensor,
                out_channel=1,
                kernel_size=3,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='sf_field_x_logits'
            )
            sf_field_x = tf.nn.tanh(sf_field_x, name='sf_field_x')[:, :, :, 0]
            sf_field_y = self.conv2d(
                inputdata=sf_fileld_input_tensor,
                out_channel=1,
                kernel_size=3,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='sf_field_y_logits'
            )
            sf_field_y = tf.nn.tanh(sf_field_y, name='sf_field_y')[:, :, :, 0]
            # warp features
            warpped_features = self._bilinear_sampler(
                input_tensor=input_tensor_high,
                x=sf_field_x,
                y=sf_field_y
            )
            # fuse features
            output = tf.add(input_tensor_low, warpped_features,
                            name='fam_output')
        return output


if __name__ == '__main__':
    """
    test code
    """
    input_feature_low = tf.random.normal(
        shape=[4, 56, 56, 256], name='input_feature_map')
    input_feature_high = tf.random.normal(
        shape=[4, 28, 28, 256], name='input_feature_map')

    fam_module = _FAMModule(phase='test')
    fam_output = fam_module(
        input_tensor_low=input_feature_low,
        input_tensor_high=input_feature_high,
        output_channels=256,
        name='fam_stage_1'
    )
    print(fam_output)
