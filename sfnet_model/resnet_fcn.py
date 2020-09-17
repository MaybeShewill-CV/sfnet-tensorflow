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
import collections
import time

import tensorflow as tf

from local_utils.config_utils import parse_config_utils
from sfnet_model import cnn_basenet, resnet


class _FFModule(cnn_basenet.CNNBaseModel):
    """
    Feature fused module for resnet fcn model
    """
    def __init__(self, phase):
        """init function

        Parameters
        ----------
        phase : str
            train phase or test phase
        """
        super(_FFModule, self).__init__()
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

    def __call__(self, *args, **kwargs):
        """fam module function
        """
        input_tensor_low_origin = kwargs['input_tensor_low']
        input_tensor_high = kwargs['input_tensor_high']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        [_, low_height, low_width, _] = input_tensor_low_origin.get_shape().as_list()
        with tf.variable_scope(name_or_scope=name_scope):
            # project input_tensor_low
            input_tensor_low_origin = self._conv_block(
                input_tensor=input_tensor_low_origin,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='input_tensor_low_align_project',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            # upsample high features
            tensor_upsample = self._conv_block(
                input_tensor=input_tensor_high,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='upsample_conv_1x1',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            tensor_upsample = tf.image.resize_bilinear(
                images=tensor_upsample,
                size=(low_height, low_width),
                align_corners=True,
                name='upsampled_high_features'
            )
            # fuse features
            fused_features = tf.add(input_tensor_low_origin, tensor_upsample, name='fused_features')
            fused_output = self._conv_block(
                input_tensor=fused_features,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='fused_output',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
        return fused_output


class _PPModule(cnn_basenet.CNNBaseModel):
    """
    Pyramid Pooling Module
    """
    def __init__(self, phase):
        """init function

        Parameters
        ----------
        phase : str
            train phase or test phase
        """
        super(_PPModule, self).__init__()
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

    def __call__(self, *args, **kwargs):
        """
        ppm module function
        """
        input_tensor = kwargs['input_tensor']
        output_pool_sizes = kwargs['output_pool_sizes']
        name_scope = kwargs['name']
        [_, in_height, in_width, in_channels] = input_tensor.get_shape().as_list()
        ppm_features = [input_tensor]
        with tf.variable_scope(name_or_scope=name_scope):
            for output_pool_size in output_pool_sizes:
                ppm_feature = self.spatial_pyramid_pool(
                    input_tensor=input_tensor,
                    out_pool_size=output_pool_size,
                    name='ppm_pool_size_{:d}'.format(output_pool_size),
                    mode='avg_pool'
                )
                ppm_feature = self._conv_block(
                    input_tensor=ppm_feature,
                    k_size=1,
                    output_channels=in_channels,
                    stride=1,
                    name='ppm_pool_size_{:d}_project'.format(output_pool_size),
                    padding='SAME',
                    use_bias=False,
                    need_activate=False
                )
                ppm_feature = tf.image.resize_bilinear(
                    images=ppm_feature,
                    size=(in_height, in_width),
                    name='ppm_pool_size_{:d}_upsample'.format(output_pool_size)
                )
                ppm_features.append(ppm_feature + input_tensor)

            output_tensor = tf.concat(ppm_features, axis=-1, name='ppm_concate')
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=in_channels,
                stride=1,
                name='ppm_output'.format(output_pool_size),
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            return output_tensor


class _SegmentationHead(cnn_basenet.CNNBaseModel):
    """
    implementation of segmentation head in sfnet v2
    """

    def __init__(self, phase):
        """

        """
        super(_SegmentationHead, self).__init__()
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

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        ratio = kwargs['upsample_ratio']
        input_tensor_size = input_tensor.get_shape().as_list()[1:3]
        output_tensor_size = [int(tmp * ratio) for tmp in input_tensor_size]
        feature_dims = kwargs['feature_dims']
        classes_nums = kwargs['classes_nums']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=feature_dims,
                stride=1,
                name='3x3_conv_block',
                padding=self._padding,
                use_bias=False,
                need_activate=True
            )
            result = self.conv2d(
                inputdata=result,
                out_channel=classes_nums,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=False,
                name='1x1_conv_block'
            )
            result = tf.image.resize_bilinear(
                result,
                output_tensor_size,
                name='segmentation_head_logits'
            )
        return result


class ResNetFCN(cnn_basenet.CNNBaseModel):
    """
    Semantic flow Net
    """
    def __init__(self, phase, cfg):
        """init net

        Args:
            phase (str): train mode or test mode
            cfg (Config): Config
        """
        super(ResNetFCN, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._basenet = resnet.ResNet(phase=phase, cfg=self._cfg)
        self._ppm_output_sizes = [1, 2, 3, 6]
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._use_boost_seg_head = self._cfg.SOLVER.BOOST_SEG_HEAD.ENABLE
        self._enable_ohem = self._cfg.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = cfg.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = cfg.SOLVER.OHEM.MIN_SAMPLE_NUMS

        # set module used in sfnet
        self._ffm_block = _FFModule(phase=phase)
        self._ppm_block = _PPModule(phase=phase)
        self._seg_head_block = _SegmentationHead(phase=phase)

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

    @classmethod
    def _compute_cross_entropy_loss(cls, seg_logits, labels, class_nums, name):
        """

        :param seg_logits:
        :param labels:
        :param class_nums:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # first check if the logits' shape is matched with the labels'
            seg_logits_shape = seg_logits.shape[1:3]
            labels_shape = labels.shape[1:3]
            seg_logits = tf.cond(
                tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
                true_fn=lambda: seg_logits,
                false_fn=lambda: tf.image.resize_bilinear(
                    seg_logits, labels_shape)
            )
            seg_logits = tf.reshape(seg_logits, [-1, class_nums])
            labels = tf.reshape(labels, [-1, ])
            indices = tf.squeeze(
                tf.where(tf.less_equal(labels, class_nums - 1)), 1)
            seg_logits = tf.gather(seg_logits, indices)
            labels = tf.cast(tf.gather(labels, indices), tf.int32)

            # compute cross entropy loss
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=seg_logits
                ),
                name='cross_entropy_loss'
            )
        return loss

    @classmethod
    def _compute_ohem_cross_entropy_loss(cls, seg_logits, labels, class_nums, name, thresh, n_min):
        """

        :param seg_logits:
        :param labels:
        :param class_nums:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # first check if the logits' shape is matched with the labels'
            seg_logits_shape = seg_logits.shape[1:3]
            labels_shape = labels.shape[1:3]
            seg_logits = tf.cond(
                tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
                true_fn=lambda: seg_logits,
                false_fn=lambda: tf.image.resize_bilinear(seg_logits, labels_shape)
            )
            seg_logits = tf.reshape(seg_logits, [-1, class_nums])
            labels = tf.reshape(labels, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(labels, class_nums - 1)), 1)
            seg_logits = tf.gather(seg_logits, indices)
            labels = tf.cast(tf.gather(labels, indices), tf.int32)

            # compute cross entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=seg_logits
            )
            loss, _ = tf.nn.top_k(loss, tf.size(loss), sorted=True)

            # apply ohem
            ohem_thresh = tf.multiply(-1.0, tf.math.log(thresh), name='ohem_score_thresh')
            ohem_cond = tf.greater(loss[n_min], ohem_thresh)
            loss_select = tf.cond(
                pred=ohem_cond,
                true_fn=lambda: tf.gather(loss, tf.squeeze(tf.where(tf.greater(loss, ohem_thresh)), 1)),
                false_fn=lambda: loss[:n_min]
            )
            loss_value = tf.reduce_mean(loss_select, name='ohem_cross_entropy_loss')
        return loss_value

    @classmethod
    def _compute_l2_reg_loss(cls, var_list, weights_decay, name):
        """

        :param var_list:
        :param weights_decay:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in var_list:
                if 'beta' in vv.name or 'gamma' in vv.name or 'b:0' in vv.name.split('/')[-1]:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= weights_decay
            l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')

        return l2_reg_loss

    def build_model(self, input_tensor, reuse=False, prepare_data_for_booster=False):
        """

        :param input_tensor:
        :param reuse:
        :param prepare_data_for_booster:
        :return:
        """
        with tf.variable_scope(name_or_scope='encoder'):
            encoded_features = self._basenet.inference(
                input_tensor=input_tensor,
                name='resnet_backbone',
                reuse=reuse
            )
        decoded_seg_logits = collections.OrderedDict()
        with tf.variable_scope(name_or_scope='decoder'):
            output_channels = 128 if self._cfg.MODEL.RESNET.NET_SIZE < 50 else 128
            # first apply ppm
            encoded_final_features = self.conv2d(
                inputdata=encoded_features['stage_4'],
                out_channel=output_channels,
                kernel_size=1,
                padding='SAME',
                stride=1,
                use_bias=False,
                name='ppm_input_tensor'
            )
            ppm_output_tensor = self._ppm_block(
                input_tensor=encoded_final_features,
                output_pool_sizes=self._ppm_output_sizes,
                name='ppm'
            )
            decode_stage_1 = ppm_output_tensor
            decode_stage_2 = self._ffm_block(
                input_tensor_low=encoded_features['stage_3'],
                input_tensor_high=decode_stage_1,
                output_channels=output_channels,
                name='decode_fam_stage_1'
            )
            decode_stage_3 = self._ffm_block(
                input_tensor_low=encoded_features['stage_2'],
                input_tensor_high=decode_stage_2,
                output_channels=output_channels,
                name='decode_fam_stage_2'
            )
            decode_stage_4 = self._ffm_block(
                input_tensor_low=encoded_features['stage_1'],
                input_tensor_high=decode_stage_3,
                output_channels=output_channels,
                name='decode_fam_stage_3'
            )
            final_decode_stage_4 = decode_stage_4
            final_decode_stage_3 = self._ffm_block(
                input_tensor_low=final_decode_stage_4,
                input_tensor_high=decode_stage_3,
                output_channels=output_channels,
                name='final_decode_fam_stage_3'
            )
            final_decode_stage_2 = self._ffm_block(
                input_tensor_low=final_decode_stage_4,
                input_tensor_high=decode_stage_2,
                output_channels=output_channels,
                name='final_decode_fam_stage_2'
            )
            final_decode_stage_1 = self._ffm_block(
                input_tensor_low=final_decode_stage_4,
                input_tensor_high=decode_stage_1,
                output_channels=output_channels,
                name='final_decode_fam_stage_1'
            )
            final_decode_features = tf.concat(
                [final_decode_stage_4, final_decode_stage_3, final_decode_stage_2, final_decode_stage_1],
                axis=-1,
                name='final_decode_features'
            )
            final_stage_logits = self._seg_head_block(
                input_tensor=final_decode_features,
                name='final_stage_logits',
                upsample_ratio=4,
                feature_dims=256,
                classes_nums=self._class_nums
            )
            decoded_seg_logits['final_stage'] = final_stage_logits
        if prepare_data_for_booster:
            decode_stage_2_logits = self._seg_head_block(
                input_tensor=decode_stage_2,
                name='decode_stage_2_logits',
                upsample_ratio=16,
                feature_dims=256,
                classes_nums=self._class_nums
            )
            decoded_seg_logits['stage_2'] = decode_stage_2_logits
            decode_stage_3_logits = self._seg_head_block(
                input_tensor=decode_stage_3,
                name='decode_stage_3_logits',
                upsample_ratio=8,
                feature_dims=256,
                classes_nums=self._class_nums
            )
            decoded_seg_logits['stage_3'] = decode_stage_3_logits
            decode_stage_4_logits = self._seg_head_block(
                input_tensor=decode_stage_4,
                name='decode_stage_4_logits',
                upsample_ratio=4,
                feature_dims=256,
                classes_nums=self._class_nums
            )
            decoded_seg_logits['stage_4'] = decode_stage_4_logits

        return decoded_seg_logits

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            segment_logits = self.build_model(
                input_tensor=input_tensor,
                reuse=reuse,
                prepare_data_for_booster=False
            )['final_stage']
            segment_score = tf.nn.softmax(logits=segment_logits, name='prob')
            segment_prediction = tf.argmax(
                segment_score, axis=-1, name='prediction')
        return segment_prediction

    def compute_loss(self, input_tensor, label_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param label_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            net_seg_logits = self.build_model(
                input_tensor=input_tensor,
                reuse=reuse,
                prepare_data_for_booster=True
            )
            # compute network loss
            segment_loss = tf.constant(0.0, tf.float32)
            for stage_name, seg_logits in net_seg_logits.items():
                loss_stage_name = '{:s}_segmentation_loss'.format(stage_name)
                if self._loss_type == 'cross_entropy':
                    if not self._enable_ohem:
                        segment_loss += self._compute_cross_entropy_loss(
                            seg_logits=seg_logits,
                            labels=label_tensor,
                            class_nums=self._class_nums,
                            name=loss_stage_name
                        )
                    else:
                        segment_loss += self._compute_ohem_cross_entropy_loss(
                            seg_logits=seg_logits,
                            labels=label_tensor,
                            class_nums=self._class_nums,
                            name=loss_stage_name,
                            thresh=self._ohem_score_thresh,
                            n_min=self._ohem_min_sample_nums
                        )
                else:
                    raise NotImplementedError('Not supported loss of type: {:s}'.format(self._loss_type))
            var_list = tf.trainable_variables()
            l2_reg_loss = self._compute_l2_reg_loss(
                var_list=var_list,
                weights_decay=self._weights_decay,
                name='segment_l2_loss'
            )
            total_loss = segment_loss + l2_reg_loss
            total_loss = tf.identity(total_loss, name='total_loss')

            ret = {
                'total_loss': total_loss,
                'l2_loss': l2_reg_loss,
            }
        return ret


def main():
    """
    test code
    """
    input_tensor = tf.random.uniform([1, 512, 512, 3], name='input_tensor')
    label_tensor = tf.ones([1, 512, 512], name='input_tensor', dtype=tf.int32)
    net = ResNetFCN(phase='train', cfg=parse_config_utils.RESNET_FCN_CITYSCAPES_CFG)

    loss_set = net.compute_loss(
        input_tensor=input_tensor,
        label_tensor=label_tensor,
        reuse=False,
        name='ResNetFcn'
    )

    inference_result = net.inference(
        input_tensor=input_tensor,
        reuse=True,
        name='ResNetFcn'
    )
    print(inference_result)

    for loss_name, loss_t in loss_set.items():
        print('Loss: {:s}, {}'.format(loss_name, loss_t))
    
    sess = tf.Session()
    with sess.as_default():
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        loop_times = 500

        t_start = time.time()
        for _ in range(loop_times):
            preds = sess.run(inference_result)
        t_cost = (time.time() - t_start) / loop_times
        print('Net inference cost time: {:.5f}'.format(t_cost))
        print('Net inference can reach: {:.5f} fps'.format(1.0 / t_cost))


if __name__ == '__main__':
    """
    main func
    """
    main()
