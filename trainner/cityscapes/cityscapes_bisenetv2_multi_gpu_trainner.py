#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/04/10 下午4:02
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : cityscapes_bisenetv2_multi_gpu_trainner.py
# @IDE: PyCharm
"""
Bisenetv2 multi gpu trainner for cityscapes dataset
"""
import os
import os.path as ops
import shutil
import time
import math

import numpy as np
import tensorflow as tf
import loguru
import tqdm

from bisenet_model import bisenet_v2
from local_utils.config_utils import parse_config_utils
from data_provider.cityscapes import cityscapes_tf_io

LOG = loguru.logger
CFG = parse_config_utils.cityscapes_cfg_v2


class BiseNetV2CityScapesMultiTrainer(object):
    """
    init bisenetv2 multi gpu trainner
    """
    def __init__(self):
        """
        initialize bisenetv2 multi gpu trainner
        """
        # define solver params and dataset
        self._cityscapes_io = cityscapes_tf_io.CityScapesTfIO()
        self._train_dataset = self._cityscapes_io.train_dataset_reader
        self._val_dataset = self._cityscapes_io.val_dataset_reader
        self._steps_per_epoch = len(self._train_dataset)
        self._val_steps_per_epoch = len(self._val_dataset)

        self._model_name = CFG.MODEL.MODEL_NAME

        self._train_epoch_nums = CFG.TRAIN.EPOCH_NUMS
        self._batch_size = CFG.TRAIN.BATCH_SIZE
        self._val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
        self._snapshot_epoch = CFG.TRAIN.SNAPSHOT_EPOCH
        self._model_save_dir = ops.join(CFG.TRAIN.MODEL_SAVE_DIR, self._model_name)
        self._tboard_save_dir = ops.join(CFG.TRAIN.TBOARD_SAVE_DIR, self._model_name)
        self._enable_miou = CFG.TRAIN.COMPUTE_MIOU.ENABLE
        if self._enable_miou:
            self._record_miou_epoch = CFG.TRAIN.COMPUTE_MIOU.EPOCH
        self._input_tensor_size = [int(tmp / 2) for tmp in CFG.AUG.TRAIN_CROP_SIZE]
        self._gpu_devices = CFG.TRAIN.MULTI_GPU.GPU_DEVICES
        self._gpu_nums = len(self._gpu_devices)
        self._chief_gpu_index = CFG.TRAIN.MULTI_GPU.CHIEF_DEVICE_INDEX
        self._batch_size_per_gpu = int(self._batch_size / self._gpu_nums)

        self._init_learning_rate = CFG.SOLVER.LR
        self._moving_ave_decay = CFG.SOLVER.MOVING_AVE_DECAY
        self._momentum = CFG.SOLVER.MOMENTUM
        self._lr_polynimal_decay_power = CFG.SOLVER.LR_POLYNOMIAL_POWER
        self._optimizer_mode = CFG.SOLVER.OPTIMIZER.lower()

        if CFG.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            self._initial_weight = CFG.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH
        else:
            self._initial_weight = None
        if CFG.TRAIN.WARM_UP.ENABLE:
            self._warmup_epoches = CFG.TRAIN.WARM_UP.EPOCH_NUMS
            self._warmup_init_learning_rate = self._init_learning_rate / 1000.0
        else:
            self._warmup_epoches = 0

        # define tensorflow session
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config)

        # define graph input tensor
        with tf.variable_scope(name_or_scope='graph_input_node'):
            self._input_src_image_list = []
            self._input_label_image_list = []
            for i in range(self._gpu_nums):
                src_imgs, label_imgs = self._train_dataset.next_batch(batch_size=self._batch_size_per_gpu)
                self._input_src_image_list.append(src_imgs)
                self._input_label_image_list.append(label_imgs)
            self._val_input_src_image, self._val_input_label_image = self._val_dataset.next_batch(
                batch_size=self._val_batch_size
            )

        # define model
        self._model = bisenet_v2.BiseNetV2(phase='train', cfg=CFG)
        self._val_model = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)

        # define average container
        tower_grads = []
        tower_total_loss = []
        tower_l2_loss = []
        batchnorm_updates = None

        # define learning rate
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            self._val_global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='val_global_step')
            self._val_global_step_update = tf.assign_add(self._val_global_step, 1.0)
            warmup_steps = tf.constant(
                self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32, name='warmup_steps'
            )
            train_steps = tf.constant(
                self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32, name='train_steps'
            )
            self._learn_rate = tf.cond(
                pred=self._global_step < warmup_steps,
                true_fn=lambda: self._compute_warmup_lr(warmup_steps=warmup_steps, name='warmup_lr'),
                false_fn=lambda: tf.train.polynomial_decay(
                    learning_rate=self._init_learning_rate,
                    global_step=self._global_step,
                    decay_steps=train_steps,
                    end_learning_rate=0.000000001,
                    power=self._lr_polynimal_decay_power)
            )
            self._learn_rate = tf.identity(self._learn_rate, 'lr')

        # define optimizer
        if self._optimizer_mode == 'sgd':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._learn_rate,
                momentum=self._momentum
            )
        elif self._optimizer_mode == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learn_rate,
            )
        else:
            raise NotImplementedError('Not support optimizer: {:s} for now'.format(self._optimizer_mode))

        # define distributed train op
        with tf.variable_scope(tf.get_variable_scope()):
            is_network_initialized = False
            for i in range(self._gpu_nums):
                with tf.device('/gpu:{:d}'.format(i)):
                    with tf.name_scope('tower_{:d}'.format(i)) as _:
                        input_images = self._input_src_image_list[i]
                        input_labels = self._input_label_image_list[i]
                        tmp_loss, tmp_grads = self._compute_net_gradients(
                            input_images, input_labels, optimizer,
                            is_net_first_initialized=is_network_initialized
                        )
                        is_network_initialized = True

                        # Only use the mean and var in the chief gpu tower to update the parameter
                        if i == self._chief_gpu_index:
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        tower_grads.append(tmp_grads)
                        tower_total_loss.append(tmp_loss['total_loss'])
                        tower_l2_loss.append(tmp_loss['l2_loss'])
        grads = self._average_gradients(tower_grads)
        self._loss = tf.reduce_mean(tower_total_loss, name='reduce_mean_tower_total_loss')
        self._l2_loss = tf.reduce_mean(tower_l2_loss, name='reduce_mean_tower_l2_loss')
        ret = self._val_model.compute_loss(
            input_tensor=self._val_input_src_image,
            label_tensor=self._val_input_label_image,
            name='BiseNetV2',
            reuse=True
        )
        self._val_loss = ret['total_loss']
        self._val_l2_loss = ret['l2_loss']

        # define moving average op
        with tf.variable_scope(name_or_scope='moving_avg'):
            if CFG.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(self._moving_ave_decay).apply(
                train_var_list + tf.moving_average_variables()
            )

        # group all the op needed for training
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self._global_step)
        self._train_op = tf.group(apply_gradient_op, moving_ave_op, batchnorm_updates_op)

        # define prediction
        self._prediciton = self._model.inference(
            input_tensor=self._input_src_image_list[self._chief_gpu_index],
            name='BiseNetV2',
            reuse=True
        )
        self._val_prediction = self._val_model.inference(
            input_tensor=self._val_input_src_image,
            name='BiseNetV2',
            reuse=True
        )

        # define miou
        if self._enable_miou:
            with tf.variable_scope('miou'):
                pred = tf.reshape(self._prediciton, [-1, ])
                gt = tf.reshape(self._input_label_image_list[self._chief_gpu_index], [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(gt, CFG.DATASET.NUM_CLASSES - 1)), 1)
                gt = tf.gather(gt, indices)
                pred = tf.gather(pred, indices)
                self._miou, self._miou_update_op = tf.metrics.mean_iou(
                    labels=gt,
                    predictions=pred,
                    num_classes=CFG.DATASET.NUM_CLASSES
                )

                val_pred = tf.reshape(self._val_prediction, [-1, ])
                val_gt = tf.reshape(self._val_input_label_image, [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(val_gt, CFG.DATASET.NUM_CLASSES - 1)), 1)
                val_gt = tf.gather(val_gt, indices)
                val_pred = tf.gather(val_pred, indices)
                self._val_miou, self._val_miou_update_op = tf.metrics.mean_iou(
                    labels=val_gt,
                    predictions=val_pred,
                    num_classes=CFG.DATASET.NUM_CLASSES
                )

        # define saver and loader
        with tf.variable_scope('loader_and_saver'):
            self._net_var = [vv for vv in tf.global_variables() if 'lr' not in vv.name]
            self._loader = tf.train.Saver(self._net_var)
            self._saver = tf.train.Saver(max_to_keep=10)

        # define summary
        with tf.variable_scope('summary'):
            summary_merge_list = [
                tf.summary.scalar("learn_rate", self._learn_rate),
                tf.summary.scalar("total_loss", self._loss),
                tf.summary.scalar('l2_loss', self._l2_loss)
            ]
            val_summary_merge_list = [
                tf.summary.scalar('val_total_loss', self._val_loss),
                tf.summary.scalar('val_l2_loss', self._val_l2_loss)
            ]
            if self._enable_miou:
                with tf.control_dependencies([self._miou_update_op]):
                    summary_merge_list_with_miou = [
                        tf.summary.scalar("learn_rate", self._learn_rate),
                        tf.summary.scalar("total_loss", self._loss),
                        tf.summary.scalar('l2_loss', self._l2_loss),
                        tf.summary.scalar('miou', self._miou)
                    ]
                    self._write_summary_op_with_miou = tf.summary.merge(summary_merge_list_with_miou)
                with tf.control_dependencies([self._val_miou_update_op, self._val_global_step_update]):
                    val_summary_merge_list_with_miou = [
                        tf.summary.scalar('val_total_loss', self._val_loss),
                        tf.summary.scalar('val_l2_loss', self._val_l2_loss),
                        tf.summary.scalar('val_miou', self._val_miou),
                    ]
                    self._val_write_summary_op_with_miou = tf.summary.merge(val_summary_merge_list_with_miou)
            if ops.exists(self._tboard_save_dir):
                shutil.rmtree(self._tboard_save_dir)
            os.makedirs(self._tboard_save_dir, exist_ok=True)
            model_params_file_save_path = ops.join(self._tboard_save_dir, CFG.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME)
            with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj:
                CFG.dump_to_json_file(f_obj)
            self._write_summary_op = tf.summary.merge(summary_merge_list)
            self._val_write_summary_op = tf.summary.merge(val_summary_merge_list)
            self._summary_writer = tf.summary.FileWriter(self._tboard_save_dir, graph=self._sess.graph)

        LOG.info('Initialize cityscapes bisenetv2 multi gpu trainner complete')

    @staticmethod
    def _average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def _compute_warmup_lr(self, warmup_steps, name):
        """

        :param warmup_steps:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            factor = tf.math.pow(self._init_learning_rate / self._warmup_init_learning_rate, 1.0 / warmup_steps)
            warmup_lr = self._warmup_init_learning_rate * tf.math.pow(factor, self._global_step)
        return warmup_lr

    def _compute_net_gradients(self, images, labels, optimizer=None, is_net_first_initialized=False):
        """
        Calculate gradients for single GPU
        :param images: images for training
        :param labels: labels corresponding to images
        :param optimizer: network optimizer
        :param is_net_first_initialized: if the network is initialized
        :return:
        """
        net_loss = self._model.compute_loss(
            input_tensor=images,
            label_tensor=labels,
            name='BiseNetV2',
            reuse=is_net_first_initialized
        )

        if CFG.TRAIN.FREEZE_BN.ENABLE:
            train_var_list = [
                v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
            ]
        else:
            train_var_list = tf.trainable_variables()

        if optimizer is not None:
            grads = optimizer.compute_gradients(net_loss['total_loss'], var_list=train_var_list)
        else:
            grads = None

        return net_loss, grads

    def train(self):
        """

        :return:
        """
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if CFG.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight))
                self._loader.restore(self._sess, self._initial_weight)
                global_step_value = self._sess.run(self._global_step)
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch)
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums
            except OSError as e:
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train BiseNetV2 from scratch ...')
                epoch_start_pt = 1
            except Exception as e:
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train BiseNetV2 from scratch ...')
                epoch_start_pt = 1
        else:
            LOG.info('=> Starts to train BiseNetV2 from scratch ...')
            epoch_start_pt = 1

        best_model = []
        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            # training part
            train_epoch_losses = []
            train_epoch_mious = []
            traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch))
            for _ in traindataset_pbar:
                if self._enable_miou and epoch % self._record_miou_epoch == 0:
                    _, _, summary, train_step_loss, global_step_val = self._sess.run(
                        fetches=[
                            self._train_op, self._miou_update_op, self._write_summary_op_with_miou,
                            self._loss, self._global_step
                        ]
                    )
                    train_step_miou = self._sess.run(
                        fetches=self._miou
                    )
                    train_epoch_losses.append(train_step_loss)
                    train_epoch_mious.append(train_step_miou)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}, miou: {:.5f}'.format(train_step_loss, train_step_miou)
                    )
                else:
                    _, summary, train_step_loss, global_step_val = self._sess.run(
                        fetches=[
                            self._train_op, self._write_summary_op,
                            self._loss, self._global_step
                        ]
                    )
                    train_epoch_losses.append(train_step_loss)
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description(
                        'train loss: {:.5f}'.format(train_step_loss)
                    )

            train_epoch_losses = np.mean(train_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                train_epoch_mious = np.mean(train_epoch_mious)

            # validation part
            val_epoch_losses = []
            val_epoch_mious = []
            valdataset_pbar = tqdm.tqdm(range(1, self._val_steps_per_epoch))
            for _ in valdataset_pbar:
                try:
                    if self._enable_miou and epoch % self._record_miou_epoch == 0:
                        _, val_summary, val_step_loss, val_global_step_val = self._sess.run(
                            fetches=[
                                self._val_miou_update_op, self._val_write_summary_op_with_miou,
                                self._val_loss, self._val_global_step
                            ]
                        )
                        val_step_miou = self._sess.run(
                            fetches=self._val_miou
                        )
                        val_epoch_losses.append(val_step_loss)
                        val_epoch_mious.append(val_step_miou)
                        self._summary_writer.add_summary(val_summary, global_step=val_global_step_val)
                        valdataset_pbar.set_description(
                            'val loss: {:.5f}, val miou: {:.5f}'.format(val_step_loss, val_step_miou)
                        )
                    else:
                        val_summary, val_step_loss, val_global_step_val = self._sess.run(
                            fetches=[
                                self._val_write_summary_op,
                                self._val_loss, self._val_global_step
                            ]
                        )
                        val_epoch_losses.append(val_step_loss)
                        self._summary_writer.add_summary(val_summary, global_step=val_global_step_val)
                        valdataset_pbar.set_description(
                            'val loss: {:.5f}'.format(val_step_loss)
                        )
                except tf.errors.OutOfRangeError as _:
                    break
            val_epoch_losses = np.mean(val_epoch_losses)
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                val_epoch_mious = np.mean(val_epoch_mious)

            # model saving part
            if epoch % self._snapshot_epoch == 0:
                if self._enable_miou:
                    if len(best_model) < 10:
                        best_model.append(val_epoch_mious)
                        best_model = sorted(best_model)
                        snapshot_model_name = 'cityscapes_val_miou={:.4f}.ckpt'.format(val_epoch_mious)
                        snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                        os.makedirs(self._model_save_dir, exist_ok=True)
                        self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                    else:
                        best_model = sorted(best_model)
                        if val_epoch_mious > best_model[0]:
                            best_model[0] = val_epoch_mious
                            best_model = sorted(best_model)
                            snapshot_model_name = 'cityscapes_val_miou={:.4f}.ckpt'.format(val_epoch_mious)
                            snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                            os.makedirs(self._model_save_dir, exist_ok=True)
                            self._saver.save(self._sess, snapshot_model_path, global_step=epoch)
                        else:
                            pass
                else:
                    snapshot_model_name = 'cityscapes_val_loss={:.4f}.ckpt'.format(val_epoch_losses)
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                    os.makedirs(self._model_save_dir, exist_ok=True)
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self._enable_miou and epoch % self._record_miou_epoch == 0:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Train miou: {:.5f} '
                    'Val loss: {:.5f} Val miou: {:.5f}...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        train_epoch_mious,
                        val_epoch_losses,
                        val_epoch_mious
                    )
                )
            else:
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Val loss: {:.5f}...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        val_epoch_losses
                    )
                )
        if self._enable_miou:
            LOG.info('Best model\'s val mious are: {}'.format(best_model))
        LOG.info('Complete training process good luck!!')

        return


if __name__ == '__main__':
    """
    test code
    """
    worker = BiseNetV2CityScapesMultiTrainer()
    print('Init complete')
