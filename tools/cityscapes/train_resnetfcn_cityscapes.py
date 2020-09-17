#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 下午5:46
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/sfnet-tensorflow
# @File    : train_sfnet_cityscapes.py
# @IDE: PyCharm
"""
Train sfnet on cityscapes dataset
"""
from trainner.cityscapes import cityscapes_resnetfcn_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger('train_resnetfcn_cityscapes')
CFG = parse_config_utils.RESNET_FCN_CITYSCAPES_CFG


def train_model():
    """

    :return:
    """
    if CFG.TRAIN.MULTI_GPU.ENABLE:
        LOG.info('Using multi gpu trainner ...')
        worker = multi_gpu_trainner.ResNetFCNCityScapesMultiTrainer(cfg=CFG)
    else:
        raise NotImplementedError

    worker.train()
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()
