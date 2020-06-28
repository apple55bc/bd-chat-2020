#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/24 22:01
@File      :cfg.py
@Desc      :
"""
import os
from easydict import EasyDict as edict

join = os.path.join

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = join(MAIN_PATH, 'data')
MID_PATH = join(MAIN_PATH, 'mid')
MODEL_PATH = join(MAIN_PATH, 'model')
BERT_PATH = join(DATA_PATH, 'roberta')
OUT_PATH = join(MAIN_PATH, 'output')
FILE_DICT = {
    'train': join(DATA_PATH, 'train/train.txt'),
    'dev': join(DATA_PATH, 'dev/dev.txt'),
    'test': join(DATA_PATH, 'test_1/test_1.txt'),
    'test2': join(DATA_PATH, 'test_2/test_2.txt'),
}


data_num = {
    0: 6618,
    1: 946,
    2: 4645,
    3: 13666,
}
train_list = [0, 1]
totle_sample = 0
for t in train_list:
    totle_sample += data_num[t]
TAG = 'd2'


def __get_config():
    _config = edict()
    return _config


def __get_logger():
    import logging
    import datetime
    if not os.path.isdir(os.path.join(MAIN_PATH, 'logs')):
        os.makedirs(os.path.join(MAIN_PATH, 'logs'))

    LOG_PATH = os.path.join(MAIN_PATH, 'logs/log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    logging.basicConfig(filename=LOG_PATH,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO,
                        filemode='w', datefmt='%Y-%m-%d%I:%M:%S %p')
    _logger = logging.getLogger(__name__)

    #  添加日志输出到控制台
    console = logging.StreamHandler()
    _logger.addHandler(console)
    _logger.setLevel(logging.INFO)

    return _logger


config = __get_config()
logger = __get_logger()