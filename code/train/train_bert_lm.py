#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/28 22:21
@File      :train_bert_lm.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.bert_lm import BertLM, Response
from data_deal.base_input import BaseInput
from bert4keras_5_8.backend import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=0)
parser.add_argument('--epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=3)
args = parser.parse_args(sys.argv[1:])

batch_size = 4
steps_per_epoch = 120

epoches = int(args.epoch * totle_sample / batch_size / steps_per_epoch)
init_epoch = args.init_epoch

save_dir = join(MODEL_PATH, 'BertLM_' + TAG)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
save_path = join(save_dir, 'trained.h5')

data_input = BaseInput(from_pre_trans=True)

model_cls = BertLM(data_input.keep_tokens, load_path=save_path)
model_cls.compile()


class LogRecord(keras.callbacks.Callback):
    def __init__(self):
        super(LogRecord, self).__init__()
        self._step = 1
        self.lowest = 1e10
        self.test_iter = data_input.get_sample(
            3,
            need_shuffle=False,
            cycle=True
        )
        self.response = Response(model_cls.model,
                                 model_cls.session,
                                 data_input,
                                 start_id=None,
                                 end_id=data_input.tokenizer._token_sep_id,
                                 maxlen=30
                                 )

    def on_epoch_end(self, epoch, logs=None):
        for i in range(2):
            sample = next(self.test_iter)
            res = self.response.generate(sample)
            logger.info('==============')
            logger.info('Context: {}'.format(sample['history']))
            logger.info('Goal: {}'.format(sample['goal']))
            logger.info('Answer: {}\n'.format(res))
            for j in range(7):
                # 很多重复的
                next(self.test_iter)

    def on_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % 20 == 0:
            logger.info('step: {}  loss: {} '.format(self._step, logs['loss']))


checkpoint_callback = keras.callbacks.ModelCheckpoint(
    save_path, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='min', period=3)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=join(save_dir, 'tf_logs'), histogram_freq=0, write_graph=False,
    write_grads=False, update_freq=320)

model_cls.model.fit_generator(
    data_input.generator(
        batch_size=batch_size,
        data_type=train_list,
        need_shuffle=True,
        cycle=True
    ),
    validation_data=data_input.generator(
        batch_size=batch_size,
        data_type=1,
        need_shuffle=True,
        cycle=True
    ),
    validation_steps=10,
    validation_freq=1,
    steps_per_epoch=steps_per_epoch,
    epochs=epoches,
    initial_epoch=init_epoch,
    verbose=2,
    class_weight=None,
    callbacks=[
        checkpoint_callback,
        tensorboard_callback,
        LogRecord()
    ]
)
