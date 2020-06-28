#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/19 22:28
@File      :train_ct.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.model_context import ModelContext
from data_deal.input_ct import CTInput
from bert4keras_5_8.backend import keras
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=0)
parser.add_argument('--epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=6)
args = parser.parse_args(sys.argv[1:])

batch_size = 4
steps_per_epoch = 600

epoches = int(args.epoch * totle_sample / batch_size / steps_per_epoch * 4.67)
init_epoch = args.init_epoch

save_dir = join(MODEL_PATH, 'CT_' + TAG)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
save_path = join(save_dir, 'trained.h5')

data_input = CTInput(from_pre_trans=False)

model_cls = ModelContext(data_input.keep_tokens, load_path=save_path)
model_cls.compile()


class LogRecord(keras.callbacks.Callback):
    def __init__(self):
        super(LogRecord, self).__init__()
        self._step = 1
        self.lowest = 1e10
        self.test_iter = data_input.generator(
            4,
            data_type=3,
            need_shuffle=False,
            cycle=True,
            need_douban=False
        )

    def on_epoch_end(self, epoch, logs=None):
        [X, S], L = next(self.test_iter)
        result = model_cls.model.predict([X, S])
        for x, l, r in zip(X, L, result):
            print(' '.join(data_input.tokenizer.ids_to_tokens(x)).rstrip(' [PAD]'))
            print('label: {}   predict: {}\n'.format(l, np.argmax(r)))

    def on_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % 60 == 0:
            logger.info('step: {}  loss: {} '.format(self._step, logs['loss']))


checkpoint_callback = keras.callbacks.ModelCheckpoint(
    save_path, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='min', period=2)
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
