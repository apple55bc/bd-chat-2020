#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/8 22:20
@File      :train_goal.py
@Desc      :
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.model_goal import BertGoal
from data_deal.input_goal import GoalInput
from bert4keras_5_8.backend import keras
import argparse
import numpy as np

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

save_dir = join(MODEL_PATH, 'Goal_' + TAG)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
save_path = join(save_dir, 'trained.h5')

data_input = GoalInput()

model_cls = BertGoal(data_input.keep_tokens, num_classes=len(data_input.reader.all_goals), load_path=save_path)
model_cls.compile()


class LogRecord(keras.callbacks.Callback):
    def __init__(self):
        super(LogRecord, self).__init__()
        self._step = 1
        self.lowest = 1e10
        self.test_iter = data_input.generator(
            batch_size=2,
            data_type=3,
            need_shuffle=False,
            cycle=True
        )

    def on_epoch_end(self, epoch, logs=None):
        X, L = next(self.test_iter)
        T = X[0]
        for i in range(len(T)):
            res = model_cls.model.predict(X)
            logger.info('==============')
            logger.info('Context: {}'.format(data_input.tokenizer.decode(T[i])))
            logger.info('Goal: {}  {}'.format(L[i], data_input.reader.all_goals[L[i]]))
            logger.info('Answer: {}  {}\n'.format(np.argmax(res[i]),
                                              data_input.reader.all_goals[np.argmax(res[i])]))

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
