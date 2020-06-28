#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/5/10 11:10
@Author  : Apple QXTD
@File    : train_rc.py
@Desc:   :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfg import *
from bert4keras_5_8.backend import keras
from data_deal.input_rc import RCInput
from model.model_rc import BertCL

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=0)
parser.add_argument('--epoch', type=int,
                    help=r'init epoch, you don\'t know ?',
                    default=3)
args = parser.parse_args(sys.argv[1:])


steps_per_epoch = 120
batch_size = 3

epoches = int(args.epoch * totle_sample / batch_size / steps_per_epoch)

save_dir = join(MODEL_PATH, 'rc_' + TAG)
save_path = join(save_dir, 'trained.h5')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

data_input = RCInput(from_pre_trans=True)
model_cls = BertCL(tag=TAG)
model_cls.compile()

tokenizer = data_input.tokenizer
max_p_len = data_input.max_p_len
max_q_len = data_input.max_q_len
max_a_len = data_input.max_a_len


class Evaluate(keras.callbacks.Callback):
    def __init__(self, ):
        super(Evaluate, self).__init__()
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        self.eva_iter = data_input.get_sample(data_files=3, cycle=True)
        self._step = 0

    def on_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % 100 == 0:
            logger.info('    step: {}  loss: {} '.format(self._step, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        sample = next(self.eva_iter)
        samples = data_input.get_rc_sample(sample)
        for sample in samples:
            for q_key, answer in sample['result'].items():
                if q_key not in sample['replace_dict'].keys():
                    continue
                context = sample['replace_dict'][q_key]  # it's a list
                context = '|'.join(context).replace(' ', '')  # 全部使用 | 作为分隔符
                question = '|'.join(sample['history']).replace(' ', '')  # 全部使用 | 作为分隔符
                answer = answer.replace(' ', '')
                question += '|{}'.format(q_key)  # question额外添加询问的标记
                predict_answer = model_cls.predict(question, context)

                logger.info('Context: {}'.format(context))
                logger.info('Question: {}'.format(question))
                logger.info('Answer: {}'.format(answer))
                logger.info('Gen Answer: {}'.format(predict_answer))
                logger.info('')


if __name__ == '__main__':
    evaluator = Evaluate()

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        save_path, monitor='loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='min', period=1)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=join(save_dir, 'tf_logs'), histogram_freq=0, write_graph=False,
        write_grads=False, update_freq=160)
    early_stop_callback = keras.callbacks.EarlyStopping()
    model_cls.model.fit_generator(
        data_input.generator(
            batch_size=batch_size,
            data_type=train_list,
            need_shuffle=True,
            cycle=True
        ),
        steps_per_epoch=steps_per_epoch,
        epochs=epoches,
        verbose=2,
        callbacks=[
            checkpoint_callback,
            tensorboard_callback,
            evaluator,
        ])
