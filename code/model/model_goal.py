#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/8 21:56
@File      :goal_predict.py
@Desc      :
"""

from bert4keras_5_8.models import build_transformer_model
from bert4keras_5_8.backend import keras, K, tf, search_layer
from bert4keras_5_8.optimizers import Adam, extend_with_gradient_accumulation
from bert4keras_5_8.layers import Lambda, Dense
from utils.snippet import adversarial_training
import numpy as np
from cfg import *


class BertGoal(object):
    def __init__(self, keep_tokens, num_classes, load_path=None):
        keras.backend.clear_session()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=gpu_config))
        self.session = keras.backend.get_session()

        need_load = False
        if load_path and os.path.exists(load_path):
            need_load = True
        bert = build_transformer_model(
            config_path=join(BERT_PATH, 'bert_config.json'),
            checkpoint_path=None if need_load else join(BERT_PATH, 'bert_model.ckpt'),
            return_keras_model=False,
            keep_tokens=keep_tokens,
        )
        output = Lambda(lambda x: x[:, 0])(bert.model.output)
        output = Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)

        self.model =  keras.models.Model(bert.model.input, output)
        # self.model.summary()

        if need_load:
            logger.info('=' * 15 + 'Load from checkpoint: {}'.format(load_path))
            self.model.load_weights(load_path)
        self.data_deal = None

    def predict(self, sample:dict):
        if self.data_deal is None:
            from data_deal.input_goal import GoalInput
            self.data_deal = GoalInput()
        x, s = self.data_deal.encode_predict(sample)
        res = self.model.predict([[x], [s]])[0]
        return self.data_deal.reader.all_goals[np.argmax(res)]

    def compile(self):
        opt = extend_with_gradient_accumulation(Adam)(learning_rate=0.000015, grad_accum_steps=2)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['sparse_categorical_accuracy'],
        )
        adversarial_training(self.model, 'Embedding-Token', 0.5)