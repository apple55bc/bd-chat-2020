#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/18 21:01
@File      :model_context.py
@Desc      :
"""
from bert4keras_5_8.backend import keras, search_layer, K, tf
from bert4keras_5_8.models import build_transformer_model
from bert4keras_5_8.optimizers import Adam
from bert4keras_5_8.layers import Lambda, Dense, Input
from bert4keras_5_8.snippets import sequence_padding
from utils.snippet import adversarial_training
import re
from cfg import *


class ModelContext(object):
    def __init__(self, keep_tokens, load_path=None):
        keras.backend.clear_session()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=gpu_config))
        self.session = keras.backend.get_session()

        need_load = False
        if load_path and os.path.exists(load_path):
            need_load = True

        bert = build_transformer_model(
            join(BERT_PATH, 'bert_config.json'),
            None if need_load else join(BERT_PATH, 'bert_model.ckpt'),
            keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
            return_keras_model=False,
        )

        layers_out_lambda = Lambda(lambda x: x[:, 0])
        layers_out_dense = Dense(units=2,
                                 activation='softmax',
                                 kernel_initializer=bert.initializer)

        output = layers_out_lambda(bert.model.output)
        output = layers_out_dense(output)

        self.model = keras.models.Model(bert.model.input, output, name='Final-Model')
        if need_load:
            logger.info('=' * 15 + 'Load from checkpoint: {}'.format(load_path))
            self.model.load_weights(load_path)
        self.data_deal = None

    def compile(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(2e-5),
            metrics=['sparse_categorical_accuracy']
        )
        adversarial_training(self.model, 'Embedding-Token', 0.3)

    def predict(self, contexts):
        if self.data_deal is None:
            from data_deal.input_ct import CTInput
            self.data_deal = CTInput(from_pre_trans=False)
        X, S = [], []
        for context in contexts:
            context = [re.sub(self.data_deal.reader.goal_num_comp, '', s).replace(' ', '') for s in context]
            x, s, l = self.data_deal.encode(ori_context=context)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        with self.session.graph.as_default():
            with self.session.as_default():
                R = self.model.predict([X, S])
        return R[:, 1]