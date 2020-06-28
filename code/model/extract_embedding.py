#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/18 20:51
@File      :extract_embedding.py
@Desc      :
"""
from bert4keras_5_8.models import build_transformer_model
from bert4keras_5_8.backend import keras, tf
from bert4keras_5_8.tokenizers import Tokenizer
from bert4keras_5_8.snippets import sequence_padding
from cfg import *


class BertEmb(object):
    def __init__(self):
        keras.backend.clear_session()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=gpu_config))
        self.session = keras.backend.get_session()

        self.tokenizer = Tokenizer(join(BERT_PATH, 'vocab.txt'), do_lower_case=True)  # 建立分词器
        self.model = build_transformer_model(
            join(BERT_PATH, 'bert_config.json'),
            join(BERT_PATH, 'bert_model.ckpt'),
        )

    def get_embedding(self, sentences):
        X, S = [], []
        for sentence in sentences:
            token_ids, segment_ids = self.tokenizer.encode(sentence)
            X.append(token_ids)
            S.append(segment_ids)
        X = sequence_padding(X)
        S = sequence_padding(S)
        with self.session.graph.as_default():
            with self.session.as_default():
                result = self.model.predict([X, S])
        return result