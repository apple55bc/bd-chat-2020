#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/28 22:07
@File      :bert_lm.py
@Desc      :
"""
from bert4keras_5_8.models import build_transformer_model
from bert4keras_5_8.backend import keras, K, tf
from bert4keras_5_8.optimizers import Adam, extend_with_gradient_accumulation
from bert4keras_5_8.snippets import AutoRegressiveDecoder
from data_deal.base_input import BaseInput
import numpy as np
from cfg import *


class Response(AutoRegressiveDecoder):
    """基于随机采样的故事续写
    """

    def __init__(self, model, session, data_deal:BaseInput, *args, **kwargs):
        self.model = model
        self.data_deal = data_deal
        self.max_len = 512
        self.session = session
        super(Response, self).__init__(*args, **kwargs)

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids = inputs[0]
        segment_ids = inputs[1]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        if token_ids.shape[1] > self.max_len:
            token_ids = token_ids[:, -self.max_len:]
            segment_ids = segment_ids[:, -self.max_len:]
        with self.session.graph.as_default():
            with self.session.as_default():
                res = self.model.predict([token_ids, segment_ids])[:, -1]
        return res

    def generate(self, sample, goals=None, need_goal=True, force_goal=False, random=False):
        if goals is None:
            goals = []
        token_ids, segs, goal_index = self.data_deal.encode_predict_final(
            sample, goals, need_goal=need_goal, force_goal=force_goal, silent=False)
        if random:
            if token_ids is None:
                return []
            res = self.nucleus_sample([token_ids, segs], 3, topk=3)
            res = [self.data_deal.tokenizer.decode(r) for r in res]
        else:
            if token_ids is None:
                return ''
            res = self.beam_search([token_ids, segs], 1)
            res = self.data_deal.tokenizer.decode(res)
        # if goal_index:
        #     if isinstance(res, list):
        #         res = ['[{}]{}'.format(goal_index, s) for s in res]
        #     else:
        #         res = '[{}]{}'.format(goal_index, res)
        return res

    def check_goal_end(self, sample, end_id):
        token_ids, segs = self.data_deal.encode_predict_final(sample, cand_goals=[], need_goal=False)
        with self.session.graph.as_default():
            with self.session.as_default():
                score = self.model.predict([[token_ids], [segs]])[0, -1]
        m = np.argmax(score)
        if m != end_id:
            return True
        else:
            return False

    def goal_generate(self, sample, n=5):
        token_ids, segs, goal_index = self.data_deal.encode_predict_final(sample, cand_goals=[], need_goal=False)
        results = self.nucleus_sample([token_ids, segs], n=n, topk=20)
        return [self.data_deal.tokenizer.decode(res) for res in results]


class BertLM(object):
    def __init__(self, keep_tokens, load_path=None):
        keras.backend.clear_session()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=gpu_config))
        self.session = keras.backend.get_session()

        need_load = False
        if load_path and os.path.exists(load_path):
            need_load = True

        self.model = build_transformer_model(
            join(BERT_PATH, 'bert_config.json'),
            None if need_load else join(BERT_PATH, 'bert_model.ckpt'),
            application='lm',
            keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        )
        self.model.summary()

        if need_load:
            logger.info('=' * 15 + 'Load from checkpoint: {}'.format(load_path))
            self.model.load_weights(load_path)

    def compile(self):
        # 交叉熵作为loss，并mask掉输入部分的预测
        y_true = self.model.input[0][:, 1:]  # 目标tokens
        y_mask = self.model.input[1][:, 1:]  # 目标mask
        y_mask = K.cast(y_mask, K.floatx())  # 转为浮点型
        y_pred = self.model.output[:, :-1]  # 预测tokens，预测与目标错开一位
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
        self.model.add_loss(cross_entropy)
        opt = extend_with_gradient_accumulation(Adam)(learning_rate=0.000015, grad_accum_steps=2)
        self.model.compile(optimizer=opt)
