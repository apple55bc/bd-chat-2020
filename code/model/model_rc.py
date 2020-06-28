#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/5/10 9:58
@Author  : Apple QXTD
@File    : model_rc.py
@Desc:   :
"""
from cfg import *
import re
import numpy as np
from bert4keras_5_8.backend import keras, K, tf, search_layer
from bert4keras_5_8.models import build_transformer_model
from bert4keras_5_8.snippets import sequence_padding
from keras.layers import Lambda
from keras.models import Model
from data_deal.input_rc import RCInput
from bert4keras_5_8.optimizers import Adam, extend_with_gradient_accumulation


class BertCL:
    def __init__(self, tag='d', is_predict=False, load_path=None):
        self.save_path = join(MODEL_PATH, 'rc_' + tag, 'trained.h5')

        keras.backend.clear_session()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=gpu_config)
        keras.backend.set_session(self.session)

        self.data_deal = RCInput()
        self.config_path = join(BERT_PATH, 'bert_config.json')
        self.checkpoint_path = join(BERT_PATH, 'bert_model.ckpt')

        self.tokenizer = self.data_deal.tokenizer
        self.max_p_len = self.data_deal.max_p_len
        self.max_q_len = self.data_deal.max_q_len
        self.max_a_len = self.data_deal.max_a_len
        self.batch_size = self.data_deal.batch_size

        model = build_transformer_model(
            self.config_path,
            None if is_predict else self.checkpoint_path,
            model='bert',
            with_mlm=True,
            keep_tokens=self.data_deal.keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        )
        output = Lambda(lambda x: x[:, 1:self.max_a_len + 1])(model.output)
        self.model = Model(model.input, output)
        # self.model.summary()
        if load_path:
            logger.info('Load from init checkpoint {} .'.format(load_path))
            self.model.load_weights(load_path)
        elif os.path.exists(self.save_path):
            logger.info('Load from init checkpoint {} .'.format(self.save_path))
            self.model.load_weights(self.save_path)

    def predict(self, question, contexts, return_items=False, in_passage=True):
        if isinstance(contexts, str):
            contexts = [contexts]
        passages = []
        if len(question) == 0:
            return None
        for context in contexts:
            add = True
            while add:
                # 每间隔200进行拆分
                passages.append(context[:self.max_p_len])
                if len(context) <= self.max_p_len:
                    add = False
                else:
                    context = context[200:]
        if len(passages) == 0:
            return None
        answer = self.gen_answer(question, passages, in_passage=in_passage)
        answer = self.max_in_dict(answer)
        if not return_items:
            if answer is None:
                answer = ''
            else:
                answer = answer[0][0]
        return answer

    @staticmethod
    def get_ngram_set(x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    def gen_answer(self, question, passages, in_passage=True):
        """由于是MLM模型，所以可以直接argmax解码。
        """
        all_p_token_ids, token_ids, segment_ids = [], [], []

        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids, _ = self.tokenizer.encode(passage, max_length=self.max_p_len + 1)
            q_token_ids, _ = self.tokenizer.encode(question, max_length=self.max_q_len + 1)
            all_p_token_ids.append(p_token_ids[1:])
            token_ids.append([self.tokenizer._token_cls_id])
            token_ids[-1] += ([self.tokenizer._token_mask_id] * self.max_a_len)
            token_ids[-1] += [self.tokenizer._token_sep_id]
            token_ids[-1] += (q_token_ids[1:] + p_token_ids[1:])
            segment_ids.append([0] * len(token_ids[-1]))

        token_ids = sequence_padding(token_ids)
        segment_ids = sequence_padding(segment_ids)
        with self.session.graph.as_default():
            with self.session.as_default():
                probas = self.model.predict([token_ids, segment_ids], batch_size=3)
        results = {}
        for t, p in zip(all_p_token_ids, probas):
            a, score = tuple(), 0.
            for i in range(self.max_a_len):
                # pi是将passage以外的token的概率置零
                if in_passage:
                    idxs = list(self.get_ngram_set(t, i + 1)[a])
                    if self.tokenizer._token_sep_id not in idxs:
                        idxs.append(self.tokenizer._token_sep_id)
                    pi = np.zeros_like(p[i])
                    pi[idxs] = p[i, idxs]
                else:
                    pi = p[i]
                a = a + (pi.argmax(),)
                score += pi.max()
                if a[-1] == self.tokenizer._token_sep_id:
                    break
            score = score / (i + 1)
            a = self.tokenizer.decode(a)
            if a:
                results[a] = results.get(a, []) + [score]
        results = {
            k: (np.array(v) ** 2).sum() / (sum(v) + 1)
            for k, v in results.items()
        }
        return results

    @staticmethod
    def max_in_dict(d):
        if d:
            return sorted(d.items(), key=lambda s: -s[1])

    def compile(self):

        def masked_cross_entropy(y_true, y_pred):
            """交叉熵作为loss，并mask掉padding部分的预测
            """
            y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
            y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
            cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
            cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
            return cross_entropy

        opt = extend_with_gradient_accumulation(Adam, name='accum')(grad_accum_steps=3, learning_rate=3e-5)
        self.model.compile(loss=masked_cross_entropy, optimizer=opt)


def test():
    m = BertLM()
    m.compile()


if __name__ == '__main__':
    test()