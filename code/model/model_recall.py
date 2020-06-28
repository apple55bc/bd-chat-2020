#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/18 19:42
@File      :model_recall.py
@Desc      :
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
import numpy as np
from utils.sif import Sentence2Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import joblib
from utils.snippet import normalization
from model.extract_embedding import BertEmb
from data_deal.base_input import BaseInput
from annoy import AnnoyIndex
import json
import re

recall_path = join(MODEL_PATH, 'recall')
if not os.path.isdir(recall_path):
    os.makedirs(recall_path)


class RC_CFG(object):
    def __init__(self):
        self.max_seq_len = 128
        self.emd_dim = 768
        self.pca_dim = 188


recall_config = RC_CFG()


def train_step_1():
    model_emb = BertEmb()
    az_comp = re.compile('[a-zA-Z0-9]+')
    num_comp = re.compile('[0-9]')
    start_num_comp = re.compile('\[\d\]')

    data_input = BaseInput(from_pre_trans=True)

    questions = []
    answers = []

    logger.info('calucate sentences ...')

    for sample in data_input.get_sample([0, 1], need_shuffle=False, cycle=False):
        # if len(questions) > 1000:
        #     break
        context, turns = sample['context'], sample['turns']
        if turns is None:
            continue
        for i, turn in enumerate(turns):
            if i == 0:
                continue
            if turn:
                ans = re.sub(start_num_comp, '', context[i])
                q = re.sub(start_num_comp, '', context[i - 1])
                if len(num_comp.findall(ans)) > 0:  # 包含数字的回复全部丢弃
                    continue
                questions.append(re.sub(az_comp, '', q))
                answers.append(ans)

    print(f'questions: {questions[:2]}')
    print(f'answers: {answers[:2]}')
    print(f'len: {len(questions)}')
    logger.info('split sentences ...')
    splited_sentences = []
    for doc in questions[:1000000]:
        splited_sentences.append(list(doc))

    logger.info('train gensim ...')
    word_model = Word2Vec(splited_sentences, min_count=1, size=recall_config.emd_dim, iter=0)
    sif_model = Sentence2Vec(word_model, max_seq_len=recall_config.max_seq_len, components=2)
    logger.info('gensim train done .')
    del splited_sentences, word_model

    logger.info('get vecotrs and train pc...')

    # Memory will explode, rewrite the logic here
    sentence_vectors = []
    vec_batch = 10000
    pca = PCA(n_components=recall_config.pca_dim, whiten=True, random_state=2112)

    pca_n = min(300000, len(questions))
    has_pca_trained = False

    for b_i, e_i in zip(range(0, len(questions), vec_batch), range(vec_batch, len(questions) + vec_batch, vec_batch)):
        sentences_out = model_emb.get_embedding(questions[b_i:e_i])
        splited_sentences = []
        for doc in questions[b_i:e_i]:
            splited_sentences.append(list(doc))
        sentences_out = sif_model.cal_output(splited_sentences, sentences_out)
        if e_i >= pca_n:
            if has_pca_trained:
                sentence_vectors.extend(normalization(pca.transform(sentences_out)))
            else:
                logger.info('Train PCA ... pca_n num: {}'.format(pca_n))
                sentence_vectors.extend(sentences_out)
                pca.fit(np.stack(sentence_vectors[:pca_n]))
                sentence_vectors = list(normalization(pca.transform(np.stack(sentence_vectors))))
                has_pca_trained = True
        else:
            sentence_vectors.extend(sentences_out)
        del sentences_out, splited_sentences
        logger.info('  complete one batch. batch_size: {}  percent {:.2f}%'.format(
            vec_batch, (100 * min(len(questions), e_i) / len(questions))))

    sentence_vectors = np.stack(sentence_vectors)

    sentences_emb = sif_model.train_pc(sentence_vectors)
    print(sentences_emb.shape)
    logger.info('train pc over.')

    logger.info('save model')
    joblib.dump(sif_model, os.path.join(recall_path, 'bert_sif.sif'))
    joblib.dump(pca, os.path.join(recall_path, 'bert_pca.pc'))
    json.dump(answers, open(join(recall_path, 'answers.json'), mode='w', encoding='utf-8'),
              ensure_ascii=False, indent=4, separators=(',', ':'))
    np.save(join(recall_path, 'sentences_emb'), sentences_emb)


def train_step_2():
    logger.info('train_step_2  ...')
    final_q_embs = np.load(join(recall_path, 'sentences_emb.npy'))

    annoy_model = AnnoyIndex(recall_config.pca_dim, metric='angular')
    logger.info('add annoy...')
    for i, emb in enumerate(final_q_embs):
        annoy_model.add_item(i, emb)
    logger.info('build annoy...')
    annoy_model.build(88)
    annoy_model.save(join(recall_path, 'annoy.an'))
    logger.info('build over...')


class SearchEMb:
    def __init__(self, top_n=3):
        self.model_emb = BertEmb()
        self.sif = joblib.load(join(recall_path, 'bert_sif.sif'))
        self.pca = joblib.load(join(recall_path, 'bert_pca.pc'))
        self.answers = json.load(open(join(recall_path, 'answers.json'), encoding='utf-8'))
        self.annoy = AnnoyIndex(recall_config.pca_dim, metric='angular')
        self.annoy.load(join(recall_path, 'annoy.an'))

        self.az_comp = re.compile('[a-zA-Z0-9]+')
        self.start_num_comp = re.compile('\[\d\]')
        self.top_n = top_n

    def get_recall(self, sentence):
        sentence = re.sub(self.start_num_comp, '', sentence)
        sentence = re.sub(self.az_comp, '', sentence)
        res_indexs, distances = self.annoy.get_nns_by_vector(self.get_emb(sentence), self.top_n, include_distances=True)
        results = []
        for idx in res_indexs:
            results.append(self.answers[idx])
        return results, distances

    def get_emb(self, sentence):
        vectors = self.model_emb.get_embedding([sentence])
        mid_vectors = self.sif.cal_output([list(sentence)], vectors)
        mid_vectors = normalization(self.pca.transform(mid_vectors))
        return self.sif.predict_pc(mid_vectors)[0]


if __name__ == '__main__':
    train_step_1()
    train_step_2()
