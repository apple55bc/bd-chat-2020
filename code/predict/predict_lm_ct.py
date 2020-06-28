#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/19 21:55
@File      :predict_lm_ct.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.bert_lm import BertLM, Response
from data_deal.base_input import BaseInput
from data_deal.input_ct import CTInput
from data_deal.trans_output import TransOutput
from model.model_context import ModelContext
from model.model_recall import SearchEMb
import jieba
import time
import numpy as np
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int,
                    help=r'default is 2',
                    default=2)
args = parser.parse_args(sys.argv[1:])
data_type = args.type
save_dir = join(MODEL_PATH, 'BertLM_' + TAG)
save_path = join(save_dir, 'trained.h5')
if not os.path.isdir(OUT_PATH):
    os.makedirs(OUT_PATH)
output_dir = join(OUT_PATH, 'out_{}_{}_{}.txt'.format(
    data_type, TAG, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

data_input = BaseInput(from_pre_trans=True)
model_cls = BertLM(data_input.keep_tokens, load_path=save_path)
response = Response(model_cls.model,
                    model_cls.session,
                    data_input,
                    start_id=None,
                    end_id=data_input.tokenizer._token_sep_id,
                    maxlen=40
                    )
goal_response = Response(model_cls.model,
                         model_cls.session,
                         data_input,
                         start_id=None,
                         end_id=data_input.tokenizer._token_goal_id,
                         maxlen=10
                         )
out_trans = TransOutput(rc_tag='')
search_rc = SearchEMb(top_n=3)

ct_dir = join(MODEL_PATH, 'CT_' + TAG)
ct_path = join(ct_dir, 'trained.h5')
ct_input = CTInput(from_pre_trans=False)
model_ct_cls = ModelContext(ct_input.keep_tokens, load_path=ct_path)
del ct_input

test_iter = data_input.get_sample(
    data_type,
    need_shuffle=False,
    cycle=False
)


def cal_participle(samp: dict):
    words = []
    words.extend(samp['situation'].split(' '))
    words.extend(samp['goal'].split(' '))
    for k, v in samp['user_profile'].items():
        if not isinstance(v, list):
            v = [v]
        for _v in v:
            words.extend(_v.split(' '))
    for kg in samp['knowledge']:
        words.extend(kg[2].split(' '))
    words = set(words)
    words = [w for w in words if len(w) > 1]
    return words


with open(output_dir, encoding='utf-8', mode='w') as fw:
    skip = 1374
    i = 0
    for sample in test_iter:
        i += 1
        # if i <= skip:
        #     continue
        samp_words = cal_participle(sample)
        for w in samp_words:
            jieba.add_word(w)

        goals = goal_response.goal_generate(sample, n=4)
        goals = list(set(goals))
        history = sample['history']
        final_answers = []
        turn = 0
        while len(final_answers) <= 0:
            answer_res = response.generate(sample, goals=goals, random=True)
            score_mul = [1] * len(answer_res)
            if (len(history) > 1 and len(history[-1]) > 4 and '新闻' not in ''.join(history[-2:])) or turn > 0:
                rc_ans, rc_dis = search_rc.get_recall(history[-1])
                answer_res.extend(rc_ans)
                score_mul = score_mul + np.minimum((1.0 - np.array(rc_dis)) * 0.5 + 1.0, 0.99).tolist()
            # 去重 转换
            mid_res_clean = []
            mid_sc = []
            for ans, sc in zip(answer_res, score_mul):
                sentence = re.sub(data_input.reader.goal_num_comp, '', ans)
                if sentence in mid_res_clean:
                    continue
                trans_answer, tag = out_trans.trans_output(sample, ans)
                if tag:
                    continue
                final_answers.append(trans_answer)
                mid_sc.append(sc)
                mid_res_clean.append(sentence)
            score_mul = mid_sc
            turn += 1
            if turn > 5:
                final_answers = ['是的呢']
                score_mul = [1.0]
                logger.warning('No proper answer! \n{}'.format(history))
        # CT score
        final_contexts = [history + [ans] for ans in final_answers]
        scores = model_ct_cls.predict(final_contexts)
        scores_md = np.multiply(scores, np.array(score_mul))
        answer = final_answers[np.argmax(scores_md)]

        e_i = 0
        if answer[0] == '[':
            for j in range(1, 4):
                if answer[j] == ']':
                    e_i = j + 1
                    break
        answer = answer[:e_i] + ' ' + ' '.join(jieba.lcut(answer[e_i:]))
        fw.writelines(answer + '\n')
        if i % 37 == 0:
            print('\rnum: {}    '.format(i), end='')
    print('\n=====> Over: ', i)
