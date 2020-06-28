#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/5/5 14:12
@Author  : Apple QXTD
@File    : check_predict.py
@Desc:   :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.bert_lm import BertLM, Response
from data_deal.base_input import BaseInput
from data_deal.trans_output import TransOutput
import jieba
# from model.model_goal import BertGoal


tag = TAG
# tag = 'd4-6ep-ng'
save_dir = join(MODEL_PATH, 'BertLM_' + tag)
save_path = join(save_dir, 'trained.h5')
data_input = BaseInput()
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

goal_dir = join(MODEL_PATH, 'Goal_' + tag)
goal_path = join(goal_dir, 'trained.h5')
# goal_cls = BertGoal(data_input.keep_tokens, num_classes=len(data_input.reader.all_goals), load_path=goal_path)


test_iter = data_input.get_sample(
            2,
            need_shuffle=False,
            cycle=False
        )


def cal_participle(samp:dict):
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


skip = 28
i = 0
last_sample = None

for sample in test_iter:
    i += 1
    if i <= 1:
        last_sample = sample
        continue
    if i <= skip:
        last_sample = sample
        continue

    samp_words = cal_participle(sample)
    for w in samp_words:
        jieba.add_word(w)
    goals = goal_response.goal_generate(last_sample, n=4)
    goals = list(set(goals))
    # goals = [goal_cls.predict(last_sample)]
    answer_res = response.generate(last_sample, goals=goals)
    answer, tag = out_trans.trans_output(last_sample, answer_res)
    if tag:
        answer_res = response.generate(last_sample, goals=goals, random=True)
        for res in answer_res:
            answer, tag = out_trans.trans_output(last_sample, res)
            if not tag:
                break
        if tag:
            answer_res = response.generate(last_sample, goals=goals, force_goal=True, random=True)
            for res in answer_res:
                answer, tag = out_trans.trans_output(last_sample, res)
                if not tag:
                    break
    e_i = 0
    if answer[0] == '[':
        for j in range(1, min(4, len(answer))):
            if answer[j] == ']':
                e_i = j + 1
                break
    answer = answer[:e_i] + ' ' + ' '.join(jieba.lcut(answer[e_i:]))
    last_sample = sample

print('\n=====> Over: ', i)