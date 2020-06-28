#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/29 21:41
@File      :predict_lm.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
from model.bert_lm import BertLM, Response
from data_deal.base_input import BaseInput
from data_deal.trans_output import TransOutput
# from model.model_goal import BertGoal
import jieba
import time

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
# out_trans = TransOutput(rc_tag=TAG)
out_trans = TransOutput(rc_tag='')
goal_dir = join(MODEL_PATH, 'Goal_' + TAG)
goal_path = join(goal_dir, 'trained.h5')
# goal_cls = BertGoal(data_input.keep_tokens, num_classes=len(data_input.reader.all_goals), load_path=goal_path)

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

        # has_goal = response.check_goal_end(sample, end_id=data_input.tokenizer._token_goal_id)
        # answer_res = response.generate(sample, has_goal=has_goal)
        goals = goal_response.goal_generate(sample, n=4)
        goals = list(set(goals))
        # goals = [goal_cls.predict(sample)]
        answer_res = response.generate(sample, goals=goals)
        answer, tag = out_trans.trans_output(sample, answer_res)
        if tag:
            answer_res = response.generate(sample, goals=goals, random=True)
            for res in answer_res:
                answer, tag = out_trans.trans_output(sample, res)
                if not tag:
                    break
            if tag:
                answer_res = response.generate(sample, goals=goals, force_goal=True, random=True)
                for res in answer_res:
                    answer, tag = out_trans.trans_output(sample, res)
                    if not tag:
                        break
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
