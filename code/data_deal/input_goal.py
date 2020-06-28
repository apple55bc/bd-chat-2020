#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/8 21:08
@File      :goal_input.py
@Desc      :
"""
from data_deal.base_input import *


class GoalInput(BaseInput):
    def __init__(self):
        super(GoalInput, self).__init__()
        self.max_len = 360

    def encode_predict(self, sample: dict):
        context, goals, turns = self.reader.trans_sample(sample, need_bot_trans=False)
        if context is None:
            return None, None
        token_ids = []
        segs = []

        token_ids.extend(self.tokenizer.encode(sample['situation'])[0])
        segs.extend([0] * len(token_ids))
        turn = False
        for i, sentence, goal, turn in zip(list(range(len(context))), context, goals, turns):
            this_goal = self.reader.all_goals[goal if goal > 0 else 1]
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]

            segs += [0 if turn else 1] * (len(goal_tokens) + 1)

            sen_tokens, _ = self.tokenizer.encode(sentence)
            sen_tokens = sen_tokens[1:]
            token_ids += sen_tokens
            if turn:
                segs += [1] * len(sen_tokens)
            else:
                segs += [0] * len(sen_tokens)
        if turn:
            raise ValueError('last turn is not user')
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:1] + token_ids[1 - self.max_len:]
            segs = segs[:1] + segs[1 - self.max_len:]
        return token_ids, segs

    def encode(self, sample: dict, need_goal_mask=True):
        context, goals, turns = self.reader.trans_sample(sample, need_bot_trans=False)
        if context is None:
            return None, None
        token_ids = []
        segs = []

        token_ids.extend(self.tokenizer.encode(sample['situation'])[0])
        segs.extend([0] * len(token_ids))
        for i, sentence, goal, turn in zip(list(range(len(context))), context, goals, turns):
            if turn and goal != 0:  # 未知只有test才有
                if len(token_ids) > self.max_len:
                    token_ids = token_ids[:1] + token_ids[1 - self.max_len:]
                    segs = segs[:1] + segs[1 - self.max_len:]
                yield token_ids.copy(), segs.copy(), goal
            if need_goal_mask and i > 0:
                if goal > 1 and random.random() < 0.5:
                    if 'history' not in sample.keys():
                        goal = 1
            this_goal = self.reader.all_goals[goal if goal > 0 else 1]
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]

            segs += [0 if turn else 1] * (len(goal_tokens) + 1)

            sen_tokens, _ = self.tokenizer.encode(sentence)
            sen_tokens = sen_tokens[1:]
            token_ids += sen_tokens
            if turn:
                segs += [1] * len(sen_tokens)
            else:
                segs += [0] * len(sen_tokens)

    def generator(self, batch_size=12, data_type=0, need_shuffle=False, cycle=False):
        data_dict = {
            0: join(DATA_PATH, 'train/train.txt'),
            1: join(DATA_PATH, 'dev/dev.txt'),
            2: join(DATA_PATH, 'test_1/test_1.txt'),
            3: join(DATA_PATH, 'test_2/test_2.txt'),
        }
        if not isinstance(data_type, list):
            data_type = [data_type]
        data_files = []
        for t in data_type:
            if t not in data_dict.keys():
                raise ValueError('data_type {} not in dict: {}'.format(t, data_dict.keys()))
            data_files.append(data_dict[t])
        X, S, L = [], [], []
        sample_iter = self.get_sample(data_files, need_shuffle=need_shuffle, cycle=cycle)
        while True:
            sample = next(sample_iter)
            piece_iter = self.encode(sample)
            for x, s, l in piece_iter:
                if x is None:
                    continue
                X.append(x)
                S.append(s)
                L.append(l)
                if len(X) >= batch_size:
                    X = sequence_padding(X)
                    S = sequence_padding(S)
                    yield [X, S], L
                    X, S, L = [], [], []
