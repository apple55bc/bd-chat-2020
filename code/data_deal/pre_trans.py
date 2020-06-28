#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/9 19:46
@File      :pre_trans.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_deal.base_input import *


def trans(data_type):
    data_dict = {
        0: join(DATA_PATH, 'train/train.txt'),
        1: join(DATA_PATH, 'dev/dev.txt'),
        2: join(DATA_PATH, 'test_1/test_1.txt'),
        3: join(DATA_PATH, 'test_2/test_2.txt'),
    }
    output_dir = join(DATA_PATH, 'trans')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = join(output_dir, 'trans_{}.txt'.format(data_type))

    data_input = BaseInput()

    all_data = []

    data_iter = data_input.get_sample(data_dict[data_type], need_shuffle=False, cycle=False)
    sn = 0
    for sample in data_iter:
        context, goals, turns, unused_goals, replace_dicts = data_input.reader.trans_sample(
            sample, return_rest_goals=True, need_replace_dict=True)
        sample.update(
            {
                'context': context,
                'goals': goals,
                'turns': turns,
                'unused_goals': unused_goals,
                'replace_dicts': replace_dicts,
            }
        )
        all_data.append(sample)
        sn += 1
        if sn % 58 == 0:
            print('\rnum {}'.format(sn), end='  ')
        # if sn > 30:
        #     break
    print('\nOver: ', sn)
    with open(output_path, encoding='utf-8', mode='w') as fw:
        for data in all_data:
            fw.writelines(json.dumps(
                data,
                ensure_ascii=False,
                # indent=4, separators=(',',':')
            ) + '\n')

def trans_v2(data_type):
    output_dir = join(DATA_PATH, 'trans')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    input_path = join(output_dir, 'trans_{}.txt'.format(data_type))
    output_path = join(output_dir, 'trans_{}_trim.txt'.format(data_type))

    data_input = BaseInput()

    all_data = []

    data_iter = data_input.get_sample(input_path, need_shuffle=False, cycle=False)
    sn = 0
    change = 0
    for sample in data_iter:
        turns = sample['turns']
        if not (turns is None or len(turns) == 0):
            if not turns[0]:
                turns = turns[1:]
            if len(turns) > sum(turns) * 2:
                context, goals, turns, unused_goals, replace_dicts = data_input.reader.trans_sample(
                    sample, return_rest_goals=True, need_replace_dict=True)
                sample.update(
                    {
                        'context': context,
                        'goals': goals,
                        'turns': turns,
                        'unused_goals': unused_goals,
                        'replace_dicts': replace_dicts,
                    }
                )
                change += 1
        all_data.append(sample)
        sn += 1
        if sn % 58 == 0:
            print('\rnum {}  change {}'.format(sn, change), end='  ')
        # if sn > 30:
        #     break
    print('\nOver: ', sn)
    with open(output_path, encoding='utf-8', mode='w') as fw:
        for data in all_data:
            fw.writelines(json.dumps(
                data,
                ensure_ascii=False,
                # indent=4, separators=(',',':')
            ) + '\n')


if __name__ == '__main__':
    trans(0)
    trans(1)
    trans(2)
    trans(3)