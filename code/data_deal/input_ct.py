#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/5/19 22:39
@File      :input_ct.py
@Desc      :
"""
from data_deal.base_input import *
import random


class CT_Tokenizer(Tokenizer):
    def truncate_sequence(self,
                          max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=-1):
        """截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(1)
            else:
                second_sequence.pop(pop_index)


class CTInput(BaseInput):
    def __init__(self, *args, **kwargs):
        super(CTInput, self).__init__(*args, **kwargs)

        self.last_sample_num = None

        # 模型相关
        self.token_dict, self.keep_tokens = load_vocab(
            join(BERT_PATH, 'vocab.txt'),
            startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[unused2]'],
            simplified=True, max_num=9000)
        logger.info('Len of token_dict:{}'.format(len(self.token_dict)))
        self.tokenizer = CT_Tokenizer(self.token_dict)
        self.batch_size = 8

        self._label_context = {
            '0': 0,
            '1': 1,
        }

    def generator(self, batch_size=4, data_type=0, need_shuffle=False, cycle=False, need_douban=True):
        if not isinstance(data_type, list):
            data_type = [data_type]
        data_files = []
        for t in data_type:
            if t not in self.data_dict.keys():
                raise ValueError('data_type {} not in dict: {}'.format(t, self.data_dict.keys()))
            data_files.append(self.data_dict[t])
        X, S, L = [], [], []
        sample_iter = self.get_sample(data_files, need_shuffle=need_shuffle, cycle=cycle)
        if need_douban:
            douban_iter = self._get_douban(join(DATA_PATH, 'douban_train.txt'), cycle=True)
        else:
            douban_iter = None
        info = True
        while True:
            if not need_douban or random.random() < 0.3:
                sample = next(sample_iter)
                bot_first = self.reader._check_bot_first(sample['goal'])
                if bot_first is None:
                    continue
                add_n = 1 if bot_first else 0
                context_str = 'conversation' if 'conversation' in sample.keys() else 'history'
                if len(sample[context_str]) < 2:
                    continue
                end_n = random.randint(2, len(sample[context_str]))
                if end_n % 2 != add_n:
                    if end_n == 2:
                        end_n = 3
                    else:
                        end_n -= 1
                # label
                context = sample[context_str][:end_n]
                label = '1'
                if random.random() < 0.5:
                    if end_n <= len(sample[context_str]) - 2:
                        context.pop(-1)
                        context.append(sample[context_str][end_n + 1])
                        label = '0'
                context = [re.sub(self.reader.goal_num_comp, '', s).replace(' ', '') for s in context]
            else:
                sample = next(douban_iter)
                context = sample[1:]
                label = sample[0]
            x, s, l = self.encode(context, label)
            if info:
                logger.info('input: {}'.format(' '.join(self.tokenizer.ids_to_tokens(x))))
                info = False
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

    def encode(self, ori_context, label=None):
        if label:
            label = self._label_context[label]
        ori_context = list(map(lambda _s: str(_s).strip().replace(' ', ''), ori_context))
        context = []
        for i, sentence in enumerate(ori_context[:-1]):
            context.extend(self.tokenizer.tokenize(sentence, add_cls=(i == 0), add_sep=(i >= len(ori_context) - 2)))
            if i < len(ori_context) - 2:
                context.append('[unused2]')
        x, s = self.tokenizer.encode(first_text=context,
                                     second_text=ori_context[-1], max_length=128)
        return x, s, label

    def _get_douban(self, file_path, cycle=True):
        with open(file_path, mode='r', encoding='utf-8') as fr:
            while True:
                line = fr.readline()
                _rn = 0
                while not line:
                    line = fr.readline()
                    if _rn > 10:
                        if cycle:
                            fr.seek(0)
                        else:
                            raise StopIteration
                    _rn += 1
                line = line.strip().split('\t')
                yield line


def test():
    rc = CTInput()
    rc_it = rc.generator(4)
    i = 0
    for [X, S], L in rc_it:
        for x, l in zip(X, L):
            print(rc.tokenizer.ids_to_tokens(x))
            print(l)
            print()
            i += 1
        if i % 30 == 0:
            if input('C?') == 'q':
                break


if __name__ == '__main__':
    test()