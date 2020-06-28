#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/5/10 9:59
@Author  : Apple QXTD
@File    : input_rc.py
@Desc:   :
"""
from data_deal.base_input import *
from data_deal.trans_output import TransOutput


class RCInput(BaseInput):
    def __init__(self, *args, **kwargs):
        super(RCInput, self).__init__(*args, **kwargs)

        self.last_sample_num = None
        self.dict_path = join(BERT_PATH, 'vocab.txt')

        token_dict, self.keep_tokens = load_vocab(
            dict_path=self.dict_path,
            simplified=True,
            startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        )
        self.tokenizer = Tokenizer(token_dict, do_lower_case=True)

        self.max_p_len = 368 - 64 - 46 - 5
        self.max_q_len = 64
        self.max_a_len = 46
        self.batch_size = 4
        self.need_evaluate = True
        self.out_trans = TransOutput()

        if self.from_pre_trans:
            self.data_dict = {
                0: join(DATA_PATH, 'trans', 'trans_rc_0.txt'),
                1: join(DATA_PATH, 'trans', 'trans_rc_1.txt'),
                2: join(DATA_PATH, 'trans', 'trans_rc_2.txt'),
                3: join(DATA_PATH, 'trans', 'trans_rc_3.txt'),
            }
        self._a = 0
        self._b = 0

    def generator(self, batch_size=4, data_type=0, need_shuffle=False, cycle=False):
        if not isinstance(data_type, list):
            data_type = [data_type]
        data_files = []
        for t in data_type:
            if t not in self.data_dict.keys():
                raise ValueError('data_type {} not in dict: {}'.format(t, self.data_dict.keys()))
            data_files.append(self.data_dict[t])
        X, S, A = [], [], []
        sample_iter = self.get_sample(data_files, need_shuffle=need_shuffle, cycle=cycle)
        while True:
            sample = next(sample_iter)
            for x, s, a in self.encode(sample):
                if x is None:
                    continue
                X.append(x)
                S.append(s)
                A.append(a)
                # dx = self.tokenizer.decode(x)
                # da = self.tokenizer.decode(a)
                # print(dx)
                # print(da)
                if len(X) >= batch_size:
                    X = sequence_padding(X)
                    S = sequence_padding(S)
                    A = sequence_padding(A, self.max_a_len)
                    yield [X, S], A
                    X, S, A = [], [], []

    def get_rc_sample(self, sample):
        if self.from_pre_trans:
            results = [sample]
        else:
            context, goals, turns, ori_replace_dict = self.reader.trans_sample(sample, need_replace_dict=True)
            if context is None:
                return []
            context_str = 'conversation' if 'conversation' in sample.keys() else 'history'
            history = sample[context_str]
            results = []
            for i, sentence, goal, turn, ori_rp_dict in zip(
                    list(range(len(context))), context, goals, turns, ori_replace_dict):
                if not turn:
                    continue
                replace_dict = self.out_trans.search_choices(sample, sentence, history=history[:i + 1])
                if len(replace_dict) > 0:
                    results.append(
                        {
                            'history': history[:i] + [sentence],
                            'replace_dict': replace_dict,
                            'result': ori_rp_dict,
                        }
                    )
        return results

    def encode(self, sample: dict):
        samples = self.get_rc_sample(sample)
        for sample in samples:
            for q_key, answer in sample['result'].items():
                if q_key not in sample['replace_dict'].keys():
                    continue
                self._a += 1
                context = sample['replace_dict'][q_key]  # it's a list
                context = '|'.join(context).replace(' ', '')  # 全部使用 | 作为分隔符
                question = '|'.join(sample['history']).replace(' ', '')  # 全部使用 | 作为分隔符
                answer = answer.replace(' ', '')
                question += '|{}'.format(q_key)  # question额外添加询问的标记
                # 如果长度超了，就截取
                if len(context) > self.max_p_len - 5:
                    answer_start = self.dynamic_find(context, answer)
                    if answer_start < 0:
                        continue
                    trunc_res = self.trans_sample((context, question, answer, answer_start))
                    if trunc_res is None:
                        continue
                    context, question, answer = trunc_res[:3]
                # 编码
                a_token_ids, _ = self.tokenizer.encode(answer, max_length=self.max_a_len + 1)
                q_token_ids, _ = self.tokenizer.encode(question)
                while len(q_token_ids) > self.max_q_len + 1:
                    q_token_ids.pop(1)
                p_token_ids, _ = self.tokenizer.encode(context, max_length=self.max_p_len + 1)
                token_ids = [self.tokenizer._token_cls_id]
                token_ids += ([self.tokenizer._token_mask_id] * self.max_a_len)
                token_ids += [self.tokenizer._token_sep_id]
                token_ids += (q_token_ids[1:] + p_token_ids[1:])
                segment_ids = [0] * len(token_ids)
                self._b += 1
                yield token_ids, segment_ids, a_token_ids[1:]

    def trans_sample(self, sample):
        context, question, answer, answer_start = sample
        if len(question) > self.max_q_len:
            question = question[-self.max_q_len:]
        if len(answer) > self.max_a_len:
            answer = answer[:self.max_a_len]
        if len(context) - len(answer) > 220:
            tail_len = len(context) - len(answer) - answer_start
            if tail_len > answer_start:  # 截取尾部的文本
                tail_index = random.randint(-tail_len, int(-tail_len / 2))
                context = context[:tail_index]
                answer_start = self.dynamic_find(context, answer)
                if answer_start < 0:
                    answer = ''
            else:
                end_index = random.randint(int(answer_start / 2), answer_start)
                context = context[end_index:]
                answer_start = self.dynamic_find(context, answer)
                if answer_start < 0:
                    answer = ''
        if len(context) > self.max_p_len:
            if answer_start < 0:
                context = context[:self.max_p_len]
            else:
                offset = len(context) - self.max_p_len
                if answer_start >= offset:
                    context = context[offset:]
                    answer_start -= offset
                elif len(answer) + answer_start <= len(context) - offset:
                    context = context[:self.max_p_len]
                else:
                    # answer 最大长度64 这种情况不存在
                    return None
        return context, question, answer, answer_start

    def dynamic_find(self, sentence, piece):
        answer_start = -1
        p = 0
        while answer_start < 0:
            if p * 2 >= len(piece) - 4:
                break
            answer_start = sentence.find(piece[p * 2:])
            p += 1
        if answer_start > 0:
            answer_start = max(0, answer_start - p * 2 + 2)
        return answer_start

def test():
    rc = RCInput()
    rc_it = rc.generator(4)
    i = 0
    for s in rc_it:
        i += 1
        print('i: ', i)
        print(rc._a)
        print(rc._b)
        if i % 30 == 0:
            if input('C?') == 'q':
                break


if __name__ == '__main__':
    test()