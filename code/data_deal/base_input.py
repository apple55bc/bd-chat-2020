#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/24 23:19
@File      :base_input.py
@Desc      :
"""
from cfg import *
import re
from bert4keras_5_8.tokenizers import Tokenizer, load_vocab
import numpy as np
from bert4keras_5_8.snippets import sequence_padding
import random
import json


class BaseRead(object):
    def __init__(self, drop_space=True):
        self.all_goals = ['未知', '', '关于 明星 的 聊天', '兴趣点 推荐', '再见', '天气 信息 推送', '寒暄', '提问', '播放 音乐', '新闻 推荐',
                          '新闻 点播', '电影 推荐', '美食 推荐', '问 User 姓名', '问 User 年龄', '问 User 性别', '问 User 爱好',
                          '问 天气', '问 日期', '问 时间', '问答', '音乐 推荐', '音乐 点播']
        self.all_profiles = ['同意 的 poi', '同意 的 兴趣点', '同意 的 新闻', '同意 的 电影', '同意 的 美食', '同意 的 音乐',
                             '喜欢 的 poi', '喜欢 的 兴趣点', '喜欢 的 新闻', '喜欢 的 明星', '喜欢 的 电影', '喜欢 的 音乐',
                             '姓名', '居住地', '年龄区间', '性别', '拒绝', '接受 的 电影', '接受 的 音乐', '没有接受 的 电影',
                             '没有接受 的 音乐', '职业状态']
        self.all_kg_type = ['[n][n][n]', '主演', '人均价格', '体重', '出生地', '口碑', '喜好', '喜欢', '国家地区', '地址', '导演',
                            '属相', '成分', '成就', '新闻', '日期', '时间', '星座', '演唱', '特色菜', '生日', '简介', '类型',
                            '获奖', '血型', '订单量', '评分', '评论', '身高', '适合吃', '适合听']
        self.double_turn_goals = ['关于 明星 的 聊天', '兴趣点 推荐', '新闻 推荐', '电影 推荐']
        self.goal_comp = re.compile('\[\d+\]\s*[^(]*')
        self.situation_comp = re.compile('聊天 ([^:]*) : ')
        self.goal_num_comp = re.compile('\[(\d+)\]')
        self.all_goals_c = [s.replace(' ', '') for s in self.all_goals]
        self.drop_space = drop_space

        self._kg_comp = re.compile('-*\d+')
        self._check_first_comp = re.compile('\[1\][^(]+\(([^(,，。.]+)主动')
        self._num_comp = re.compile("\[(\d)+\](.*)")
        self._ch_date_comp = re.compile('\d[\d ]{3,}(年)[\d ]+(月)[\d ]+(日)')
        self._birthday_comp = re.compile('\d[\d ]{3,}(-)[\d ]+(-)[\d ]+')
        self._sign_comp = re.compile('[,.?!:，。？！： ]')
        self._search_comp = re.compile('\[[PCKpck]-[^\[\]]+\]')
        # 统一一下部分玩意的格式
        self.trans_kv = [
            ['摄氏度', '℃'],
            [' ', ''],  # 空格全删！
        ]
        if self.drop_space:
            self.all_goals = [v.replace(' ', '') for v in self.all_goals]
            self.double_turn_goals = [v.replace(' ', '') for v in self.double_turn_goals]

    def trans_sample(self, sample: dict, return_rest_goals=False,
                     need_replace_dict=False, need_bot_trans=True, is_predict=False):
        context_str = 'conversation' if 'conversation' in sample.keys() else 'history'
        if is_predict:
            bot_turn = True if len(sample[context_str]) % 2 == 0 else False
        else:
            bot_turn = self._check_bot_first(sample['goal'])
        if bot_turn is None:
            return [None] * (3 + (1 if return_rest_goals else 0) + (1 if need_replace_dict else 0))
        all_goals = self.find_all_goals(sample['goal'])
        context = []
        goals = []
        turns = []
        used_goals = []
        replace_dicts = []
        for ori_sentence in sample[context_str]:
            replace_dict = {}
            sentence, n = self._strip_n(ori_sentence)
            if bot_turn and need_bot_trans:
                sentence = self.trans_bot_sentence(
                    sentence, sample['user_profile'], sample['knowledge'], need_replace_dict=need_replace_dict)
                if need_replace_dict:
                    sentence, replace_dict = sentence
            if sentence is None:
                sentence, n = self._strip_n(ori_sentence)
                context.append(sentence)
                if n is None:
                    # goals.append(goals[-1] if len(goals) > 0 else 0)
                    goals.append(1)
                elif n not in all_goals.keys():
                    goals.append(0)
                else:
                    goals.append(all_goals[n])
                    used_goals.append(n)
                turns.append(False)
                bot_turn = not bot_turn  # 不进行训练，不然乱七八糟的
                if need_replace_dict:
                    replace_dicts.append({})
            else:
                if self.drop_space:
                    sentence = sentence.replace(' ', '')
                context.append(sentence)
                if n is None:
                    # goals.append(goals[-1] if len(goals) > 0 else 0)
                    goals.append(1)
                elif n not in all_goals.keys():
                    goals.append(0)
                else:
                    goals.append(all_goals[n])
                    used_goals.append(n)
                turns.append(bot_turn)
                bot_turn = not bot_turn
                if need_replace_dict:
                    replace_dicts.append(replace_dict)
        unused_goals = [(k, v) for k, v in all_goals.items() if k not in used_goals]
        unused_goals = sorted(unused_goals, key=lambda x: x[0])
        results = [context, goals, turns]
        if return_rest_goals:
            results += [unused_goals]
        if need_replace_dict:
            results += [replace_dicts]
        # unused_goals:  {goal_num: goal_index, ... }
        return results

    def trans_bot_sentence(self, sentence, pro: dict, kg: list, need_replace_dict=False):
        # 统一一下部分玩意的格式
        for k, v in self.trans_kv:
            sentence = sentence.replace(k, v)
        # 先以图谱优先替换，排序总是按照长短依次排序
        replace_dict = {}
        kg = sorted(kg, key=lambda _k: len(_k[2]), reverse=True)
        check_complete_words = []  # 如果整句都被替换了，验证下替换的是不是知识库里的原句，如果不是就不替换
        for kg_ in kg:
            kg_type = self.clean_kg_type(kg_[1])
            res_dict = self._trans_bot_sen(sentence, kg_[2], 'K', kg_type)
            replace_dict.update(res_dict)
            check_complete_words.append(kg_[2].replace(' ', ''))
            # 针对某些奇怪的坑爹字符 ~ 做一些处理
            if kg_type == '[n][n][n]' and '~' in kg_[2][:4]:
                kg_v = kg_[2]
                kg_v = kg_v[:4].replace('~', '转') + kg_v[4:]
                res_dict = self._trans_bot_sen(sentence, kg_v, 'K', kg_type)
                check_complete_words.append(kg_v.replace(' ', ''))
            replace_dict.update(res_dict)
            if kg_[1] == '评论':
                res_dict = self._trans_bot_sen(sentence, kg_[0], 'C', self.clean_kg_type(kg_[1]))
                replace_dict.update(res_dict)
        pro_list = sorted([[k, v] for k, v in pro.items()], key=lambda _k: len(_k[1]), reverse=True)
        for k, v in pro_list:
            if not isinstance(v, list):
                v = [v]
            for v_ in v:
                res_dict = self._trans_bot_sen(sentence, v_, 'P', k, strict=True)
                replace_dict.update(res_dict)
                check_complete_words.append(v_.replace(' ', ''))
        replace_items = sorted([[k, v] for k, v in replace_dict.items()],
                               key=lambda x: len(x[0].replace(' ', '')), reverse=True)

        def drop_duplicate(s, replace_str):
            # 标点符号去重
            replace_str = replace_str[1:-1]
            r = re.search('\[{}\] *'.format(replace_str) + '([？！，。]+) *([？！，。]+)', s)
            if r and len(r.groups()) > 1:
                mid_s = r.group()[:r.span(1)[1] - r.span(0)[0]]
                s = s[:r.span()[0]] + mid_s + s[r.span()[1]:]
            return s

        replace_dict = {}
        # print("replace items:\n", replace_items)
        for k, v in replace_items:
            # k 是替换的原句，v是标识符
            c_k = self._sign_comp.sub('', k)
            if len(c_k) <= 0:
                continue
            if len(self._sign_comp.sub('', sentence)) - len(c_k) <= 0:
                tag = False
                for cp_s in check_complete_words:
                    if cp_s == sentence:
                        tag = True
                        break
                if not tag:
                    continue
            if len(c_k) == 1 and '0' <= c_k <= '9':
                continue
            if self.drop_space:
                v = v.replace(' ', '')
            if v not in replace_dict.keys():
                replace_dict[v] = k
                sentence = sentence.replace(k, v)
            else:
                if k in sentence:  # 如果有同类型的属性存在，并且这个类型的值也存在于句子中，那么特殊处理
                    # print('*' * 30)
                    # print('Inner')
                    # print('sentence: ', sentence)
                    last_index = sentence.find(v) + len(v)
                    sentence = sentence[:last_index]
                    sentence = sentence.replace(k, '')  # 清除重复的内容
                    # print('after: ', sentence)
                    # print()
                else:
                    sentence = drop_duplicate(sentence, v)
        # print('>' * 20)
        # print('', json.dumps(replace_dict, ensure_ascii=False, indent=4))
        # print()
        # 最后 对评论的前面部分没有匹配到的内容进行去除
        sentence = self._clean_redundant(sentence)

        # 新闻的回复修正。新闻训练时候提取不干净，会出现多个碎片句子。这时候会出现： [K-新闻]  [P-喜欢的明星]
        replace_items = self._search_comp.findall(sentence)
        spe_items = {'[K-新闻]', '[k-新闻]', '[P-喜欢的明星]', '[p-喜欢的明星]'}
        inner = set(replace_items).intersection(spe_items)
        if len(inner) >= 2:
            # 寻找 b_i
            dots = list('，。？！,.?!')
            b_i = sentence.find('[P-喜欢的明星]')
            if b_i < 0:
                b_i = sentence.find('[p-喜欢的明星]')
            while b_i > 1:
                if sentence[b_i - 1] in dots:
                    break
                b_i -= 1
            # 寻找 e_i
            e_i = sentence.find('[K-新闻]')
            if e_i < 0:
                e_i = sentence.find('[k-新闻]')
            while e_i > 1:
                if sentence[e_i - 1] in dots:
                    break
                e_i -= 1
            sentence = sentence[:b_i] + sentence[e_i:]
        
        if need_replace_dict:
            return sentence, replace_dict
        return sentence

    def _clean_redundant(self, sentence):
        # 对评论的前面部分没有匹配到的内容进行去除
        # 找评论、新闻与前一个标识符之间的标点符号，去除标点符号之间的内容。
        safe_tag = 0
        except_list = []
        while True:
            all_signs = self._search_comp.findall(sentence)
            if len(all_signs) <=  1:
                break
            safe_tag += 1
            if safe_tag > 5:
                break
            c_span = None
            c_sign = None
            duplicate = False
            exist_list = []  #任何标记都不重复两次
            for sign in all_signs:
                if 'K-评论' in sign or 'K-新闻' in sign or sign in exist_list:
                    if sign not in except_list:
                        b_i = sentence.find(sign)
                        c_span = (b_i, b_i + len(sign))
                        c_sign = sign
                        if sign in exist_list:
                            duplicate = True
                        break
                exist_list.append(sign)
            if c_span:
                if not duplicate:
                    max_e_i = 0
                    for sign in all_signs:
                        if sign == c_sign:
                            continue
                        e_i = sentence.find(sign) + len(sign)
                        if e_i >  max_e_i:
                            max_e_i = e_i
                    # 搜寻区间内的标点符号
                    if max_e_i < c_span[0]:  # 只有在新闻、评论之前出现的才有效，在后面不会处理
                        rp_index_b = c_span[0] - max_e_i
                        for d in ['，', '。', ',', '！', '？']:
                            if d in sentence[max_e_i:c_span[0]]:
                                _rp_index_b = sentence[max_e_i:c_span[0]].find(d) + 1
                                if _rp_index_b < rp_index_b:
                                    rp_index_b = _rp_index_b
                        rp_index_e = 0
                        rev = sentence[max_e_i:c_span[0]][::-1]
                        for d in ['，', '。', ',', '！', '？']:
                            if d in rev:
                                rp_index_e = rev.find(d)
                                if rp_index_e == 0 or rp_index_e < rp_index_b:
                                    rp_index_e = rp_index_e

                        rp_index_b += max_e_i
                        rp_index_e = c_span[0] - rp_index_e
                        sentence = sentence[:rp_index_b] + sentence[rp_index_e:c_span[1]]
                    except_list.append(c_sign)
                else: # 处理重复的情况
                    first_i = sentence.find(c_sign)
                    second_i = sentence[first_i + len(c_sign):].find(c_sign)
                    sentence = sentence[:first_i] + sentence[first_i + len(c_sign) + second_i:]
            else:
                break
        return sentence

    def _trans_bot_sen(self, input_sen, value, tag, key, strict=False):
        key = key.replace(' ', '')
        clean_value = value.replace(' ', '')
        replace_dict = {}
        replace_str = '[{}-{}]'.format(tag, key)
        # 处理日期 日期直接处理
        d = self._ch_date_comp.search(input_sen)
        if d:
            s_p = input_sen[d.span()[0]:d.span()[1]].replace('年', '-').replace('月', '-').replace('日', '').strip()
            if s_p in clean_value:
                replace_dict[input_sen[d.span()[0]:d.span()[1]]] = replace_str
                input_sen = input_sen.replace(input_sen[d.span()[0]:d.span()[1]], replace_str)
        # 简单的包含情况
        m_ed_s = self.find_min_sentence_ed(clean_value, input_sen, strict=strict)
        if m_ed_s != '':
            replace_dict[m_ed_s] = replace_str
        # 多项的情况
        values = [v.replace(' ', '') for v in value.split('  ')]
        for v in values:
            if v in input_sen:
                replace_dict[v] = replace_str
        # 截取的情况
        values = [v.replace(' ', '') for v in re.split('[,.?!？！，。]', value)]
        if len(values) > 1:
            for v in values:
                if len(v) < 5:  # 过短的截取的文本过滤掉
                    continue
                m_ed_s = self.find_min_sentence_ed(v, input_sen, strict=strict)
                if m_ed_s != '':
                    replace_dict[m_ed_s] = replace_str
        return replace_dict

    def find_min_sentence_ed(self, sen_c, sen_m, strict=False):
        # if '曲风' in sen_c and '曲风' in sen_m:
        #     print('ll')
        if sen_c in sen_m:
            return sen_c
        elif not strict:
            cand_s = self.compare(sen_c, sen_m)
            if cand_s != '':
                return cand_s
            sen_m = re.split('[,.?!？！，。]', sen_m)
            if sen_c == '':
                return ''
            for s in sen_m:
                if len(s) <= 3:
                    continue
                cand_s = self.compare(sen_c, s)
                if cand_s != '':
                    return cand_s
            return ''
        else:
            return ''

    def find_all_goals(self, goal: str):
        goals = self.goal_comp.findall(goal)
        goal_dict = {}
        for g in goals:
            try:
                g_r = self._num_comp.search(g).groups()
                if self.drop_space:
                    v = g_r[1].strip().replace(' ', '')
                else:
                    v = g_r[1].strip()
                if v not in self.all_goals:
                    return self.error_return('Goal not in dict: {}'.format(v))
                goal_dict[g_r[0]] = self.all_goals.index(v)
            except Exception:
                return self.error_return('Find goal error: {}'.format(goal))
        return goal_dict

    def _strip_n(self, sentence: str):
        ns = self.goal_num_comp.search(sentence)
        n = None
        if ns is not None:
            sentence = sentence.replace(ns.group(), '')
            n = ns.group(1)
        return sentence, n

    def clean_kg_type(self, type_str):
        if '评论' in type_str:
            type_str = '评论'
        ns = self._kg_comp.findall(type_str)
        for n in ns:
            type_str = type_str.replace(n, '[n]')
        return type_str.split(' ')[-1]

    def _check_bot_first(self, goal: str):
        tag = self._check_first_comp.findall(goal)[0].strip().lower()
        if tag == 'user':
            return False
        elif tag == 'bot':
            return True
        else:
            return self.error_return('Bot goal error! {}'.format(goal))

    def error_return(self, message):
        logger.info('=' * 20)
        logger.warning(message)
        return None

    def compare(self, word1, word2):
        """
        :param word1: 搜寻的句子
        :param word2: 搜寻的母句子
        :return:
        """
        word1 = word1.strip()
        start = word2.find(word1[:2])
        if start < 0:
            start = 0
        rate = 0.9
        while rate > 0.4:
            end_s = start + int(len(word1) * rate)
            end = word2[end_s:].find(word1[-2:])
            if end >= 0:
                break
            rate -= 0.1
        else:
            end_s = 0
            end = len(word2) - 2
        clean_word1 = self._sign_comp.sub('', word1)
        clean_word2 = self._sign_comp.sub('', word2[start:end_s + end + 2])
        if start < 0 or end < 0:
            return ''
        dis = self.edit_distance(clean_word1, clean_word2)
        spe_list = ['这', '他', '她', '《', '》']
        if dis <= int(len(clean_word1) / 4):
            return word2[start:end_s + end + 2]
        elif dis <= int(len(clean_word1) / 2) and len(clean_word1) > 8:
            for sp in spe_list:
                if sp in word2:
                    return word2[start:end_s + end + 2]
        return ''

    def edit_distance(self, word1, word2):
        word1 = self._sign_comp.sub('', word1)
        word2 = self._sign_comp.sub('', word2)
        if word1 == word2:
            return 0
        len1 = len(word1)
        len2 = len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return int(dp[len1][len2])


class LMTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super(LMTokenizer, self).__init__(*args, **kwargs)
        for token in ['[GOAL]']:
            _token_id = self._token_dict[token]
            setattr(self, '_token_{}_id'.format(token.lstrip('[').rstrip(']').lower()), _token_id)


class BaseInput(object):
    def __init__(self, from_pre_trans=False):
        self.reader = BaseRead()

        self.last_sample_num = None
        self.from_pre_trans = from_pre_trans
        self.dict_path = join(BERT_PATH, 'vocab.txt')

        token_dict, self.keep_tokens = load_vocab(
            dict_path=self.dict_path,
            simplified=True,
            startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[GOAL]'],
        )
        self.tokenizer = LMTokenizer(token_dict, do_lower_case=True)

        self.max_len = 388
        self.batch_size = 4
        self.need_evaluate = True

        self.data_dict = {
            0: join(DATA_PATH, 'train/train.txt'),
            1: join(DATA_PATH, 'dev/dev.txt'),
            2: join(DATA_PATH, 'test_1/test_1.txt'),
            3: join(DATA_PATH, 'test_2/test_2.txt'),
        }
        if self.from_pre_trans:
            self.data_dict = {
                0: join(DATA_PATH, 'trans', 'trans_0.txt'),
                1: join(DATA_PATH, 'trans', 'trans_1.txt'),
                2: join(DATA_PATH, 'trans', 'trans_2.txt'),
                3: join(DATA_PATH, 'trans', 'trans_3.txt'),
            }

    def get_trans_sample(self, sample, return_rest_goals=False,
                         need_replace_dict=False, need_bot_trans=True, is_predict=False):
        if self.from_pre_trans:
            results = [sample['context'], sample['goals'], sample['turns']]
            if return_rest_goals:
                results.append(sample['unused_goals'])
            if need_replace_dict:
                results.append(sample['replace_dicts'])
        else:
            results = self.reader.trans_sample(sample,
                                               return_rest_goals=return_rest_goals,
                                               need_replace_dict=need_replace_dict,
                                               need_bot_trans=need_bot_trans,
                                               is_predict=is_predict,
                                               )
        return results

    def encode(self, sample: dict, need_goal_mask=True):
        context, goals, turns = self.get_trans_sample(sample)
        if context is None:
            return None, None
        token_ids = []
        segs = []

        all_goals = self.reader.find_all_goals(sample['goal'])
        all_goals = [(k, v) for k, v in all_goals.items()]
        if len(all_goals) > 3:
            all_goals = [all_goals[0]] + all_goals[-2:-1]
        goal_info = ''
        for k, v in all_goals:
            if goal_info != '':
                goal_info += ','
            if v > 1:
                goal_info += self.reader.all_goals[v]
        token_ids.extend(self.tokenizer.encode(goal_info)[0])
        token_ids.extend(self.tokenizer.encode(sample['situation'])[0][1:])
        segs.extend([0] * len(token_ids))
        for i, sentence, goal, turn in zip(list(range(len(context))), context, goals, turns):
            goal_cut = False
            if need_goal_mask and i > 0:
                if goal > 1 and random.random() < 0.5:
                    if 'history' not in sample.keys():
                        goal = 1
                        goal_cut = True
            this_goal = self.reader.all_goals[goal]
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]

            if goal == 0:
                goal_cut = True
            segs += [0 if goal_cut else 1] * (len(goal_tokens) + 1)  # goals 是否需要预测

            sen_tokens, _ = self.tokenizer.encode(sentence)
            sen_tokens = sen_tokens[1:]
            token_ids += sen_tokens
            if turn:
                segs += [1] * len(sen_tokens)
            else:
                segs += [0] * len(sen_tokens)
            if len(token_ids) >= self.max_len:
                token_ids = token_ids[-self.max_len:]
                segs = segs[-self.max_len:]
                break
        return token_ids, segs

    def encode_predict(self, sample: dict, cand_goals=None, need_goal=True, force_goal=False):
        if cand_goals is None:
            cand_goals = []
        context, goals, turns, unused_goals = self.get_trans_sample(sample, return_rest_goals=True)
        if context is None:
            return None, None
        token_ids = []
        segs = []

        all_goals = self.reader.find_all_goals(sample['goal'])
        all_goals = [(k, v) for k, v in all_goals.items()]
        if len(all_goals) >= 3:
            all_goals = [all_goals[0]] + all_goals[-2:-1]
        goal_info = ''
        for k, v in all_goals:
            if goal_info != '':
                goal_info += ','
            if v > 1:
                goal_info += self.reader.all_goals[v]
        token_ids.extend(self.tokenizer.encode(goal_info)[0])
        token_ids.extend(self.tokenizer.encode(sample['situation'])[0][1:])
        segs.extend([0] * len(token_ids))
        turn_add = 0
        goal_step = 1  # 这个用来标记上一个goal距离最后一句话有多少轮
        exist_goals = 0
        for sentence, goal, turn in zip(context, goals, turns):
            this_goal = self.reader.all_goals[goal if goal != 0 else 1]
            if this_goal in self.reader.double_turn_goals:
                turn_add += 1
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]
            segs += [1] * (len(goal_tokens) + 1)
            sen_tokens, _ = self.tokenizer.encode(sentence)
            sen_tokens = sen_tokens[1:]
            token_ids += sen_tokens
            if turn:
                segs += [1] * len(sen_tokens)
            else:
                segs += [0] * len(sen_tokens)
            if goal != 1:
                goal_step = 1
                exist_goals += 1
            else:
                goal_step += 1

        goal_index = None
        if need_goal:
            this_goal = ''
            if len(cand_goals) > 0 and len(unused_goals) > 0 and '1' <= unused_goals[0][0] <= '9':
                if (int(unused_goals[0][0]) + turn_add - 1) * 2 <= len(context) and '' not in cand_goals:
                    if not (goal_step < 4 < len(context)) and exist_goals + 1 == int(unused_goals[0][0]):
                        this_goal = self.reader.all_goals[unused_goals[0][1]]
                        if this_goal not in cand_goals:  # 检查和候选goal的区别。如果不一致，就清空
                            keep = False
                            if '推荐' in this_goal:  # 推荐很难一致
                                for cand in cand_goals:
                                    if '推荐' in cand:
                                        keep = True
                                        break
                            if not keep:
                                this_goal = ''
            if force_goal:
                if this_goal == '':
                    if len(unused_goals) == 0:
                        return None, None, None
                    this_goal = self.reader.all_goals[unused_goals[0][1]]
                else:
                    this_goal = ''
            if this_goal == '再见':  # 不可以主动再见
                this_goal = ''
            # 添加goal信息
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]
            segs += [1] * (len(goal_tokens) + 1)
            if len(this_goal) > 0:
                goal_index = unused_goals[0][0]

        if len(token_ids) >= self.max_len:
            token_ids = token_ids[-self.max_len:]
            segs = segs[-self.max_len:]
        return token_ids, segs, goal_index

    def encode_predict_final(self, sample: dict, cand_goals=None, need_goal=True, force_goal=False, silent=True):
        if cand_goals is None:
            cand_goals = []
        context, goals, turns, unused_goals = self.get_trans_sample(sample, return_rest_goals=True, is_predict=True)
        if context is None:
            return None, None
        token_ids = []
        segs = []

        all_goals = self.reader.find_all_goals(sample['goal'])
        all_goals = [(k, v) for k, v in all_goals.items()]
        if len(all_goals) >= 2:
            # all_goals = [all_goals[0]] + all_goals[-2:-1]
            all_goals = all_goals[-2:-1]
        goal_info = ''
        for k, v in all_goals:
            if goal_info != '':
                goal_info += ','
            if v > 1:
                goal_info += self.reader.all_goals[v]
        if force_goal:
            goal_info = ''
        token_ids.extend(self.tokenizer.encode(goal_info)[0])
        token_ids.extend(self.tokenizer.encode(sample['situation'])[0][1:])
        segs.extend([0] * len(token_ids))
        turn_add = 0
        goal_step = 1  # 这个用来标记上一个goal距离最后一句话有多少轮
        exist_goals = 0
        for sentence, goal, turn in zip(context, goals, turns):
            this_goal = self.reader.all_goals[goal if goal != 0 else 1]
            if this_goal in self.reader.double_turn_goals:
                turn_add += 1
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]
            segs += [1] * (len(goal_tokens) + 1)
            sen_tokens, _ = self.tokenizer.encode(sentence)
            sen_tokens = sen_tokens[1:]
            token_ids += sen_tokens
            if turn:
                segs += [1] * len(sen_tokens)
            else:
                segs += [0] * len(sen_tokens)
            if goal != 1:
                goal_step = 1
                exist_goals += 1
            else:
                goal_step += 1

        unused_goals_words = []
        for i in range(len(unused_goals)):
            word = self.reader.all_goals[unused_goals[i][1]]
            unused_goals_words.append(word)
            if '推荐' in word and '推荐' not in unused_goals_words:
                unused_goals_words.append('推荐')

        goal_index = None
        if need_goal:
            this_goal = ''
            if len(context) == 0:
                this_goal = '寒暄'
            elif len(cand_goals) > 0 and len(unused_goals) > 0 and '1' <= unused_goals[0][0] <= '9':
                if (int(unused_goals[0][0]) + turn_add - 1) * 2 <= len(context) and '' not in cand_goals:
                    if not (goal_step < 4 < len(context)):  # ------------- 删掉goals限制
                        for g in cand_goals:
                            if '推荐' in g and '推荐' in unused_goals_words:
                                # 搜寻推荐的内容
                                for _g in unused_goals_words:
                                    if '推荐' in _g:
                                        this_goal = _g
                                        break
                                break
                            elif g in unused_goals_words:
                                this_goal = g
                                break
            if force_goal:
                if this_goal == '':
                    if len(unused_goals) + len(cand_goals) == 0:
                        return None, None, None
                    for g in cand_goals:
                        if g != '':
                            this_goal = g
                    if this_goal == '' and len(unused_goals_words) > 1:
                        this_goal = unused_goals_words[-2]
                else:
                    this_goal = ''
            if this_goal == '再见' and not force_goal:  # 不可以主动再见
                this_goal = ''
            # 添加goal信息
            goal_tokens, _ = self.tokenizer.encode(this_goal)
            goal_tokens = goal_tokens[1:-1]
            token_ids += goal_tokens
            token_ids += [self.tokenizer._token_goal_id]
            segs += [1] * (len(goal_tokens) + 1)
            if len(this_goal) > 0:
                goal_index = unused_goals[0][0]
            if not silent:
                print('Actual goal: ', this_goal)

        if len(token_ids) >= self.max_len:
            token_ids = token_ids[-self.max_len:]
            segs = segs[-self.max_len:]
        if not silent:
            print('Sentence: ')
            print(self.tokenizer.decode(token_ids))
        return token_ids, segs, goal_index

    def generator(self, batch_size=12, data_type=0, need_shuffle=False, cycle=False):
        if not isinstance(data_type, list):
            data_type = [data_type]
        data_files = []
        for t in data_type:
            if t not in self.data_dict.keys():
                raise ValueError('data_type {} not in dict: {}'.format(t, self.data_dict.keys()))
            data_files.append(self.data_dict[t])
        X, S = [], []
        sample_iter = self.get_sample(data_files, need_shuffle=need_shuffle, cycle=cycle)
        info = True
        while True:
            sample = next(sample_iter)
            x, s = self.encode(sample)
            if info:
                logger.info('input: {}'.format(' '.join(self.tokenizer.ids_to_tokens(x))))
                info = False
            if x is None:
                continue
            X.append(x)
            S.append(s)
            if len(X) >= batch_size:
                X = sequence_padding(X)
                S = sequence_padding(S)
                yield [X, S], None
                X, S = [], []

    def get_sample(self, data_files, need_shuffle=False, cycle=False):
        if not isinstance(data_files, list):
            data_files = [data_files]
        new_files = []
        for f in data_files:
            if isinstance(f, int):
                new_files.append(self.data_dict[f])
            else:
                new_files.append(f)
        data_files = new_files
        assert len(data_files) > 0
        if not need_shuffle:
            file_index = 0
            while True:
                with open(data_files[file_index], encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        piece = json.loads(line)
                        yield piece
                file_index += 1
                if file_index >= len(data_files):
                    if cycle:
                        file_index = 0
                    else:
                        break
        else:
            all_data = []
            for file in data_files:
                with open(file, encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        piece = json.loads(line)
                        all_data.append(piece)
            all_index = list(range(len(all_data)))
            while True:
                random.shuffle(all_index)
                for i in all_index:
                    yield all_data[i]
                if not cycle:
                    break


def test():
    d = BaseInput()
    file_path = join(DATA_PATH, 'train/train.txt')
    # file_path = join(DATA_PATH, 'dev/dev.txt')
    # file_path = join(DATA_PATH, 'test_1/test_1.txt')
    n = int(input('Skip n: '))

    with open(file_path, encoding='utf-8') as f:
        i = 0
        for j in range(n):
            f.readline()
            i += 1
        while input('Continue?') != 'q':
            line = f.readline()
            if not line:
                break
            i += 1
            piece = json.loads(line)
            t, s = d.encode(piece)
            context_str = 'conversation' if 'conversation' in piece.keys() else 'history'
            print('piece: ', piece[context_str])
            print('goal: ', piece['goal'])
            # print('t: ', t)
            print('s: ', s)
            print('decode: ', ' '.join(d.tokenizer.ids_to_tokens(t)).replace('[SEP]', '\n==> '))
            print('n: ', i)


def test_generator():
    d = BaseInput()
    g = d.generator(10, need_shuffle=False)
    while input('continue?') != 'n':
        b = next(g)
        print(b[0][0][0])
        print(b[0][1][0])
        print(b[0][0].shape)
        print(b[0][1].shape)
        print(d.tokenizer.decode(b[0][0][0]))


if __name__ == '__main__':
    # test()
    test_generator()
