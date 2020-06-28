#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :Apple
@Time      :2020/4/29 20:16
@File      :trans_output.py
@Desc      :
"""
from cfg import *
from data_deal.base_input import BaseRead
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class TransOutput(object):
    def __init__(self, rc_tag=''):
        self.reader = BaseRead()
        self.search_comp = re.compile('\[[PCKpck]-[^\[\]]+\]')
        self.search_comp_date = re.compile('\[[kK]-\[n\]\[n\]\[n\]\]')
        self._enc_comp = re.compile('\[[PCKpck]-[^\[\]]{,8}$')
        self.goal_comp = re.compile('\[[0-9]\]')
        self.date_comp = re.compile('\d{1,2} ?月\d{1,2} ?[日号]')
        self._birthday_comp = re.compile('\d[\d ]{3,}(-)[\d ]+(-)[\d ]+')
        self.replace_couple = [
            (' ', ''),
            ('同意', '喜欢'),
            ('没有接受', '拒绝'),
            ('接受', '喜欢'),
            ('喜好', '喜欢'),
        ]
        if rc_tag != '':
            from model.model_rc import BertCL
            self.rc_model_cls = BertCL(tag=rc_tag, is_predict=True)
        else:
            self.rc_model_cls = None

    def clean_sentence(self, sentence):
        for k, v in self.replace_couple:
            sentence = sentence.replace(k, v)
        return sentence

    def pre_trans(self, answer):
        if '生日' in answer and len(answer) < 20:
            for d in self.date_comp.findall(answer):
                answer = answer.replace(d, '[k-生日]')
        return answer

    def trans_output(self, sample: dict, answer: str):
        answer = self.clean_sentence(answer)
        answer = self.pre_trans(answer)
        user_profile = {}
        for k, v in sample['user_profile'].items():
            clean_k = self.clean_sentence(k)
            user_profile[clean_k] = user_profile.get(clean_k, [])
            if not isinstance(v, list):
                v = [v]
            user_profile[clean_k].extend(v)
        kg_dict = {}
        for k in sample['knowledge']:
            clean_p = self.clean_sentence(self.reader.clean_kg_type(k[1]))
            kg_dict[clean_p] = kg_dict.get(clean_p, [])
            kg_dict[clean_p].append({
                'S': k[0],
                'O': k[2],
            })

        # 寻找goal
        context_str = 'conversation' if 'conversation' in sample.keys() else 'history'
        exist_goals = []
        for s in sample[context_str]:
            exist_goals.extend(self.goal_comp.findall(s))
        if len(exist_goals) == 0:
            max_goal = 0
        else:
            max_goal = max([s[1] for s in exist_goals])
        goal = sample['goal']
        idx = goal.find('[{}]'.format(max_goal))
        if idx < 0:
            idx = goal.find('[{}]'.format(int(max_goal) + 1))
        if idx >= 0:
            goal = goal[idx:]
        else:
            goal = ''

        replace_items = self.search_comp.findall(answer)
        # 新闻的回复修正。新闻训练时候提取不干净，会出现多个碎片句子。这时候会出现： [K-新闻]  [P-喜欢的明星]
        spe_items = {'[K-新闻]', '[k-新闻]', '[P-喜欢的明星]', '[p-喜欢的明星]'}
        inner = set(replace_items).intersection(spe_items)
        if len(inner) >= 2:
            # 寻找 b_i
            dots = list('，。？！,.?!')
            b_i = answer.find('[P-喜欢的明星]')
            if b_i < 0:
                b_i = answer.find('[p-喜欢的明星]')
            while b_i > 1:
                if answer[b_i - 1] in dots:
                    break
                b_i -= 1
            # 寻找 e_i
            e_i = answer.find('[K-新闻]')
            if e_i < 0:
                e_i = answer.find('[k-新闻]')
            while e_i > 1:
                if answer[e_i - 1] in dots:
                    break
                e_i -= 1
            answer = answer[:b_i] + answer[e_i:]
            replace_items = self.search_comp.findall(answer)

        replace_items.extend(self.search_comp_date.findall(answer))
        replace_cuple = []
        last_choice = None
        exist_none_replace = False
        for rp_item in replace_items:
            choice = None
            last_rp_len = len(replace_cuple)
            if rp_item[1] in ['p', 'P']:
                choice = user_profile.get(rp_item[3:-1], '')
                if isinstance(choice, list):
                    choice = self.judge_choices(choice, sample, goal=goal, last_choice=last_choice)
                replace_cuple.append((rp_item, choice))
            elif rp_item[1] in ['k', 'K']:
                #  新闻和评论额外的进行判定
                choices = kg_dict.get(rp_item[3:-1], [])
                obj = [d['O'] for d in choices]
                sbj = [d['S'] for d in choices]
                choice = self.judge_choices(obj, sample, sbj, goal=goal, last_choice=last_choice,
                                            identifier=rp_item, response=answer)
                replace_cuple.append((rp_item, choice))
            elif rp_item[1] in ['c', 'C']:
                choices = kg_dict.get(rp_item[3:-1], [])
                choices = [d['S'] for d in choices]
                choice = self.judge_choices(choices, sample, goal=goal, last_choice=last_choice)
                replace_cuple.append((rp_item, choice))
            else:
                logger.info('=' * 20)
                logger.info('Error rp item: {}'.format(rp_item))
                logger.info('KG: {}'.format(kg_dict))
                logger.info('P: {}'.format(user_profile))
                replace_cuple.append((rp_item, ''))
            if choice is not None and len(choice) > 0:
                last_choice = choice
            if len(replace_cuple) == last_rp_len or choice == '':
                exist_none_replace = True

        all_tags = []  # 有些可能有重复的?
        for k, v in replace_cuple:
            if k in all_tags:
                continue
            all_tags.append(k)
            start = answer.find(k)
            if start < 0:
                continue
            if '[n][n][n]' in k:
                if '~' in v:
                    v = v.replace('~', '转')
            answer = answer[:start] + v + answer[start + len(k):]
        # 清除多余的标记
        for k in self.search_comp.findall(answer):
            answer = answer.replace(k, '')
        for k in self._enc_comp.findall(answer):
            answer = answer.replace(k, '')
        # 日期转换
        sp = self._birthday_comp.search(answer)
        if sp:
            idx_0 = sp.group().find('-')
            idx_1 = sp.group()[idx_0 + 1:].find('-') + idx_0 + 1
            sp_str = list(sp.group())
            sp_str[idx_0] = '年'
            sp_str[idx_1] = '月'
            sp_str = ''.join(sp_str) + '日'
            answer_after = answer[:sp.span()[0]] + sp_str
            if sp.span()[1] < len(answer) and answer[sp.span()[1]] == '号':
                answer_after = answer_after[:-1] + answer[sp.span()[1]:]
            elif sp.span()[1] + 1 < len(answer) and answer[sp.span()[1]:sp.span()[1] + 2] == ' 号':
                answer_after = answer_after[:-1] + answer[sp.span()[1] + 1:]
            else:
                answer_after += answer[sp.span()[1]:]
            answer = answer_after
        # 标点符号去重
        answer = re.sub('([，,.!?？！。])+', '\\1', answer)
        return answer.replace(' ', ''), exist_none_replace

    def search_choices(self, sample: dict, answer: str, history:list):
        """纯粹给训练做样本删选 history需要包含正确的answer"""
        answer = self.clean_sentence(answer)
        answer = self.pre_trans(answer)
        user_profile = {}
        for k, v in sample['user_profile'].items():
            clean_k = self.clean_sentence(k)
            user_profile[clean_k] = user_profile.get(clean_k, [])
            if not isinstance(v, list):
                v = [v]
            user_profile[clean_k].extend(v)
        kg_dict = {}
        for k in sample['knowledge']:
            clean_p = self.clean_sentence(self.reader.clean_kg_type(k[1]))
            kg_dict[clean_p] = kg_dict.get(clean_p, [])
            kg_dict[clean_p].append({
                'S': k[0],
                'O': k[2],
            })

        replace_items = self.search_comp.findall(answer)
        replace_items.extend(self.search_comp_date.findall(answer))
        replace_dict = {}
        for rp_item in replace_items:
            if rp_item[1] in ['p', 'P']:
                choices = user_profile.get(rp_item[3:-1], '')
                if not isinstance(choices, list):
                    choices = None
                choices = self.filter_choices(choices, history[:-1])
            elif rp_item[1] in ['k', 'K']:
                choices = kg_dict.get(rp_item[3:-1], [])
                obj = [d['O'] for d in choices]
                sbj = [d['S'] for d in choices]
                choices = self.filter_choices(obj, history, sbj)
            elif rp_item[1] in ['c', 'C']:
                choices = kg_dict.get(rp_item[3:-1], [])
                choices = [d['S'] for d in choices]
                choices = self.filter_choices(choices, history[:-1])
            else:
                logger.info('=' * 20)
                logger.info('Error rp item: {}'.format(rp_item))
                logger.info('KG: {}'.format(kg_dict))
                logger.info('P: {}'.format(user_profile))
                continue
            if choices is not None:
                replace_dict[rp_item] = choices
        return replace_dict

    def judge_choices(self, choices: list, sample: dict, sbj=None, goal='',
                      last_choice=None, identifier=None, response=None):
        if len(choices) == 0:
            return ''
        if len(choices) == 1:
            return choices[0]
        if sbj is not None:
            assert len(sbj) == len(choices)
        scores = [0] * len(choices)
        context_str = 'conversation' if 'conversation' in sample.keys() else 'history'
        for i, choice in enumerate(choices):
            # 上下文
            check_word = choice if sbj is None else sbj[i]
            if sbj is not None:  # 如果上一个选择和这个主题相同，就给个最高分数加成
                if check_word == last_choice:
                    scores[i] += 20
            context = sample[context_str]
            for j in range(1, len(context) + 1):
                sentence = context[-j]
                scores[i] += (self._get_score(check_word, sentence) / min(j, 4))
                if sbj is not None:
                    if choice.replace(' ', '') in sentence.replace(' ', ''):
                        scores[i] -= 2 / min(j, 4)
            if last_choice is not None:
                scores[i] += (self._get_score(check_word, last_choice) * 1.3)
            # 和goal 的 匹配
            if goal != '':
                scores[i] += (self._get_score(check_word, goal) / 4)
                # 内容的匹配
                if sbj is not None:
                    scores[i] += self.bleu(choice.replace(' ', ''), goal.replace(' ', ''))
            # 如果obj存在，优先选择subject内容短的
            if sbj is not None:
                scores[i] += 2 / (len(choice) + 5)
        if identifier is not None and response is not None and \
                identifier in ['[K-新闻]', '[K-评论]', '[k-新闻]', '[k-评论]'] and self.rc_model_cls is not None:
            history = sample[context_str]
            score_gap = np.array(scores).mean()
            cands = []
            for c, s in zip(choices, scores):
                if c in cands:
                    continue
                if s >= score_gap:
                    cands.append(c)
            gen_res = self.get_rc_result(history, response, cands, identifier)
            if len(gen_res) < 4:
                index = np.array(scores).argmax()
                result = choices[index]
            else:
                for i, choice in enumerate(choices):
                    if gen_res in choice:
                        scores[i] += 1
                index = np.array(scores).argmax()
                result = choices[index]
                if gen_res in result: # 修正，补全残缺的话
                    b_i = result.find(gen_res)
                    e_i = b_i + len(gen_res)
                    dot_str = list(',.?!:，。？！：')
                    while e_i < len(result):
                        if result[e_i] not in dot_str:
                            e_i += 1
                        else:
                            break
                    result = result[b_i:e_i]
        else:
            index = np.array(scores).argmax()
            result = choices[index]
        return result

    def filter_choices(self, choices: list, history: list, sbj=None):
        """纯粹给训练做样本删选"""
        if choices is None:
            return None
        if len(choices) == 0:
            return None
        if len(choices) == 1:
            return None
        if sbj is not None:
            assert len(sbj) == len(choices)
        scores = [0] * len(choices)
        for i, choice in enumerate(choices):
            # 上下文
            check_word = choice if sbj is None else sbj[i]
            for j in range(1, len(history) + 1):
                sentence = history[-j].replace(' ', '')
                scores[i] += (self._get_score(check_word, sentence) / min(j, 4))
            # 内容的部分就不做比较
        max_score = max(scores)
        keep_choices = []
        for c, s in zip(choices, scores):
            if c in keep_choices:
                continue
            if s >= max_score:
                keep_choices.append(c)
        if len(keep_choices) <= 1:
            return None
        return keep_choices

    def get_rc_result(self, history:list, response:str, cands:list, identifier:str):
        question = '|'.join(history) + '|{}|{}'.format(response, identifier)
        context = '|'.join(cands)
        with self.rc_model_cls.session.graph.as_default():
            with self.rc_model_cls.session.as_default():
                predict_answer = self.rc_model_cls.predict(question, context)
        return predict_answer

    def _get_score(self, choice, sentence):
        choice_clean = choice.replace(' ', '')
        sentence = sentence.replace(' ', '')
        if choice_clean in sentence:
            return 2.0
        else:
            c = choice.split('  ')
            if len(c) > 1:
                for c_ in c:
                    if c_.replace(' ', '') in sentence:
                        return 1.0
        return 0.0

    def bleu(self, sen0, sen1):
        return sentence_bleu([list(sen0)], list(sen1), smoothing_function=SmoothingFunction().method1)

    def edit_distance(self, word1, word2):
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
