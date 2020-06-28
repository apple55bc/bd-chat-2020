#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/5/8 18:15
@File      :z_t.py
@Desc      :
"""

import os, sys
import re

birthday_comp = re.compile('\d[\d ]{3,}(-)[\d ]+(-)[\d ]+')
file = r'../../output/out_2020-05-10_15-23-31.txt'
file_w = r'../../output/out_2020-05-10_15-23-31-rn.txt'
i = 0
with open(file, encoding='utf-8') as fr:
    with open(file_w, mode='w', encoding='utf-8') as fw:
        while True:
            i += 1
            line = fr.readline()
            if not line:
                break
            else:
                line = line.strip()
            sp = birthday_comp.search(line)
            if sp:
                idx_0 = sp.group().find('-')
                idx_1 = sp.group()[idx_0 + 1:].find('-') + idx_0 + 1
                sp_str = list(sp.group())
                sp_str[idx_0] = '年'
                sp_str[idx_1] = '月'
                sp_str = ''.join(sp_str) + '日'
                line_after = line[:sp.span()[0]] + sp_str
                if sp.span()[1] < len(line) and line[sp.span()[1]] == '号':
                    line_after = line_after[:-1] + line[sp.span()[1]:]
                elif sp.span()[1] + 1 < len(line) and line[sp.span()[1]:sp.span()[1] + 2] == ' 号':
                    line_after = line_after[:-1] + line[sp.span()[1] + 1:]
                else:
                    line_after += line[sp.span()[1]:]
                line = line_after
            fw.writelines(line + '\n')
            if i % 43 == 0:
                print('\r', i , end='   ')