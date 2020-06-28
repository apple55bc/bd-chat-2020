#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/6/4 13:56
@File      :score_fn.py
@Desc      :
"""


def test_score(file_path):
    with open(file_path, encoding='utf-8') as fr:
        score = 0.0
        sample_num = 0
        history = []
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.strip()
            if line == '':
                print('\n'.join(history))
                while True:
                    s = input('Score ? ')
                    try:
                        s = float(s)
                        break
                    except ValueError:
                        print('"s" Must be float type !')
                        continue
                score += s
                sample_num += 1
                print('Score: {:.2f}. Num: {}. Ave score: {:.4f}\n'.format(score, sample_num, score / sample_num))
                history = []
            else:
                history.append(line)
        print('Final ===> Score: {:.2f}. Num: {}. Ave score: {:.4f}\n\n'.format(score, sample_num, score / sample_num))


if __name__ == '__main__':
    # test_score('../test_1_sample.txt')  # 0.979
    test_score('../test_2_sample.txt')  # 0.9650

"""
[1] 你 告诉 我 一下 几点 了 可以 吗 ？
现在 是 上午 8 点 哦 。
那 还好 ， 迟 不了 ， 谢谢 你 了 。
不客气哦，对了今天济南晴转多云,南风,最高气温:24℃,最低气温:14℃，注意保暖哦。

[1] 你好 啊 ， 麻烦 问 一下 现在 几点 了 ？
现在 是 20 18 年 10 月 17 日 ， 上午 7 : 00 。
好 嘞 ， 谢 啦 ， 有 你 真 好 哦 。
[2] 嘿嘿 ， 能 帮到 你 我 很 开心 呢 ， 天气 方面 要 不要 看看 呀 ？
好 啊 ， 正想 问 你 呢 。
成都今天阴转小雨,无持续风向,最高气温:18℃,最低气温:14℃，注意保暖哦。

[1] 我 想 问 有 关于 周杰伦 的 新闻 吗 ？
当然有啦。18日，周杰伦发布了新歌《等你下课》，勾起了大家对青春的回忆。除了周杰伦，你的青春日记里是否还有这些歌手？陳奕迅所長张惠妹aMEI_feat_阿密特刘若英梁静茹孙燕姿五月天王力宏……你还记得那些骑车上学听歌的岁月吗？
"""
