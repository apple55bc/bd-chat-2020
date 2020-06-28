#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/5/21 14:23
@File      :predict_final.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfg import *
from model.bert_lm import BertLM, Response
from data_deal.base_input import BaseInput
from data_deal.trans_output import TransOutput
import json
import collections


class FinalPredict(object):
    def __init__(self):
        save_dir = join(MODEL_PATH, 'BertLM_' + TAG)
        save_path = join(save_dir, 'trained.h5')

        data_input = BaseInput(from_pre_trans=False)
        model_cls = BertLM(data_input.keep_tokens, load_path=save_path)
        self.response = Response(model_cls.model,
                                 model_cls.session,
                                 data_input,
                                 start_id=None,
                                 end_id=data_input.tokenizer._token_sep_id,
                                 maxlen=40
                                 )
        self.goal_response = Response(model_cls.model,
                                      model_cls.session,
                                      data_input,
                                      start_id=None,
                                      end_id=data_input.tokenizer._token_goal_id,
                                      maxlen=10
                                      )
        self.out_trans = TransOutput(rc_tag='')

    def predict(self, text):

        try:
            sample = json.loads(text, encoding="utf-8", object_pairs_hook=collections.OrderedDict)
        except Exception:
            print('Error type: ', text)
            raise

        # 格式转换：
        sample['goal'] = self.strip_list(sample['goal'])
        sample['situation'] = self.strip_list(sample['situation'])
        if isinstance(sample['user_profile'], list):
            # 修正 : , 的问题
            ori_p = sample['user_profile'][0]
            new_p = ori_p
            # tag = False
            # for c in ori_p:
            #     if c == ':':
            #         if tag:
            #             c = r'","'
            #             tag = False
            #         else:
            #             tag = True
            #     elif c == ',':
            #         tag = False
            #     new_p += c
            # print('=========== new_p: ', new_p)
            sample['user_profile'] = json.loads(new_p)
        print(sample['user_profile'])

        ct_str = 'history' if 'history' in sample.keys() else 'conversation'

        goal = sample["goal"]
        knowledge = sample["knowledge"]
        history = sample[ct_str]
        response = sample["response"] if "response" in sample else "null"
        assert 'user_profile' in sample.keys(), 'user_profile is needed !'
        assert 'situation' in sample.keys(), 'situation is needed !'

        # 清理history的格式
        history = [s.replace('bot:', '') for s in history]
        history = [s.replace('Bot:', '') for s in history]
        sample[ct_str] = history

        # 对goal进行格式转换。。
        if isinstance(goal, list):
            raise ValueError('goal 需要为类似test的原始格式的！')
            bot_first = True if len(history) % 2 == 0 else False
            goal_str = ''
            exist_goals = []
            for i, goal_triple in enumerate(goal):
                if goal_triple[0] not in exist_goals:
                    exist_goals.append(goal_triple[0])
                    if i == 0:
                        goal_str = goal_str + '[{}] {} ( {} {} ) --> '.format(
                            i + 1, goal_triple[0], 'Bot 主动' if bot_first else 'User 主动', goal_triple[2])
                    else:
                        goal_str = goal_str + ' --> [{}] {} ( {} )'.format(
                            i + 1, goal_triple[0], goal_triple[2])
            sample['goal'] = goal_str

        goals = self.goal_response.goal_generate(sample, n=4)
        goals = list(set(goals))
        print('goals: ', goals)
        answer_res = self.response.generate(sample, goals=goals)
        answer, tag = self.out_trans.trans_output(sample, answer_res)
        in_context = answer in sample[ct_str]
        if tag or in_context:
            print('Ori generation: {}'.format(answer_res))
            answer_res = self.response.generate(sample, goals=goals, random=True, force_goal=in_context)
            print('More generation: {}'.format(answer_res))
            for res in answer_res:
                answer, tag = self.out_trans.trans_output(sample, res)
                if not tag:
                    break
            if tag:
                answer_res = self.response.generate(sample, goals=goals, force_goal=True, random=True)
                print('More More generation: {}'.format(answer_res))
                for res in answer_res:
                    answer, tag = self.out_trans.trans_output(sample, res)
                    if not tag:
                        break
        print()
        return answer

    def strip_list(self, value):
        res = ''
        if isinstance(value, list):
            for v in value:
                res += self.strip_list(v)
        else:
            res = value
        return res


def test():
    m = FinalPredict()
    s = r'{"situation": ["聊天 时间 : 晚上 20 : 00 ， 在 家里"], "history": ["你好 啊"], "goal": [["[1] 问答 ( User 主动 按 『 参考 知识 』   问   『 周迅 』   的 信息 ， Bot 回答 ， User 满意 并 好评 ) --> ' \
        r'[2] 关于 明星 的 聊天 ( Bot 主动 ， 根据 给定 的 明星 信息 聊   『 周迅 』   相关 内容 ， 至少 要 聊 2 轮 ， 避免 话题 切换 太 僵硬 ， 不够 自然 ) --> [3] 电影 推荐 ( Bot 主动 ， Bot 使用   『 李米的猜想 』   ' \
        r'的 某个 评论 当做 推荐 理由 来 推荐   『 李米的猜想 』 ， User 先问 电影 『 国家 地区  、 导演 、 类型 、 主演 、 口碑 、 评分 』 中 的 一个 或 多个 ， Bot 回答 ， 最终 User 接受 ) --> [4] 再见"]], ' \
        r'"knowledge": [["周迅", "主演", "李米的猜想"], ["李米的猜想", "评论", "疯狂 的 女人 疯狂 地爱 着 一个 男人"], ["李米的猜想", "评论", "故事 可以 ， 配乐 更棒 。"], ["李米的猜想", "评论", "周迅 的 灵性 在 这部 片子 里 展露 无遗 。"], ' \
        r'["李米的猜想", "评论", "放肆 的 哭 ， 为 爱 付出"]], "user_profile": ["{\"姓名\": \"杨丽菲\", \"性别\": \"女\", \"居住地\": \"深圳\", \"年龄区间\": \"18-25\", \"职业状态\": \"学生\", \"喜欢 的 明星\": [\"周迅\"], ' \
        r'\"喜欢 的 电影\": [\"苏州河\"], \"喜欢 的 poi\": [\"宅宅湘菜\"], \"同意 的 美食\": \" 剁椒鱼头\", \"同意 的 新闻\": \" 周迅 的新闻\", \"拒绝\": [\"音乐\"], \"接受 的 电影\": [\"巴尔扎克和小裁缝\", \"香港有个好莱坞\"], ' \
        r'\"没有接受 的 电影\": [\"鸳鸯蝴蝶\"]}"]}'
    s2 = r'{"situation": ["聊天 时间 : 晚上 22 : 00 ， 在 家里     聊天 主题 : 学习 退步"], "history": [], "goal": [["] 寒暄 ( Bot 主动 ， 根据 给定 的 『 聊天 主题 』 寒暄 ， 第一句 问候 要 带 User 名字 ， 聊天 内容 不要 与 『 聊天 时间 』 矛盾 ， 聊天 要 自然 ， 不要 太 生硬 ) --> [2] 提问 ( Bot 主动 ， 最 喜欢 谁 的 新闻 ？ User 回答 ", " 最 喜欢   『 周杰伦 』   的 新闻 ) --> [3] 新闻 推荐 ( Bot 主动 ， 推荐   『 周杰伦 』   的 新闻   『 台湾歌手 周杰伦 今天 被 聘请 成为 “ 中国 禁毒 宣传 形象大使 ” 。 周杰伦 表示 ， 他 将 以 阳光 健康 的 形象 向 广大 青少年 发出 “ 拒绝 毒品 ， 拥有 健康 ” 的 倡议 ， 并 承诺 今后 将 积极 宣传 毒品 危害 ， 倡导 全民 珍爱 生命 ， 远离 毒品 。 转 ， 与 周杰伦 一起 拒绝 毒品 ！ 』 ,   User 接受 。 需要 聊 2 轮 ) --> [4] 再见"]], "knowledge": [["金立国", "喜欢 的 新闻", "周杰伦"], ["周杰伦", "新闻", "台湾歌手 周杰伦 今天 被 聘请 成为 “ 中国 禁毒 宣传 形象大使 ” 。 周杰伦 表示 ， 他 将 以 阳光 健康 的 形象 向 广大 青少年 发出 “ 拒绝 毒品 ， 拥有 健康 ” 的 倡议 ， 并 承诺 今后 将 积极 宣传 毒品 危害 ， 倡导 全民 珍爱 生命 ， 远离 毒品 。 转 ， 与 周杰伦 一起 拒绝 毒品 ！"]], "user_profile": ["{\"姓名\": \"金立国:性别\": \"男:居住地\": \"厦门:年龄区间\": \"小于18:职业状态\": \"学生:喜欢 的 明星\": 周杰伦, \"喜欢 的 音乐\": 淡水海边, \"喜欢 的 兴趣点\": 探炉烤鱼（湾悦城店）, \"同意 的 美食\": \" 烤鱼:同意 的 新闻\": \" 周杰伦 的新闻:拒绝\": 电影, \"接受 的 音乐\": 眼泪成诗(Live):刀马旦:骑士精神:屋顶:花海:黄浦江深, \"没有接受 的 音乐\": 迷魂曲:雨下一整晚}"]}'
    s3 = r'{"situation": ["聊天 时间 : 上午 8 : 00 ， 去 上班 路上     聊天 主题 : 工作 压力 大"], "history": [], "goal": [["[1] 寒暄 ( Bot 主动 ， 根据 给定 的 『 聊天 主题 』 寒暄 ， 第一句 问候 要 带 User 名字 ， 聊天 内容 不要 与 『 聊天 时间 』 矛盾 ， 聊天 要 自然 ， 不要 太 生硬 ) --> [2] 提问 ( Bot 主动 ， 问 User 最 喜欢 的 电影 名 ？ User 回答 ", "   最 喜欢 『 刺客聂隐娘 』 ) --> [3] 提问 ( Bot 主动 ， 问 User 最 喜欢   『 刺客聂隐娘 』   的 哪个 主演 ， 不 可以 问 User 『 刺客聂隐娘 』 的 主演 是 谁 。 User 回答 ", "   最 喜欢 『 舒淇 』 ) --> [4] 关于 明星 的 聊天 ( Bot 主动 ， 根据 给定 的 明星 信息 聊   『 舒淇 』   相关 内容 ， 至少 要 聊 2 轮 ， 避免 话题 切换 太 僵硬 ， 不够 自然 ) --> [5] 电影 推荐 ( Bot 主动 ， Bot 使用   『 千禧曼波之蔷薇的名字 』   的 某个 评论 当做 推荐 理由 来 推荐   『 千禧曼波之蔷薇的名字 』 ， User 拒绝 ， 拒绝 原因 可以 是 『   看过 、 暂时 不想 看 、 对 这个 电影 不感兴趣   或   其他 原因 』 ；   Bot 使用 『 飞一般爱情小说 』   的 某个 评论 当做 推荐 理由 来 推荐   『 飞一般爱情小说 』 ， User 先问 电影 『 国家 地区 、 导演 、 类型 、 主演 、 口碑 、 评分 』 中 的 一个 或 多个 ， Bot 回答 ， 最终 User 接受 。 注意 ", "   不要 在 一句 话 推荐 两个 电影 ) --> [6] 再见"]], "knowledge": [["王力宏", "获奖", "华语 电影 传媒 大奖 _ 观众 票选 最受 瞩目 表现"], ["王力宏", "获奖", "台湾 电影 金马奖 _ 金马奖 - 最佳 原创 歌曲"], ["王力宏", "获奖", "华语 电影 传媒 大奖 _ 观众 票选 最受 瞩目 男演员"], ["王力宏", "获奖", "香港电影 金像奖 _ 金像奖 - 最佳 新 演员"], ["王力宏", "出生地", "美国   纽约"], ["王力宏", "简介", "男明星"], ["王力宏", "简介", "很 认真 的 艺人"], ["王力宏", "简介", "一向 严谨"], ["王力宏", "简介", "“ 小将 ”"], ["王力宏", "简介", "好 偶像"], ["王力宏", "体重", "67kg"], ["王力宏", "成就", "全球 流行音乐 金榜 年度 最佳 男歌手"], ["王力宏", "成就", "加拿大 全国 推崇 男歌手"], ["王力宏", "成就", "第 15 届华鼎奖 全球 最佳 歌唱演员 奖"], ["王力宏", "成就", "MTV 亚洲 音乐 台湾 最 受欢迎 男歌手"], ["王力宏", "成就", "两届 金曲奖 国语 男 演唱 人奖"], ["王力宏", "评论", "力宏 必然 是 最 棒 的 ～ ～ ！"], ["王力宏", "评论", "永远 的 FOREVER   LOVE ~ ！ ！"], ["王力宏", "评论", "在 银幕 的 表演 和 做 娱乐节目 一样 无趣 ， 装 逼成 性"], ["王力宏", "评论", "PERFECT   MR . RIGHT ! !"], ["王力宏", "评论", "有些 歌 一直 唱進 心底 。 。 高學歷 又 有 才 華 。 。"], ["王力宏", "生日", "1976 - 5 - 17"], ["王力宏", "身高", "180cm"], ["王力宏", "星座", "金牛座"], ["王力宏", "血型", "O型"], ["王力宏", "演唱", "一首 简单 的 歌 ( Live )"], ["王力宏", "演唱", "KISS   GOODBYE ( Live )"], ["一首简单的歌(Live)", "评论", "一首 简单 的 歌 ， 却是 一首 最 不 简单 的 歌 。"], ["一首简单的歌(Live)", "评论", "你 唱 的 也好 好听 ， 是 宝藏 啊 兔 兔"], ["一首简单的歌(Live)", "评论", "超爱 王力宏 的 歌 ， 但是 ， 唱起来 真难 呀 ， 哈哈哈 哈哈哈 ， 这才 是 大神 级别 的 歌手 ！"], ["一首简单的歌(Live)", "评论", "97 年 的 我 ， 不 知道 是否 有 同道中人 ， 一直 喜欢 这些 歌"], ["一首简单的歌(Live)", "评论", "07 年 那 年初三 ， 第一次 无意 从 同学 手机 中 听到 ， 深深 被 吸引 ， 一直 如此"], ["KISS", "GOODBYE(Live) 评论", "明明 不爱 我 了   为什么 不放过 我"], ["KISS", "GOODBYE(Live) 评论", "得不到 就是 得不到 不要 说 你 不 想要"], ["KISS", "GOODBYE(Live) 评论", "我 知道 你 无意 想 绿 我 ， 只是 忘 了 说 分手 ， 只能 说 我 还是 太嫩 了 ， 没想到 还是 会 被 影响 到 心情 ， 我 是 真的 深深 被 你 打败 了"], ["KISS GOODBYE(Live)", "评论", "《 Kiss   Goodbye 》 是 一首 朴实无华 、 自然 悦耳 的 抒情歌 ， 歌曲 充分 展现 了 王力宏 自创 的 Chinked - out 音乐风格 的 独特 魅力   。"], ["KISS GOODBYE(Live)", "评论", "《 Kiss   Goodbye 》 是 王氏 情歌 的 催泪 之作 ；"], ["KISS GOODBYE(Live)", "评论", "这 首歌曲 表达 了 恋人 每 一次 的 分离 都 让 人 难以 释怀 ， 每 一次 “ Kiss   Goodbye ” 都 让 人 更 期待 下 一次 的 相聚 。"], ["KISS GOODBYE(Live)", "评论", "王力宏 在 这 首歌 里 写出 了 恋人们 的 心声 ， 抒发 了 恋人 之间 互相 思念 对方 的 痛苦 。"]], "user_profile": ["{\"姓名\": \"周明奇\", \"性别\": \"男\", \"居住地\": \"桂林\", \"年龄区间\": \"大于50\", \"职业状态\": \"工作\", \"喜欢 的 明星\": [\"舒淇\", \"周杰伦\"], \"喜欢 的 电影\": [\"刺客聂隐娘\"], \"喜欢 的 音乐\": [\"兰亭序\"], \"同意 的 美食\": \" 烤鱼\", \"同意 的 新闻\": \" 舒淇 的新闻; 周杰伦 的新闻\", \"拒绝\": [\"poi\"], \"接受 的 电影\": [], \"接受 的 音乐\": [], \"没有接受 的 电影\": [], \"没有接受 的 音乐\": []}"]}'
    s4 = r'{"situation": ["聊天 时间 : 中午 12 : 00 ， 在 学校     聊天 主题 : 学习 退步"], "history": [], "goal": [["[1] 寒暄 ( Bot 主动 ， 根据 给定 的 『 聊天 主题 』 寒暄 ， 第一句 问候 要 带 User 名字 ， 聊天 内容 不要 与 『 聊天 时间 』 矛盾 ， 聊天 要 自然 ， 不要 太 生硬 ) --> [2] 提问 ( Bot 主动 ， 问 User 最 喜欢 的 电影 名 ？ User 回答 ", " 最 喜欢 『 一起飞 』 ) --> [3] 提问 ( Bot 主动 ， 问 User 最 喜欢   『 一起飞 』   的 哪个 主演 ， 不 可以 问 User 『 一起飞 』 的 主演 是 谁 。 User 回答 ", " 最 喜欢 『 林志颖 』 ) --> [4] 关于 明星 的 聊天 ( Bot 主动 ， 根据 给定 的 明星 信息 聊   『 林志颖 』   相关 内容 ， 至少 要 聊 2 轮 ， 避免 话题 切换 太 僵硬 ， 不够 自然 ) --> [5] 电影 推荐 ( Bot 主动 ， Bot 使用   『 天庭外传 』   的 某个 评论 当做 推荐 理由 来 推荐   『 天庭外传 』 ， User 拒绝 ， 拒绝 原因 可以 是 『   看过 、 暂时 不想 看 、 对 这个 电影 不感兴趣   或   其他 原因 』 ；   Bot 使用 『 一屋哨牙鬼 』   的 某个 评论 当做 推荐 理由 来 推荐   『 一屋哨牙鬼 』 ， User 先问 电影 『 国家 地区 、 导演 、 类型 、 主演 、 口碑 、 评分 』 中 的 一个 或 多个 ， Bot 回答 ， 最终 User 接受 。 注意 ", " 不要 在 一句 话 推荐 两个 电影 ) --> [6] 再见"]], "knowledge": [["张晓佳", "喜欢", "一起飞"], ["张晓佳", "喜欢", "林志颖"], ["林志颖", "出生地", "中国   台湾"], ["林志颖", "简介", "典型 的 完美 主义者"], ["林志颖", "简介", "娱乐圈 不老 男神"], ["林志颖", "简介", "隐形 富豪"], ["林志颖", "简介", "明星 艺人"], ["林志颖", "简介", "“ 完美 奶爸 ”"], ["林志颖", "体重", "58kg"], ["林志颖", "成就", "20 08 年度 最佳 公益 慈善 明星 典范"], ["林志颖", "成就", "华鼎奖 偶像 励志 类 最佳 男演员"], ["林志颖", "成就", "1996 年 马英九 颁赠 反毒 大使 奖章"], ["林志颖", "成就", "台湾 第一位 授薪 职业 赛车手"], ["林志颖", "成就", "最具 影响力 全能 偶像 艺人"], ["林志颖", "评论", "看过 了 他 的 个人 心路历程 ， 很 值得 敬佩 ！"], ["林志颖", "评论", "喜欢 他 的 娃娃脸 。 精美 的 五官 佩服 他 的 经历 ， 他 ， 偶像 ！"], ["林志颖", "评论", "不 知道 该 怎么 来 形容 他 ， 太 完美 的 一个 人 了 ！ ！"], ["林志颖", "评论", "喜欢 他 在 《 变 身 男女 》 中 与 姚笛 的 对手戏 。"], ["林志颖", "评论", "perfect   “ boy ”"], ["林志 颖", "生日", "1974 - 10 - 15"], ["林志颖", "身高", "172cm"], ["林志颖", "星座", "天秤座"], ["林志颖", "血型", "O型"], ["林志颖", "属相", "虎"], ["林志颖", "主演", "一起飞"], ["林志 颖", "主演", "天庭外传"], ["林志颖", "主演", "一屋哨牙鬼"], ["天庭外传", "评论", "喜欢 这部 电影"], ["天庭外传", "评论", "没事 笑一笑 ， 青春 永不 老"], ["天庭外传", "评论", "930 ", " 只能 笑笑 了 。 观影 方式 ", " VCD"], ["天庭外传", "评论", "这片 是 暑假 看 的 枪版 VCD 啊啊啊"], ["一屋哨牙鬼", "评论", "当时 看 还是 蛮 搞笑 的"], ["一屋哨牙鬼", "评论", "要是 提前 个 二十年 看 应该 能 感觉 好笑 吧 ， 现在 实在 是 看不下去 了 ！"], ["一屋哨牙鬼", "评论", "只是 为了 看 一下 这些 知名演员 的 年轻 时代   至于 剧情 不敢恭维"], ["一屋哨牙鬼", "评论", "大 烂片 啊 ， 要不是 给 那么 多 明星 面子 ， 肯定 一分 都 不 给"], ["一屋哨牙鬼", "评论", "怀念 那个 随随便便 都 能 出 不过 不失 作品 的 香港电影 黄金 年代 。"], ["一屋哨牙鬼", "国家地区", "中国香港"], ["一屋哨牙鬼", "导演", "曹建南   曾志伟"], ["一屋哨牙鬼", "类型", "恐怖   喜剧"], ["一屋哨牙鬼", "主演", "林志颖   朱茵   张卫健"], ["一屋哨牙鬼", "口 碑", "口碑 还 可以"], ["一屋哨牙鬼", "评分", "6.2"]], "user_profile": ["{\"姓名\": \"张晓佳\", \"性别\": \"女\", \"居住地\": \"保定\", \"年龄区间\": \"小于18\", \"职业状态\": \"学生\", \"喜欢 的 明星\": [\"林志颖\"], \"喜欢 的 电影\": [\"一起飞\"], \"喜欢 的 新闻\": [\"林志颖 的新闻\"], \"同意 的 美食\": \" 麻辣烫\", \"同意 的 poi\": \" 金权道韩式自助烤肉火锅\", \"拒绝\": [\"音乐\"], \"接受 的 电影\": [], \"没有接受 的 电影\": []}"]}'
    # print(m.predict(s))
    # print(m.predict(s2))
    print(m.predict(s4))


if __name__ == '__main__':
    test()