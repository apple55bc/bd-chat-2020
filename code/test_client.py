#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import print_function
"""
@Author    :apple.li
@Time      :2020/5/21 17:52
@File      :test_client.py
@Desc      :File: conversation_client.py
"""
import sys
import socket
import importlib
import json
import re

importlib.reload(sys)

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8601
goal_comp = re.compile('\[\d+\]\s*[^(]*')



def conversation_client(text):
    """
    conversation_client
    """
    mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysocket.connect((SERVER_IP, SERVER_PORT))
    mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096 * 5)
    mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096 * 5)

    mysocket.sendall(text.encode())
    result = mysocket.recv(4096 * 5).decode()

    mysocket.close()

    return result


def main():
    """
    main
    """
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " eval_file")
        exit()

    skip_n = 0
    for line in open(sys.argv[1], encoding='utf-8'):
        if skip_n > 0:
            skip_n -= 1
            continue
        sample = json.loads(line, encoding="utf-8")
        # all_goals = goal_comp.findall(sample['goal'])
        # all_goals = [[s[3:].strip(), '', ''] for s in all_goals]
        # sample['goal'] = all_goals
        print('\n\n' + '=' * 20)
        print('New goal: ', sample['goal'])
        history = sample['history']

        print('Ori history:' + '\n'.join(history))

        s = input('E ：bot first; U: user first; C:continue; Z: quit; 其它：继续原有的样本对话历史\n')
        if s in ['E', 'e']:
            history = []
        elif s in ['U', 'u']:
            s = input('Q:')
            history = [s]
        elif s in ['Z', 'z']:
            return
        elif s in ['continue', 'c', 'C']:
            continue
        else:
            try:
                s = int(s)
                skip_n = s
                print('Input: {}. Skip {} times'.format(s, skip_n))
                continue
            except ValueError:
                print('Input: {}. Process dialogue'.format(s))
                pass
        sample['history'] = history

        response = conversation_client(json.dumps(sample, ensure_ascii=False))
        print('A: ', response)
        while True:
            question = input('Q:')
            if question in ['continue', 'c', 'C']:
                break
            if question in ['q', 'Q', 'Z', 'z']:
                return
            history.append(response)
            history.append(question)
            sample['history'] = history
            response = conversation_client(json.dumps(sample, ensure_ascii=False))
            print('A: ', response)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
