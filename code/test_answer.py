#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import print_function
"""
@Author    :apple.li
@Time      :2020/6/4 13:39
@File      :test_answer.py
@Desc      :
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


def main(file_path, out_path):
    """
    main
    """
    import numpy as np

    all_lines = [line for line in open(file_path, encoding='utf-8')]
    all_lines = np.random.choice(all_lines, size=200, replace=False)
    with open(out_path, encoding='utf-8', mode='w') as fw:
        for line in all_lines:
            sample = json.loads(line, encoding="utf-8")
            history = sample['history']
            response = conversation_client(json.dumps(sample, ensure_ascii=False))
            history.append(response)
            fw.writelines('\n'.join(history) + '\n\n')


if __name__ == '__main__':
    main('../test_1.txt', '../test_1_sample.txt')
    main('../test_2.txt', '../test_2_sample.txt')
