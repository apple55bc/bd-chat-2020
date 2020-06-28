#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_server.py
"""

from __future__ import print_function

import sys
sys.path.append("../")
import socket
import importlib
from _thread import start_new_thread
from predict.predict_final import FinalPredict

importlib.reload(sys)

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8601

print("starting conversation server ...")
print("binding socket ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096 * 20)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096 * 20)
bufsize = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print( "Buffer size [After]: %d" %bufsize)
#Bind socket to local host and port
try:
    s.bind((SERVER_IP, SERVER_PORT))
except socket.error as msg:
    print("Bind failed. Error Code : " + str(msg[0]) + " Message " + msg[1])
    exit()
#Start listening on socket
s.listen(10)
print("bind socket success !")

print("loading model...")
model = FinalPredict()
print("load model success !")

print("start conversation server success !")


def clientthread(conn, addr):
    """
    client thread
    """
    logstr = "addr:" + addr[0]+ "_" + str(addr[1])
    try:
        #Receiving from client
        param_ori = conn.recv(4096 * 20)
        param = param_ori.decode('utf-8', "ignore")
        # logstr += "\tparam:" + param
        if param is not None:
            response = model.predict(param.strip())
            logstr += "\tresponse:" + response 
            conn.sendall(response.encode())
        conn.close()
        print(logstr + "\n")
    except Exception as e:
        print(logstr + "\n", e)
        print('==========')
        conn.close()
        raise


while True:
    conn, addr = s.accept()
    start_new_thread(clientthread, (conn, addr))
s.close()
