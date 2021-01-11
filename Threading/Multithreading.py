# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/12 下午12:35
import threading
import time

def sing():
    for i in range(3):
        print("唱歌。。。")
        time.sleep(1)

def dance():
    for i in range(3):
        print("跳舞。。。")
        time.sleep(1)

if __name__ == "__main__":
    sing_thread = threading.Thread(target=sing)
    dance_thread = threading.Thread(target=dance)
    # dance_thread.setDaemon(True)  与process不同的
    sing_thread.start()
    dance_thread.start()
