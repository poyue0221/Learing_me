# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/12 上午12:48

import multiprocessing
import time
import os

#唱歌
def sing():
    print("唱歌子进程的编号", os.getpid())
    print("唱歌父进程的编号", os.getppid())
    for i in range(3):
        print("唱歌")
        time.sleep(0.5)
#跳舞
def dance():
    print("跳舞子进程的编号", os.getpid())
    print("跳舞父进程的编号", os.getppid())
    for i in range(3):
        print("跳舞")
        time.sleep(0.5)

if __name__=='__main__':   #（2,3步骤为主进程  其他为子进程）
    print("当前主进程的编号", os.getpid())
    # 2,使用进程创建进程对象
      #target进程函数名
    sing_process = multiprocessing.Process(target=sing)
    dance_process = multiprocessing.Process(target=dance)
    # 3,使用进程对象启动进程执行指定任务
    sing_process.start()
    dance_process.start()