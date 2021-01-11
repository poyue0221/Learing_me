# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/12 上午1:02

import multiprocessing
import time

def work():
    for i in range(10):
        print("工作中。。。")
        time.sleep(0.2)

if __name__=='__main__':
    work_process = multiprocessing.Process(target=work)
    #主进程等待子进程结束再结束
    work_process.start()
    time.sleep(1)
    print("主进程执行完毕")
