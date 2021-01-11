# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/12 上午1:10

import multiprocessing
import time

def work():
    for i in range(10):
        print("工作中。。。")
        time.sleep(0.2)

if __name__=='__main__':
    #设置守护主进程，主进程结束所有子进程都结束
    work_process = multiprocessing.Process(target=work,daemon=True)
    #or：
    # work_process = multiprocessing.Process(target=work)
    # work_process.daemon = True
    work_process.start()
    time.sleep(1)
    print("主进程执行完毕")
