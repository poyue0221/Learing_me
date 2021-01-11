# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/11 下午11:53

#1,导入进程包
#2,指定进程任务
#3,开始进程

#1,导入进程包
import multiprocessing
import time

#唱歌
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(0.5)
#跳舞
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(0.5)

if __name__=='__main__':
    # 2,使用进程创建进程对象
      #target进程函数名
    sing_process = multiprocessing.Process(target=sing)
    dance_process = multiprocessing.Process(target=dance)
    # 3,使用进程对象启动进程执行指定任务
    sing_process.start()
    dance_process.start()

