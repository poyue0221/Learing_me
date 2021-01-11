# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2020/12/12 上午12:23

#1,导入进程包
#2,指定进程任务
#3,开始进程

#1,导入进程包
import multiprocessing
import time

#唱歌
def sing(num,name):
    for i in range(num):
        print(name)
        print("唱歌")
        time.sleep(1)
#跳舞
def dance(num,name):
    for i in range(num):
        print(name)
        print("跳舞")
        time.sleep(1)

if __name__ == '__main__':
    # 2,使用进程创建进程对象
        #target进程函数名
        #args 使用元组方式给指定任务按顺序传参
        #kwargs 使用字典方式给指定任务传惨（key和参数名一定要保持一致）
    # sing_process = multiprocessing.Process(target=sing,args=(3,)) 单个参数用逗号隔开
    sing_process = multiprocessing.Process(target=sing, args=(3, "小明"))
    dance_process = multiprocessing.Process(target=dance, kwargs={"name": "小红", 'num': 2})
    # 3,使用进程对象启动进程执行指定任务
    sing_process.start()
    dance_process.start()

