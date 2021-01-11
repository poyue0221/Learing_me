# -* encoding: utf-8 -*-
# @Author : Allen
# @Time : 2021/1/8 下午1:34

import threading
import time

def task():
    time.sleep(1)
    #获取当前线程的线程对象
    current= threading.current_thread()
    print(current)

if __name__=="__main__":
    for i in range(5):
        task_thread = threading.Thread(target=task)
        task_thread.start()

#结论：多线程之间执行是无序的，由cpu调度的