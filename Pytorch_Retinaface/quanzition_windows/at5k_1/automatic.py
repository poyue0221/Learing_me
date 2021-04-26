import os
import pdb
from write_file import create_final
f = open("E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model_txt/model.txt")  # 返回一个文件对象

line = f.readline()  # 调用文件的 readline()方法
weight_p = 0
layer_last = '0'
layer_next = '1'
current_l_ind = 0
mem_p = []
mem_p.append(0)
f1 = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model/model.c', 'w')
f1.close()
while line:
    if '//layer' in line:
        current_l_ind = int(line[7:9])
        create_final(line)
        line = f.readline()
        pass
    elif 'CONV' in line:
        list1 = line.split(',')
        # weight idx ready
        weight_size = list1[11]


        list1[11] = str(weight_p)
        weight_p += int(weight_size)
        # layer idx
        # 这里如果有条约结构的话，当前是需要自己去手动改；
        input_layer  = current_l_ind - 1
        output_layer = current_l_ind + 1
        list1[26] = '{' + str(input_layer)
        list1[34] = '{' + str(output_layer)
        # mem_p 这个文本里面还没有做，下一个根据index完成的结果进行更改计算；
        #替换好的文本写入新的脚本中；
        aaa = ''
        for i in list1:
            aaa += i + ', '
        create_final(aaa[:-2])
        line = f.readline()
    else:
        aaa = ''
        list1 = line.split(',')
        for i in list1:
            aaa += i + ', '
        create_final(aaa[:-2])
        line = f.readline()

f.close()
