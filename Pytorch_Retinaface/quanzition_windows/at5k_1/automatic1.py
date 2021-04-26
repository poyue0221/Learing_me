import os
import pdb
from write_file import create_retina

f = open("E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model_txt/model.c")  # 返回一个文件对象
line = f.readline()  # 调用文件的 readline()方法
mem_p  = []
mem_pn = []
f1 = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model_txt/retinaface_model.c', 'w')
f1.close()

while line:
    if '\n' == line:
        line = f.readline()
        continue
    if '//layer' in line:
        list1 = line.split(',')
        current_l_ind = int(line[7:9])
        line = f.readline()
        pass
    elif 'CONV' in line:
        if current_l_ind == 1:
            list1 = line.split(',')
            in_layer = int(list1[26][2:])
            out_size = int(list1[50][2:])
            out_size_next = int(list1[51])
            out_in = 0
            out_out = out_size
            mem_p.append([out_in, out_out])
            mem_pn.append(out_size_next+out_out)
            list1[42] = '{' + str(out_in)
            list1[50] = '{' + str(out_out)
            list1[51] = str(0)
            line = f.readline()
        else:
            list1 = line.split(',')
            in_layer = int(list1[26][2:])
            out_size = int(list1[50][2:])
            out_size_next = int(list1[51])
            out_in  = mem_p[in_layer-1][1]
            out_out = mem_p[-1][1] + out_size
            list1[50] = '{' + str(mem_pn[-1])
            mem_p.append([out_in, mem_pn[-1]])
            mem_pn.append(out_size_next + mem_pn[-1])
            list1[42] = '{'+str(out_in)
            list1[51] = str(0)
            line = f.readline()
    else: #other layers
        if 'RELU' in line: #RELU
            list1 = line.split(',')
            in_layer = int(list1[28][2:])
            out_size = int(list1[52][2:])
            out_size_next = int(list1[53])
            out_in  = mem_p[in_layer-1][1]
            out_out = mem_p[-1][1] + out_size
            list1[52] = '{'+str(mem_pn[-1])
            mem_p.append([out_in, mem_pn[-1]])
            mem_pn.append(out_size_next + mem_pn[-1])
            list1[44] = str(out_in)
            for i in range(5) :
                if int(list1[29+i][1:]) != 0:
                    in_layer = int(list1[29+i][1:])
                    out_in = mem_p[in_layer - 1][1]
                    list1[45+i] = str(out_in)
                else:
                    break
            list1[44] = '{'+list1[44]
            list1[53] = str(0)
            line = f.readline()
        else: #UPSAMPLING
            list1 = line.split(',')
            in_layer = int(list1[22][2:])
            output_layer = int(list1[30][2:])
            out_size = int(list1[38][2:])
            out_size_next = int(list1[39])
            out_in  = mem_p[in_layer-1][1]
            list1[38] = '{'+str(mem_pn[-1])
            out_out = mem_p[-1][1] + out_size
            mem_p.append([out_in, mem_pn[-1]])
            mem_pn.append(out_size_next + mem_pn[-1])
            list1[30] = '{'+str(out_in)
            list1[39] = str(0)
            line = f.readline()
    # weight idx ready
    aaa = ''
    for i in list1:
        aaa += i + ','
    create_retina(aaa[:-2])
f.close()