import os
import struct
import numpy
import pdb

def create_final(string): # parameter
    f = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model/model.c', 'a')
    f.write(string)
    f.write('\n')
    f.close()

def create_retina(string): # parameter
    f = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model/retinaface_model.c', 'a')
    if string[0] == '{':
        string = string[1:]
    f.write(string)
    f.write('\n')
    f.close()

def write_weight_b(weight, bias): # parameter
    f = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model/retinaface_weight.b', 'ab+')
    #pdb.set_trace()
    # print(bias)
    # z = open("./bias")
    # z.write(bias)
    # z.close()
    for x in weight:
        s = struct.pack('b', int(x))
        f.write(s)

    for x in bias:
        if x > 2**16:
            x = 2**16
        elif x < -2**16:
            x = -2**16
        s = struct.pack('i', int(x))
        f.write(s)
    f.close()