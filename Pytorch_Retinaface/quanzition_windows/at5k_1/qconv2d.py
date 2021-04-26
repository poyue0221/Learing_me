import numpy as np
import pdb
import torch.nn as nn
import torch
import time
from at5k_1.write_file import write_weight_b


def wr_str(str): # parameter
    f = open('E:/AT5000/AT5K_Pytorch_2/Pytorch_Retinaface_master/at5k_1/model_txt/model.txt', 'a')
    f.write(str)
    f.write('\n')
    f.close()


class cal_l:
  def __init__(self, n=1):
    self.n = n
  def __call__(self):
    last_idx = self.n
    self.n += 1
    next_idx = self.n
    return '//layer' + str(last_idx)

def serializeWeight(data, Csize=8, Ksize=8):
    result = []

    #
    # if data.shape[2] < Csize:
    #     temp = np.zeros((data.shape[0], data.shape[1], Csize, data.shape[3]))
    #     temp[:,:,0:data.shape[2],:] = data
    #     data = temp


    if len(data.shape) == 4:
        # for the conv weight
        F1, F2, C, N = data.shape
    elif len(data.shape) == 1:
        # for the conv bias
        N = data.shape[0]
        result.extend(data)
        if N % Ksize:
            result.extend(np.zeros(Ksize - (N % Ksize), dtype=np.int8))
        return result
    else:
        print('Weight shape not correct')
        return None
    flag = False
    if C ==1:
        data.permute(0,1,3,2)
        F1, F2, C, N = data.shape
        flag = True
    Cleft = C
    Nleft = N
    Cidx = 0
    Nidx = 0
    CC = Csize

    while Nleft > 0:
        if Nleft < Ksize:
            KK = Nleft
        else:
            KK = Ksize

        Cleft = C
        Cidx = 0
        while Cleft > 0:
            if Cleft > Csize:
                CC = Csize
            else:
                CC = Cleft
            for f1 in range(F1):
                for f2 in range(F2):
                    for k in range(KK):
                        # result.extend(np.int8(data[f1, f2, Cidx:Cidx + CC, k + Nidx]))
                        result.extend(data[f1, f2, Cidx:Cidx + CC, k + Nidx])
                        # print('offset:', f1, f2, Cidx, k+Nidx)
                        if flag == True:
                            pass
                        else:
                            if CC < Csize:
                                result.extend(np.zeros(Csize - CC, dtype=np.int8))
                    if (KK < Ksize):
                        result.extend(np.zeros((Ksize - KK) * Csize, dtype=np.int8))

            Cleft -= Csize
            Cidx += CC

        Nleft = Nleft - Ksize
        Nidx += KK

    return result

aaa = cal_l()

def cal_layer():
    string = aaa()
    wr_str(string)

def shift(M):
    for i in range(-32, 32):
        if M >= 2 ** (-i):
            return 2 ** (-i)
            break

def add_scale(M):
    for i in range(-32,32):
        if M >= 2 ** (-i):
            if np.abs( M - 2**(-i) ) > np.abs( M - 2**(-i+1) ): # 上一个
                return 2**(-i+1)
            else:
                return 2**(-i)

def M2scale(M):
    thre = 0.1
    for i in range(-32,32):
        if M >= 2 ** (-i):
            if np.abs( M - 2**(-i) ) > np.abs( M - 2**(-i+1) ): # 上一个
                if ( np.abs(2 ** (-i+1) -M) / M > thre ) :
                    aaa1 = shift(M)
                    aaa2 = shift(M- aaa1)
                    aaa3 = shift(M -aaa1 -aaa2)
                    if (np.abs( M - aaa1 -aaa2-aaa3 ) > np.abs( M - 2**(-i+1) )):
                        ll = []
                        ll.append(2**(-i+1))
                        ll.append(False)
                        return ll  # loss 小
                    else:
                        ll = []
                        ll.append( aaa1 + aaa2 + aaa3 )
                        ll.append(True)
                        return ll  # loss 小
                else:
                    ll = []
                    ll.append(2 ** ( -i + 1))
                    ll.append(False)
                    return ll  # loss 小


            else:
                if ( np.abs(2 ** (-i) -M) / M > thre ) :
                    aaa1 = shift(M)
                    aaa2 = shift(M- aaa1)
                    aaa3 = shift(M -aaa1 -aaa2)
                    if (np.abs( M - aaa1 -aaa2-aaa3 ) > np.abs( M - 2**(-i) )):
                        ll = []
                        ll.append(2**(-i))
                        ll.append(False)
                        return ll  # loss 小
                    else:
                        ll = []
                        ll.append( aaa1 + aaa2 + aaa3 )
                        ll.append(True)
                        return ll  # loss 小
                else:
                    ll = []
                    ll.append( 2 ** ( -i) )
                    ll.append(False)
                    return ll  # loss 小
            break

def M2scale_TSME(M):
    for i in range(-32,0):
        if M <= 2 ** (i):
            shift1 = i-1
            cpu2 = np.floor((M / 2**(i-1)) * 2 **(6))
            print(shift1, 2**(i-1),cpu2)
            return -1*shift1, 2**(i-1), cpu2

def conv_jit(input, weight_int, bias_float, weight, bias, self_M, inp, oup, stride, padding, groups, scale, zp):
    aaa = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    aaa.weight = torch.nn.Parameter(torch.tensor(np.array(weight).astype(np.float32)))
    #aaa.bias = torch.nn.Parameter(torch.zeros_like(torch.tensor(np.array(bias).astype(np.float32))))
    #aaa.bias = torch.nn.Parameter(torch.tensor(np.array(bias).astype(np.float32)))
    aaa.bias = torch.nn.Parameter(torch.tensor(np.array(bias).astype(np.float32)))
    input_ = np.array(input.int_repr()*1.0).astype(np.float32) - np.array(input.q_zero_point()).astype(np.float32)
    res = aaa( torch.tensor(input_))
    res = res.detach().numpy()
    M =  M2scale(self_M)
    #if ( (self_M - M[0])/self_M)  >  0.1:
    #    pdb.set_trace()
    #print('##################', M[0], self_M, (self_M - M[0])/self_M, (self_M - M[0])/M[0] )
    #if M[1]:
    #    print('$$$$$$$$$$$$$ counting $$$$$$$$$$$$$$')
    #res = res * M[0]
    res = res * self_M
    res = np.floor(res)
    res = res + zp  # add ly
    #res[res>120] = 0
    res = np.floor(np.clip(res, -128, 127))

    #对齐格式
    res = res.astype(np.float32)
    res = torch.tensor(scale * (res - zp))
    res1 = torch.quantize_per_tensor(res ,scale = scale , zero_point = zp, dtype = torch.qint8)

    # float
    aaa = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    input_ = (np.array(input.int_repr()).astype(np.float32) - input.q_zero_point())*input.q_scale()
    weight_ = (np.array(weight_int.int_repr()).astype(np.float32)- weight_int.q_zero_point()).astype(np.float32) * weight_int.q_scale()
    aaa.weight = torch.nn.Parameter(torch.tensor(np.array(weight_)))
    #aaa.bias = torch.nn.Parameter(torch.zeros_like(torch.tensor(np.array(bias_float))))
    aaa.bias = torch.nn.Parameter(torch.tensor(np.array(bias_float)))
    res = aaa(torch.tensor(input_))
    res = res.detach().numpy()
    #对齐格式
    res = torch.tensor(res)
    res2 = torch.quantize_per_tensor(res, scale = scale , zero_point = zp, dtype = torch.qint8)
    #finish

    err = (np.array(res1.int_repr()).astype(np.float32) - np.array(res2.int_repr()).astype(np.float32)).max()
    #err1 = (ori - np.array(res2.int_repr()).astype(np.float32) ).max()
    return res1, res2

def conv_jit2(input, weight_int, bias_float, weight, bias, self_M, inp, oup, stride, padding, groups, scale, zp):
    aaa = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    aaa.weight = torch.nn.Parameter(torch.tensor(np.array(weight).astype(np.float32)))
    aaa.bias = torch.nn.Parameter(torch.tensor(np.array(bias).astype(np.float32)))
    input_ = np.array(input.int_repr()*1.0).astype(np.float32)  - np.array(input.q_zero_point()).astype(np.float32)
    res = aaa( torch.tensor(input_))

    bbb = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    bbb.weight = torch.nn.Parameter(torch.tensor(np.array(weight).astype(np.float32)))
    # aaa.bias = torch.nn.Parameter(torch.zeros_like(torch.tensor(np.array(bias).astype(np.float32))))
    # aaa.bias = torch.nn.Parameter(torch.tensor(np.array(bias).astype(np.float32)))
    bbb.bias = torch.nn.Parameter(torch.tensor(np.array(bias).astype(np.float32)))
    input_ = np.array(input.int_repr() * 1.0).astype(np.float32)# - np.array(input.q_zero_point()).astype(np.float32)
    # input_ = np.array(input.int_repr() ).astype(np.float32)
    res_b = bbb(torch.tensor(input_))
    diff = res.detach().numpy() - res_b.detach().numpy()
    '''
    #res是直接的卷积输出，res_b是 input和weight的卷积
    ccc = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    ccc.weight = torch.nn.Parameter(torch.tensor(np.array(weight).astype(np.float32)))
    ccc.bias = torch.nn.Parameter(torch.tensor(np.array(bias+diff[0,:,10,10]).astype(np.float32)))
    input_ = np.array(input.int_repr()*1.0).astype(np.float32) # - np.array(input.q_zero_point()).astype(np.float32)
    #input_ = np.array(input.int_repr() ).astype(np.float32)
    res_c = ccc( torch.tensor(input_))
    '''
    ddd = nn.Conv2d(inp, oup, 3, stride, padding, groups=groups, bias=True)
    ddd.weight = torch.nn.Parameter(torch.tensor(np.array(weight).astype(np.float32)))
    ddd.bias = torch.nn.Parameter(torch.tensor(np.array(np.floor(bias+diff[0,:,3,3]+zp/self_M)).astype(np.float32))) #
    input_ = np.array(input.int_repr()*1.0).astype(np.float32)
    res_d = ddd( torch.tensor(input_))
    res_d = res_d.detach().numpy()
    shift1, shift_cal, cpu2 = M2scale_TSME(self_M)
    res_d = np.floor( res_d * shift_cal  )
    res_d = np.floor(res_d)
    res_d = res_d * cpu2/2**(6)
    res_d = np.floor(res_d)
    res = res.detach().numpy()
    #res = res * M[0]
    res = res * self_M
    res = np.floor(res)
    res = res  + zp  # add ly
    #res[res>120] = 0
    res = np.floor(np.clip(res, -128, 127))
    res_d = np.floor(np.clip(res_d, -128, 127))

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

    t = time.time()
    now = str(int(round(t * 1000000)))
    np.save('./layer_output_time/' + 'm' + now + '_conv_0_layer_weight_int8' + '.npy', weight)
    np.save('./layer_output_time/' + 'm' + now + '_conv_1_layer_weight_scale' + '.npy', weight_int.q_scale())
    np.save('./layer_output_time/' + 'm' + now + '_conv_2_layer_weight_zero_point' + '.npy', weight_int.q_zero_point())
    np.save('./layer_output_time/' + 'm' + now + '_conv_3_layer_input_int8' + '.npy', input_)
    np.save('./layer_output_time/' + 'm' + now + '_conv_4_layer_input_scale' + '.npy', input.q_scale())
    np.save('./layer_output_time/' + 'm' + now + '_conv_5_layer_input_zero_point' + '.npy', input.q_zero_point())
    np.save('./layer_output_time/' + 'm' + now + '_conv_6_layer_self_M_shift' + '.npy', shift1)
    np.save('./layer_output_time/' + 'm' + now + '_conv_7_layer_self_M_cal' + '.npy', shift_cal)
    np.save('./layer_output_time/' + 'm' + now + '_conv_8_layer_self_M_cpu' + '.npy', cpu2)
    np.save('./layer_output_time/' + 'm' + now + '_conv_9_layer_bias_int32' + '.npy', np.floor(bias+diff[0,:,3,3]+zp/self_M))
    np.save('./layer_output_time/' + 'm' + now + '_conv_A_layer_output_int8' + '.npy', res_d)
    np.save('./layer_output_time/' + 'm' + now + '_conv_B_layer_out_scale' + '.npy', scale)
    np.save('./layer_output_time/' + 'm' + now + '_conv_C_layer_out_zero_point' + '.npy', zp)
    np.save('./layer_output_time/' + 'm' + now + '_conv_D_layer_stride' + '.npy', stride)
    weight_ser = serializeWeight(weight.permute(2,3,1,0))

    write_weight_b(weight_ser, np.floor(bias+diff[0,:,3,3]+zp/self_M))


    #对齐格式
    res = res.astype(np.float32)
    res = torch.tensor(scale * (res - zp))
    res1 = torch.quantize_per_tensor(res ,scale = scale , zero_point = zp, dtype = torch.qint8)
    res_d = res_d.astype(np.float32)
    res_d = torch.tensor(scale * (res_d - zp))
    res_d = torch.quantize_per_tensor(res_d, scale=scale, zero_point=zp, dtype=torch.qint8)
    return res_d