#import time
import torch
import numpy as np
import torch.nn as nn
#import torchvision.models._utils as _utils
#import torchvision.models as models
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional
# from qconv2d import wr_str
#from torch.quantization import QuantStub, DeQuantStub
#from torch.autograd import Variable
import pdb
# from serializeData import serializeData
# from serializeWeight import serializeWeight


def conv_bn(inp, oup, stride = 1, name = None):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def conv_dw_1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU()
    )

def conv_dw_2(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel): # 输入是：64,64
        super(SSH, self).__init__()
        #self.quant = QuantStub()
        assert out_channel % 4 == 0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1) # 64 --> 32

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1) # 64 --> 16
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1) # 16 --> 16

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1) # 16 --> 16
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1) # 16 --> 16

        self.f_add = FloatFunctional()

    def forward(self, input):
        #wr_str('//SSH 3X3')
        conv3X3 = self.conv3X3(input)
        #wr_str('//SSH 5X5_1')
        conv5X5_1 = self.conv5X5_1(input)
        #wr_str('//SSH 5X5')
        conv5X5 = self.conv5X5_2(conv5X5_1)
        #wr_str('//SSH 7X7_2')
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        #wr_str('//SSH 7X7')
        conv7X7 = self.conv7x7_3(conv7X7_2)
        #wr_str('//SSH cat')

        out = self.f_add.cat([conv3X3, conv5X5, conv7X7], dim=1) #小幺蛾子？
        # try:
        #     aaa= np.array(conv3X3)[0, 0, 0, 0]
        # except:
        #     AC = 8
        #     if out.shape[1] < AC:
        #         out_shape1 = AC
        #     else:
        #         out_shape1 = out.shape[1]
        #     wr_str('{LY_CAT_RELU,{ %d,%d,%d,0,0,0,0,0,  %d,%d,%d,0,0,0,0,0 ,%d, %d,%d,%d,0,0,0,0,0 },1,1,{input_idx,0,0,0,0,0,0,0},{output_idx,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{%d,%d,0,0,0,0,0,0}}' % ( np.floor(conv3X3.q_scale()*(2**6)/out.q_scale()), np.floor(conv5X5.q_scale()*(2**6)/out.q_scale()),
        #         np.floor(conv7X7.q_scale()*(2**6)/out.q_scale()), conv3X3.q_zero_point(), conv5X5.q_zero_point(), conv7X7.q_zero_point(), out.q_zero_point(),
        #         conv3X3.shape[1]*conv3X3.shape[2]*conv3X3.shape[3], conv5X5.shape[1]*conv5X5.shape[2]*conv5X5.shape[3],
        #         conv7X7.shape[1]*conv7X7.shape[2]*conv7X7.shape[3], out_shape1*out.shape[2]*out.shape[3],out_shape1*out.shape[2]*out.shape[3]))
        # # 第一个循环： [1, 32, 34, 60], [1, 16, 34, 60], [1, 16, 34, 60]
        # # 第二个循环： [1, 32, 17, 30], [1, 16, 17, 30], [1, 16, 17, 30]
        # # 第三个循环： [1, 32,  9, 15], [1, 16,  9, 15], [1, 16,  9, 15]
        # #wr_str('//SSH relu')
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()  #out channels:64
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)
        self.f_add = FloatFunctional()

    def forward(self, input):
        #input = list(input.values())
        #wr_str('//fpn output1')
        output1 = self.output1(input[0]) # input[0].shape = [1,64,34,60]
        # output1.shape = [1,64,34,60]
        #wr_str('//fpn output2')
        output2 = self.output2(input[1]) # input[1].shape = [1,128,17,30]
        # output2.shape = [1,64,17,30]
        #wr_str('//fpn output3')
        output3 = self.output3(input[2]) # input[2].shape = [1,256,9,15]
        # output3.shape = [1,64,9,15]
        #wr_str('//fpn sampling1')
        #没有scale zp的变化
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        # up3.shape = [1,64,17,30]
        #wr_str('//fpn add')
        temp = output2
        output2 = self.f_add.add(output2, up3)
        # try:
        #     aaa= np.array(temp)[0, 0, 0, 0]
        # except:
        #     AC = 8
        #     if output2.shape[1] < AC:
        #         out_shape1 = AC
        #     else:
        #         out_shape1 = output2.shape[1]
        #     if up3.shape[1] < AC:
        #         up3_shape1 = AC
        #     else:
        #         up3_shape1 = up3.shape[1]
        #     wr_str('{LY_UPSAMPLING_ADD,{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}},1,1,{input_idx,0,0,0,0,0,0,0},{output_idx,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{%d,%d,0,0,0,0,0,0}}' %(
        #         np.floor(up3.q_scale() * (2 ** 6) / output2.q_scale()),
        #         np.floor(temp.q_scale() * (2 ** 6) / output2.q_scale()),
        #         temp.q_zero_point(), up3.q_zero_point(), output2.q_zero_point(),
        #         output3.shape[2], output3.shape[3], output3.shape[1],
        #         output2.shape[2], output2.shape[3], output2.shape[1], out_shape1*output2.shape[2]*output2.shape[3], up3_shape1*up3.shape[2]*up3.shape[3]))
        # #wr_str('//fpn merge2')
        output2 = self.merge2(output2)
        # output2.shape = [1,64,17,30]
        #wr_str('//fpn sampling2')
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") #上采样和下采样专用
        # up2.shape = [1,64,34,60]
        #wr_str('//fpn add')
        temp = output1
        output1 = self.f_add.add(output1, up2)
        try:
            aaa= np.array(temp)[0, 0, 0, 0]
        except:
            AC = 8
            if output1.shape[1] < AC:
                out_shape1 = AC
            else:
                out_shape1 = output1.shape[1]
            if up2.shape[1] < AC:
                up2_shape1 = AC
            else:
                up2_shape1 = up2.shape[1]
            wr_str('{LY_UPSAMPLING_ADD,{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}},1,1,{input_idx,0,0,0,0,0,0,0},{output_idx,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{%d,%d,0,0,0,0,0,0}}' %(
            np.floor(up2.q_scale() * (2 ** 6) / output1.q_scale()),
            np.floor(temp.q_scale() * (2 ** 6) / output1.q_scale()),
            temp.q_zero_point(), up2.q_zero_point(), output1.q_zero_point(),
            output2.shape[2], output2.shape[3], output2.shape[1],
            output1.shape[2], output1.shape[3], output1.shape[1], out_shape1*output1.shape[2]*output1.shape[3], up2_shape1*up2.shape[2]*up2.shape[3]))

            # #wr_str('{LY_UPSAMPLING_ADD,{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}},' % (
            # np.floor(up2.q_scale() * (2 ** 6) / output2.q_scale()),
            # np.floor(temp.q_scale() * (2 ** 6) / output2.q_scale()),
            # temp.q_zero_point(), up3.q_zero_point(), output2.q_zero_point(),
            # output2.shape[2], output2.shape[3], output2.shape[1],
            # output1.shape[2], output1.shape[3], output1.shape[1]))
        # up2.shape = [1,64,34,60]
        #wr_str('//fpn merge1')
        output1 = self.merge1(output1)
        # output1.shape = [1,64,34,60]
        #wr_str('//fpn put together')
        out = [output1, output2, output3]
        # output1.shape = [1, 64, 34, 60]
        # output2.shape = [1, 64, 17, 30]
        # output3.shape = [1, 64,  9, 15]
        #out = self.dequant(out)
        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1_0 = nn.Sequential(conv_bn(3, 8, 2))      # 3
        self.stage1_1 = nn.Sequential(conv_dw_1(8, 16, 1))   # 7                  1
        self.stage1_2 = nn.Sequential(conv_dw_2(8, 16, 1))   # 7                  1
        self.stage1_3 = nn.Sequential(conv_dw_1(16, 32, 2))  # 11                 2
        self.stage1_4 = nn.Sequential(conv_dw_2(16, 32, 2))  # 11                 2
        self.stage1_5 = nn.Sequential(conv_dw_1(32, 32, 1))  # 19                 3
        self.stage1_6 = nn.Sequential(conv_dw_2(32, 32, 1))  # 19                 3
        self.stage1_7 = nn.Sequential(conv_dw_1(32, 64, 2))  # 27                 4
        self.stage1_8 = nn.Sequential(conv_dw_2(32, 64, 2))  # 27                 4
        self.stage1_9 = nn.Sequential(conv_dw_1(64, 64, 1))  # 43                 5
        self.stage1_10 = nn.Sequential(conv_dw_2(64, 64, 1)) # 43                 5

        self.stage2_0 = nn.Sequential(conv_dw_1(64, 128, 2))  # 43 + 16 = 59      1
        self.stage2_1 = nn.Sequential(conv_dw_2(64, 128, 2))  # 43 + 16 = 59      1
        self.stage2_2 = nn.Sequential(conv_dw_1(128, 128, 1)) # 59 + 32 = 91      2
        self.stage2_3 = nn.Sequential(conv_dw_2(128, 128, 1)) # 59 + 32 = 91      2
        self.stage2_4 = nn.Sequential(conv_dw_1(128, 128, 1)) # 91 + 32 = 123     3
        self.stage2_5 = nn.Sequential(conv_dw_2(128, 128, 1)) # 91 + 32 = 123     3
        self.stage2_6 = nn.Sequential(conv_dw_1(128, 128, 1)) # 123 + 32 = 155    4
        self.stage2_7 = nn.Sequential(conv_dw_2(128, 128, 1)) # 123 + 32 = 155    4
        self.stage2_8 = nn.Sequential(conv_dw_1(128, 128, 1)) # 155 + 32 = 187    5
        self.stage2_9 = nn.Sequential(conv_dw_2(128, 128, 1)) # 155 + 32 = 187    5
        self.stage2_10 = nn.Sequential(conv_dw_1(128, 128, 1)) # 187 + 32 = 219    6
        self.stage2_11 = nn.Sequential(conv_dw_2(128, 128, 1)) # 187 + 32 = 219    6

        self.stage3_0 = nn.Sequential(conv_dw_1(128, 256, 2)) # 219 + 32 = 241    1
        self.stage3_1 = nn.Sequential(conv_dw_2(128, 256, 2)) # 219 + 32 = 241    1
        self.stage3_2 = nn.Sequential(conv_dw_1(256, 256, 1)) # 241 + 64 = 301    2
        self.stage3_3 = nn.Sequential(conv_dw_2(256, 256, 1)) # 241 + 64 = 301    2

    def forward(self, x):
        out=[]
        # [1,3,270,480]
        #wr_str('//stage1_0,conv2d')
        x1 = self.stage1_0(x)
        # [1,8,135,240]
        #wr_str('//stage1_1,depth_wise')
        x1 = self.stage1_1(x1)
        # [1,8,135,240]
        #wr_str('//stage1_2,conv2d')
        x1 = self.stage1_2(x1)
        # [1, 16, 135, 240]
        #wr_str('//stage1_3,depth_wise')
        x1 = self.stage1_3(x1)
        # [1, 16, 68, 120]
        #wr_str('//stage1_4,conv2d')
        x1 = self.stage1_4(x1)
        # [1, 32, 68, 120]
        #wr_str('//stage1_5,depth_wise')
        x1 = self.stage1_5(x1)
        # [1, 32, 68, 120]
        #wr_str('//stage1_6,conv2d')
        x1 = self.stage1_6(x1)
        # [1, 32, 68, 120]
        #wr_str('//stage1_7,depth_wise')
        x1 = self.stage1_7(x1)
        # [1, 32, 34, 60]
        #wr_str('//stage1_8,conv2d')
        x1 = self.stage1_8(x1)
        # [1, 64, 34, 60]
        #wr_str('//stage1_9,depth_wise')
        x1 = self.stage1_9(x1)
        # [1, 64, 34, 60]
        #wr_str('//stage1_10,conv2d')
        x1 = self.stage1_10(x1)
        # [1, 64, 34, 60]
        #wr_str('//stage2_0,depth_wise')
        x2 = self.stage2_0(x1)
        # [1, 64, 17, 30]
        #wr_str('//stage2_1,conv2d')
        x2 = self.stage2_1(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_2,depth_wise')
        x2 = self.stage2_2(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_3,conv2d')
        x2 = self.stage2_3(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_4,depth_wise')
        x2 = self.stage2_4(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_5,conv2d')
        x2 = self.stage2_5(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_6,depth_wise')
        x2 = self.stage2_6(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_8,conv2d')
        x2 = self.stage2_7(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage1_1,depth_wise')
        x2 = self.stage2_8(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_9,conv2d')
        x2 = self.stage2_9(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_10,depth_wise')
        x2 = self.stage2_10(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage2_11,conv2d')
        x2 = self.stage2_11(x2)
        # [1, 128, 17, 30]
        #wr_str('//stage3_0,depth_wise')
        x3 = self.stage3_0(x2)
        # [1, 128, 9, 15]
        #wr_str('//stage3_1,conv2d')
        x3 = self.stage3_1(x3)
        # [1, 256, 9, 15]
        #wr_str('//stage3_2,depth_wise')
        x3 = self.stage3_2(x3)
        # [1, 256, 9, 15]
        #wr_str('//stage3_3,conv2d')
        x3 = self.stage3_3(x3)
        # [1, 256, 9, 15]
        #wr_str('//body output, put together')
        out.append(x1)
        out.append(x2)
        out.append(x3)
        # x1.shape = [1, 64, 34, 60]
        # x2.shape = [1, 128, 17, 30]
        # x3.shape = [1, 256, 9, 15]
        return out