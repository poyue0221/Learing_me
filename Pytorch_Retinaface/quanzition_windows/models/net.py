import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        # nn.LeakyReLU(negative_slope=leaky, inplace=True)
        nn.ReLU()
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.LeakyReLU(negative_slope=leaky, inplace=True)
        nn.ReLU()
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # nn.LeakyReLU(negative_slope= leaky,inplace=True),
        nn.ReLU(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.LeakyReLU(negative_slope= leaky,inplace=True),
        nn.ReLU()
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

# def conv_dw_1(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.ReLU()
#     )
#
# def conv_dw_2(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU()
#     )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        # print(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # x = self.avg(x)
        # # x = self.model(x)
        # x = x.view(-1, 256)
        # x = self.fc(x)
        return x
    # def __init__(self):
    #     super(MobileNetV1, self).__init__()
    #     self.stage1_0 = nn.Sequential(conv_bn(3, 8, 2))  # 3
    #     self.stage1_1 = nn.Sequential(conv_dw_1(8, 16, 1))  # 7                  1
    #     self.stage1_2 = nn.Sequential(conv_dw_2(8, 16, 1))  # 7                  1
    #     self.stage1_3 = nn.Sequential(conv_dw_1(16, 32, 2))  # 11                 2
    #     self.stage1_4 = nn.Sequential(conv_dw_2(16, 32, 2))  # 11                 2
    #     self.stage1_5 = nn.Sequential(conv_dw_1(32, 32, 1))  # 19                 3
    #     self.stage1_6 = nn.Sequential(conv_dw_2(32, 32, 1))  # 19                 3
    #     self.stage1_7 = nn.Sequential(conv_dw_1(32, 64, 2))  # 27                 4
    #     self.stage1_8 = nn.Sequential(conv_dw_2(32, 64, 2))  # 27                 4
    #     self.stage1_9 = nn.Sequential(conv_dw_1(64, 64, 1))  # 43                 5
    #     self.stage1_10 = nn.Sequential(conv_dw_2(64, 64, 1))  # 43                 5
    #
    #     self.stage2_0 = nn.Sequential(conv_dw_1(64, 128, 2))  # 43 + 16 = 59      1
    #     self.stage2_1 = nn.Sequential(conv_dw_2(64, 128, 2))  # 43 + 16 = 59      1
    #     self.stage2_2 = nn.Sequential(conv_dw_1(128, 128, 1))  # 59 + 32 = 91      2
    #     self.stage2_3 = nn.Sequential(conv_dw_2(128, 128, 1))  # 59 + 32 = 91      2
    #     self.stage2_4 = nn.Sequential(conv_dw_1(128, 128, 1))  # 91 + 32 = 123     3
    #     self.stage2_5 = nn.Sequential(conv_dw_2(128, 128, 1))  # 91 + 32 = 123     3
    #     self.stage2_6 = nn.Sequential(conv_dw_1(128, 128, 1))  # 123 + 32 = 155    4
    #     self.stage2_7 = nn.Sequential(conv_dw_2(128, 128, 1))  # 123 + 32 = 155    4
    #     self.stage2_8 = nn.Sequential(conv_dw_1(128, 128, 1))  # 155 + 32 = 187    5
    #     self.stage2_9 = nn.Sequential(conv_dw_2(128, 128, 1))  # 155 + 32 = 187    5
    #     self.stage2_10 = nn.Sequential(conv_dw_1(128, 128, 1))  # 187 + 32 = 219    6
    #     self.stage2_11 = nn.Sequential(conv_dw_2(128, 128, 1))  # 187 + 32 = 219    6
    #
    #     self.stage3_0 = nn.Sequential(conv_dw_1(128, 256, 2))  # 219 + 32 = 241    1
    #     self.stage3_1 = nn.Sequential(conv_dw_2(128, 256, 2))  # 219 + 32 = 241    1
    #     self.stage3_2 = nn.Sequential(conv_dw_1(256, 256, 1))  # 241 + 64 = 301    2
    #     self.stage3_3 = nn.Sequential(conv_dw_2(256, 256, 1))  # 241 + 64 = 301    2
    #
    # def forward(self, x):
    #     out = []
    #     # [1,3,270,480]
    #     # wr_str('//stage1_0,conv2d')
    #     x1 = self.stage1_0(x)
    #     # [1,8,135,240]
    #     # wr_str('//stage1_1,depth_wise')
    #     x1 = self.stage1_1(x1)
    #     # [1,8,135,240]
    #     # wr_str('//stage1_2,conv2d')
    #     x1 = self.stage1_2(x1)
    #     # [1, 16, 135, 240]
    #     # wr_str('//stage1_3,depth_wise')
    #     x1 = self.stage1_3(x1)
    #     # [1, 16, 68, 120]
    #     # wr_str('//stage1_4,conv2d')
    #     x1 = self.stage1_4(x1)
    #     # [1, 32, 68, 120]
    #     # wr_str('//stage1_5,depth_wise')
    #     x1 = self.stage1_5(x1)
    #     # [1, 32, 68, 120]
    #     # wr_str('//stage1_6,conv2d')
    #     x1 = self.stage1_6(x1)
    #     # [1, 32, 68, 120]
    #     # wr_str('//stage1_7,depth_wise')
    #     x1 = self.stage1_7(x1)
    #     # [1, 32, 34, 60]
    #     # wr_str('//stage1_8,conv2d')
    #     x1 = self.stage1_8(x1)
    #     # [1, 64, 34, 60]
    #     # wr_str('//stage1_9,depth_wise')
    #     x1 = self.stage1_9(x1)
    #     # [1, 64, 34, 60]
    #     # wr_str('//stage1_10,conv2d')
    #     x1 = self.stage1_10(x1)
    #     # [1, 64, 34, 60]
    #     # wr_str('//stage2_0,depth_wise')
    #     x2 = self.stage2_0(x1)
    #     # [1, 64, 17, 30]
    #     # wr_str('//stage2_1,conv2d')
    #     x2 = self.stage2_1(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_2,depth_wise')
    #     x2 = self.stage2_2(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_3,conv2d')
    #     x2 = self.stage2_3(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_4,depth_wise')
    #     x2 = self.stage2_4(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_5,conv2d')
    #     x2 = self.stage2_5(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_6,depth_wise')
    #     x2 = self.stage2_6(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_8,conv2d')
    #     x2 = self.stage2_7(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage1_1,depth_wise')
    #     x2 = self.stage2_8(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_9,conv2d')
    #     x2 = self.stage2_9(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_10,depth_wise')
    #     x2 = self.stage2_10(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage2_11,conv2d')
    #     x2 = self.stage2_11(x2)
    #     # [1, 128, 17, 30]
    #     # wr_str('//stage3_0,depth_wise')
    #     x3 = self.stage3_0(x2)
    #     # [1, 128, 9, 15]
    #     # wr_str('//stage3_1,conv2d')
    #     x3 = self.stage3_1(x3)
    #     # [1, 256, 9, 15]
    #     # wr_str('//stage3_2,depth_wise')
    #     x3 = self.stage3_2(x3)
    #     # [1, 256, 9, 15]
    #     # wr_str('//stage3_3,conv2d')
    #     x3 = self.stage3_3(x3)
    #     # [1, 256, 9, 15]
    #     # wr_str('//body output, put together')
    #     out.append(x1)
    #     out.append(x2)
    #     out.append(x3)
    #     # x1.shape = [1, 64, 34, 60]
    #     # x2.shape = [1, 128, 17, 30]
    #     # x3.shape = [1, 256, 9, 15]
    #     return out


# def _make_divisible(ch, divisor=8, min_ch=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_ch is None:
#         min_ch = divisor
#     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_ch < 0.9 * ch:
#         new_ch += divisor
#     return new_ch
#
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU6(inplace=True)
#         )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channel, out_channel, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         hidden_channel = in_channel * expand_ratio
#         self.use_shortcut = stride == 1 and in_channel == out_channel
#
#         layers = []
#         if expand_ratio != 1:
#             # 1x1 pointwise conv
#             layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
#         layers.extend([
#             # 3x3 depthwise conv
#             ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
#             # 1x1 pointwise conv(linear)
#             nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#         ])
#
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_shortcut:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, alpha=0.25, round_nearest=8):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = _make_divisible(32 * alpha, round_nearest)
#         last_channel = _make_divisible(1280 * alpha, round_nearest)
#
#         inverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             # #
#             [6, 128, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         features = []
#         feature1 = []
#         feature2 = []
#         feature3 = []
#         feature1_ = []
#         feature2_ = []
#         feature3_ = []
#         # conv1 layer
#         features.append(ConvBNReLU(3, input_channel, stride=2))
#         feature1.append(ConvBNReLU(3, input_channel, stride=2))
#         # building inverted residual residual blockes
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * alpha, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 if output_channel in [4, 6, 8]:
#                     feature1.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 if output_channel in [16, 24]:
#                     feature2.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 if output_channel in [32, 80]:
#                     feature3.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#
#         # building last several layers
#         # features.append(ConvBNReLU(input_channel, last_channel, 1))
#         feature1_.append(ConvBNReLU(8, 64, 1))
#         feature2_.append(ConvBNReLU(24, 128, 1))
#         feature3_.append(ConvBNReLU(80, 256, 1))
#
#         # combine feature layers
#         self.features = nn.Sequential(*features)
#         self.feature1 = nn.Sequential(*feature1)
#         self.feature2 = nn.Sequential(*feature2)
#         self.feature3 = nn.Sequential(*feature3)
#         self.feature1_ = nn.Sequential(*feature1_)
#         self.feature2_ = nn.Sequential(*feature2_)
#         self.feature3_ = nn.Sequential(*feature3_)
#
#         # building classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, num_classes)
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         # x = self.features(x)
#         j_ = self.feature1(x)
#         j = self.feature1_(j_)
#         k_ = self.feature2(j_)
#         k = self.feature2_(k_)
#         l_ = self.feature3(k_)
#         l= self.feature3_(l_)
#         # x = self.avgpool(x)
#         # x = torch.flatten(x, 1)
#         # x = self.classifier(x)
#         return j,k,l
#         # return x