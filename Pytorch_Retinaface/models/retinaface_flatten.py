import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from models.net_1 import MobileNetV1 as MobileNetV1
from models.net_1 import FPN as FPN
from models.net_1 import SSH as SSH
import numpy as np
import pdb


class ClassHead(nn.Module):
    def __init__(self, inchannels=512,num_anchors=3):  # 传进的是: 64, 2
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)
    def forward(self,x):
        #x.shape = [1,64,34,60]
        #pdb.set_trace()
        #wr_str('//ClassHead conv')
        out = self.conv1x1(x)
        # out.shape = [1,4,34,60]
        #wr_str('//ClassHead permute')
        out = out.permute(0,2,3,1).contiguous()
        # x.shape = [1,34,60,4]
        return out.view(out.shape[0], -1, 2) #shape = [ 1, 4080, 2]

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):#2
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x): # model.c要写在每一个forward里面；并且自带注释
        #wr_str('//BboxHead conv')
        out = self.conv1x1(x)
        #wr_str('//BboxHead permute')
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)
    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        param cfg:  Network related settings.
        param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        self.quant = QuantStub()
        self.body = MobileNetV1()
        in_channels_stage2 = cfg['in_channel'] #in channels :32, out channels 64;
        in_channels_list = [
            in_channels_stage2 * 2, # 64
            in_channels_stage2 * 4, # 128
            in_channels_stage2 * 8, # 256
        ]
        out_channels = cfg['out_channel'] # 64
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3,       inchannels=cfg['out_channel'])#64
        self.BboxHead = self._make_bbox_head(fpn_num=3,         inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.dequant = DeQuantStub()
        # in_channel 32: out_channel:64

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num)) # inchannels: 64, anchor_num: 2
        return classhead

    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        # f = open('D:/TS_doc/Nov/RetinaFace_pj/Pytorch_Retinaface/model.txt', 'w')
        # f.write('//layer0')
        # f.write('\n')
        # f.close()
        inputs = self.quant(inputs)
        # 这个数据格式的变换resnet和mobile net版本,其实是一毛一样的:
        # [1,3,375,500]
        out1 = self.body(inputs) #mobile_net
        fpn  = self.fpn(out1)
        # out1 = fpn
        # list len = 3
        # fpn[0].shape = [1,64,47,63]
        # fpn[1].shape = [1,64,24,32]
        # fpn[2].shape = [1,64,12,16]

        #list len = 3
        # fpn[0].shape = [1,64,47,63]
        # fpn[1].shape = [1,64,24,32]
        # fpn[2].shape = [1,64,12,16]
        # SSH
        #wr_str('//ssh1')
        feature1 = self.ssh1(fpn[0])
        # feature1.shape = [1,64,34,60]
        #wr_str('//ssh2')
        feature2 = self.ssh2(fpn[1])
        # feature2.shape = [1,64,17,30]
        #wr_str('//ssh3')
        feature3 = self.ssh3(fpn[2])
        # feature3.shape = [1,64, 9,15]
        features = [feature1, feature2, feature3]
        #bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        bbox_regressions0 = self.BboxHead[0](features[0]) # [ 1, 4080, 2]
        bbox_regressions1 = self.BboxHead[1](features[1]) # [ 1, 1020, 2]
        bbox_regressions2 = self.BboxHead[2](features[2]) # [ 1,  270, 2]

        conf0 = self.ClassHead[0](features[0])
        conf1 = self.ClassHead[1](features[1])
        conf2 = self.ClassHead[2](features[2])
        #
        # try:
        #     aaa= np.array(conf0)[0, 0, 0]
        # except:
        #     np.save('./layer_output_time/' + 'm' + '_loc0' + '.npy', bbox_regressions0.int_repr())
        #     np.save('./layer_output_time/' + 'm' + '_loc1' + '.npy', bbox_regressions1.int_repr())
        #     np.save('./layer_output_time/' + 'm' + '_loc2' + '.npy', bbox_regressions2.int_repr())
        #     np.save('./layer_output_time/' + 'm' + '_conf0' + '.npy', conf0.int_repr())
        #     np.save('./layer_output_time/' + 'm' + '_conf1' + '.npy', conf1.int_repr())
        #     np.save('./layer_output_time/' + 'm' + '_conf2' + '.npy', conf2.int_repr())
        #
        # try:
        #     aaa= np.array(conf0)[0, 0, 0]
        # except:
        #     print('//layer1 cat_loc_list parameters(s,q):', bbox_regressions0.q_scale(), bbox_regressions0.q_zero_point(),
        #           bbox_regressions1.q_scale(), bbox_regressions1.q_zero_point(),
        #           bbox_regressions2.q_scale(), bbox_regressions2.q_zero_point())
        #     print('//layer1 cat_conf_list parameters(s,q):',conf0.q_scale(), conf0.q_zero_point(),
        #           conf1.q_scale(), conf1.q_zero_point(),
        #           conf2.q_scale(), conf2.q_zero_point())
        conf0 = self.dequant(conf0)
        conf1 = self.dequant(conf1)
        conf2 = self.dequant(conf2)
        bbox_regressions0 = self.dequant(bbox_regressions0)
        bbox_regressions1 = self.dequant(bbox_regressions1)
        bbox_regressions2 = self.dequant(bbox_regressions2)
        bbox_regressions = torch.cat([bbox_regressions0, bbox_regressions1, bbox_regressions2], dim=1)
        classifications = torch.cat([conf0, conf1, conf2], dim=1)

        b_bbox_regressions = bbox_regressions
        b_classifications = classifications
        #ldm_regressions = self.dequant(ldm_regressions)
        output = b_bbox_regressions, b_classifications,bbox_regressions,classifications # , ldm_regressions
        #平铺开的网络可以打印feature map;
        # model.c 主要依赖于解析网络；
        return output