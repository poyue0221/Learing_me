from __future__ import print_function
import os
import argparse
import torch
# import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface_flatten import RetinaFace
# from models.retinaface import RetinaFaces
from utils.box_utils import decode, decode_landm
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc
import time
import torch.nn.functional as F
from utils.timer import Timer
# from bn_fusion import fuse_bn_recursively
# import copy
# import torch.quantization
import struct
import pdb
import glob

# import torch.nn.functional as F
# .int_repr()

# 目前公司支持的量化方式：每层量化，对称量化，weight qint8, bias qint32;
# pytorch 的量化工具路径：C:\Users\12923\Anaconda3\envs\pytorch\Lib\site-packages\torch\quantization
# C:\Users\12923\Anaconda3\envs\pytorch\Lib\site-packages\torch\nn\quantized\modules\conv.py 这个文件中有卷积的实现方式；
# command: python test_widerface_module.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25
# command: python test_single_quint8_backup.py
# 改动的127有，fake quantize 文件中的
#  config文件中的
# C:\Users\12923\Anaconda3\envs\pytorch\Lib\site-packages\torch\nn\quantized\modules 小函数的位置
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_epoch_190.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='../data/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='../data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')
args = parser.parse_args()


def calibrate(model, dataset, n_calibration, batch_size):
    batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=1, collate_fn=detection_collate))
    with torch.no_grad():
        for i in range(n_calibration):
            print('###### calibrating ########', i)
            images, targets = next(batch_iterator)
            model(images)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
# def load_model(model, pretrained_path, load_to_cpu):
#     print('Loading pretrained model from {}'.format(pretrained_path))
#     if load_to_cpu:
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
#     else:
#         device = torch.cuda.current_device()
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
#
#     ori_name = ['body.stage1.0.0.weight', 'body.stage1.0.1.weight', 'body.stage1.0.1.bias',
#                 'body.stage1.0.1.running_mean', 'body.stage1.0.1.running_var', 'body.stage1.0.1.num_batches_tracked',
#                 'body.stage1.1.0.weight', 'body.stage1.1.1.weight', 'body.s\
# tage1.1.1.bias', 'body.stage1.1.1.running_mean', 'body.stage1.1.1.running_var', 'body.stage1.1.1.num_batches_tracked',
#                 'body.stage1.1.3.weight', 'body.stage1.1.4.weight', 'body.stage1.1.4.bias',
#                 'body.stage1.1.4.running_mean', 'body.stage1.1.4.runn\
# ing_var', 'body.stage1.1.4.num_batches_tracked', 'body.stage1.2.0.weight', 'body.stage1.2.1.weight',
#                 'body.stage1.2.1.bias', 'body.stage1.2.1.running_mean', 'body.stage1.2.1.running_var',
#                 'body.stage1.2.1.num_batches_tracked', 'body.stage1.2.3.weig\
# ht', 'body.stage1.2.4.weight', 'body.stage1.2.4.bias', 'body.stage1.2.4.running_mean', 'body.stage1.2.4.running_var',
#                 'body.stage1.2.4.num_batches_tracked', 'body.stage1.3.0.weight', 'body.stage1.3.1.weight',
#                 'body.stage1.3.1.bias', 'body.stage1.3.\
# 1.running_mean', 'body.stage1.3.1.running_var', 'body.stage1.3.1.num_batches_tracked', 'body.stage1.3.3.weight',
#                 'body.stage1.3.4.weight', 'body.stage1.3.4.bias', 'body.stage1.3.4.running_mean',
#                 'body.stage1.3.4.running_var', 'body.stage1.3.4.num_b\
# atches_tracked', 'body.stage1.4.0.weight', 'body.stage1.4.1.weight', 'body.stage1.4.1.bias',
#                 'body.stage1.4.1.running_mean', 'body.stage1.4.1.running_var', 'body.stage1.4.1.num_batches_tracked',
#                 'body.stage1.4.3.weight', 'body.stage1.4.4.weight', '\
# body.stage1.4.4.bias', 'body.stage1.4.4.running_mean', 'body.stage1.4.4.running_var',
#                 'body.stage1.4.4.num_batches_tracked', 'body.stage1.5.0.weight', 'body.stage1.5.1.weight',
#                 'body.stage1.5.1.bias', 'body.stage1.5.1.running_mean', 'body.stage1.5.\
# 1.running_var', 'body.stage1.5.1.num_batches_tracked', 'body.stage1.5.3.weight', 'body.stage1.5.4.weight',
#                 'body.stage1.5.4.bias', 'body.stage1.5.4.running_mean', 'body.stage1.5.4.running_var',
#                 'body.stage1.5.4.num_batches_tracked', 'body.stage2.0.\
# 0.weight', 'body.stage2.0.1.weight', 'body.stage2.0.1.bias', 'body.stage2.0.1.running_mean',
#                 'body.stage2.0.1.running_var', 'body.stage2.0.1.num_batches_tracked', 'body.stage2.0.3.weight',
#                 'body.stage2.0.4.weight', 'body.stage2.0.4.bias', 'body.sta\
# ge2.0.4.running_mean', 'body.stage2.0.4.running_var', 'body.stage2.0.4.num_batches_tracked', 'body.stage2.1.0.weight',
#                 'body.stage2.1.1.weight', 'body.stage2.1.1.bias', 'body.stage2.1.1.running_mean',
#                 'body.stage2.1.1.running_var', 'body.stage2.1.1\
# .num_batches_tracked', 'body.sta\
# ge2.1.3.weight', 'body.stage2.1.4.weight', 'body.stage2.1.4.bias', 'body.stage2.1.4.running_mean',
#                 'body.stage2.1.4.running_var', 'body.stage2.1.4.num_batches_tracked', 'body.stage2.2.0.weight', 'body.stage2.2.1.weig\
# ht', 'body.stage2.2.1.bias', 'body.stage2.2.1.running_mean', 'body.stage2.2.1.running_var',
#                 'body.stage2.2.1.num_batches_tracked', 'body.stage2.2.3.weight', 'body.stage2.2.4.weight',
#                 'body.stage2.2.4.bias', 'body.stage2.2.4.running_mean', 'body.sta\
# ge2.2.4.running_var', 'body.stage2.2.4.num_batches_tracked', 'body.stage2.3.0.weight', 'body.stage2.3.1.weight',
#                 'body.stage2.3.1.bias', 'body.stage2.3.1.running_mean', 'body.stage2.3.1.running_var',
#                 'body.stage2.3.1.num_batches_tracked', 'body.sta\
# ge2.3.3.weight', 'body.stage2.3.4.weight', 'body.stage2.3.4.bias', 'body.stage2.3.4.running_mean',
#                 'body.stage2.3.4.running_var', 'body.stage2.3.4.num_batches_tracked', 'body.stage2.4.0.weight',
#                 'body.stage2.4.1.weight', 'body.stage2.4.1.bias', 'bo\
# dy.stage2.4.1.running_mean', 'body.stage2.4.1.running_var', 'body.stage2.4.1.num_batches_tracked',
#                 'body.stage2.4.3.weight', 'body.stage2.4.4.weight', 'body.stage2.4.4.bias',
#                 'body.stage2.4.4.running_mean', 'body.stage2.4.4.running_var', 'body.stag\
# e2.4.4.num_batches_tracked', 'body.stage2.5.0.weight', 'body.stage2.5.1.weight', 'body.stage2.5.1.bias',
#                 'body.stage2.5.1.running_mean', 'body.stage2.5.1.running_var', 'body.stage2.5.1.num_batches_tracked',
#                 'body.stage2.5.3.weight', 'body.stage2.5.\
# 4.weight', 'body.stage2.5.4.bias', 'body.stage2.5.4.running_mean', 'body.stage2.5.4.running_var',
#                 'body.stage2.5.4.num_batches_tracked', 'body.stage3.0.0.weight', 'body.stage3.0.1.weight',
#                 'body.stage3.0.1.bias', 'body.stage3.0.1.running_mean', 'bo\
# dy.stage3.0.1.running_var', 'body.stage3.0.1.num_batches_tracked', 'body.stage3.0.3.weight', 'body.stage3.0.4.weight',
#                 'body.stage3.0.4.bias', 'body.stage3.0.4.running_mean', 'body.stage3.0.4.running_var',
#                 'body.stage3.0.4.num_batches_tracked', 'bo\
# dy.stage3.1.0.weight', 'body.stage3.1.1.weight', 'body.stage3.1.1.bias', 'body.stage3.1.1.running_mean',
#                 'body.stage3.1.1.running_var', 'body.stage3.1.1.num_batches_tracked', 'body.stage3.1.3.weight',
#                 'body.stage3.1.4.weight', 'body.stage3.1.4.bias\
# ', 'body.stage3.1\
# .4.running_mean', 'body.stage3.1.4.running_var', 'body.stage3.1.4.num_batches_tracked']
#     new_name = ['body.stage1_0.0.0.weight', 'body.stage1_0.0.1.weight', 'body.stage1_0.0.1.bias',
#                 'body.stage1_0.0.1.running_mean', 'body.stage1_0.0.1.running_var',
#                 'body.stage1_0.0.1.num_batches_tracked', 'body.stage1_1.0.0.weight', 'body.stage1_1.0.1\
# .weight', 'body.stage1_1.0.1.bias', 'body.stage1_1.0.1.running_mean', 'body.stage1_1.0.1.running_var',
#                 'body.stage1_1.0.1.num_batches_tracked', 'body.stage1_2.0.0.weight', 'body.stage1_2.0.1.weight',
#                 'body.stage1_2.0.1.bias', 'body.stage1_2.0.1.run\
# ning_mean', 'body.stage1_2.0.1.running_var', 'body.stage1_2.0.1.num_batches_tracked', 'body.stage1_3.0.0.weight',
#                 'body.stage1_3.0.1.weight', 'body.stage1_3.0.1.bias', 'body.stage1_3.0.1.running_mean',
#                 'body.stage1_3.0.1.running_var', 'body.stage1_\
# 3.0.1.num_batches_tracked', 'body.stage1_4.0.0.weight', 'body.stage1_4.0.1.weight', 'body.stage1_4.0.1.bias',
#                 'body.stage1_4.0.1.running_mean', 'body.stage1_4.0.1.running_var',
#                 'body.stage1_4.0.1.num_batches_tracked', 'body.stage1_5.0.0.weight', 'b\
# ody.stage1_5.0.1.weight', 'body.stage1_5.0.1.bias', 'body.stage1_5.0.1.running_mean', 'body.stage1_5.0.1.running_var',
#                 'body.stage1_5.0.1.num_batches_tracked', 'body.stage1_6.0.0.weight', 'body.stage1_6.0.1.weight',
#                 'body.stage1_6.0.1.bias', 'body.\
# stage1_6.0.1.running_mean', 'body.stage1_6.0.1.running_var', 'body.stage1_6.0.1.num_batches_tracked',
#                 'body.stage1_7.0.0.weight', 'body.stage1_7.0.1.weight', 'body.stage1_7.0.1.bias',
#                 'body.stage1_7.0.1.running_mean', 'body.stage1_7.0.1.running_var\
# ', 'body.stage1_7.0.1.num_batches_tracked', 'body.stage1_8.0.0.weight', 'body.stage1_8.0.1.weight',
#                 'body.stage1_8.0.1.bias', 'body.stage1_8.0.1.running_mean', 'body.stage1_8.0.1.running_var',
#                 'body.stage1_8.0.1.num_batches_tracked', 'body.stage1_9\
# .0.0.weight', 'body.stage1_9.0.1.weight', 'body.stage1_9.0.1.bias', 'body.stage1_9.0.1.running_mean',
#                 'body.stage1_9.0.1.running_var', 'body.stage1_9.0.1.num_batches_tracked', 'body.stage1_10.0.0.weight',
#                 'body.stage1_10.0.1.weight', 'body.stage1_1\
# 0.0.1.bias', 'body.stage1_10.0.1.running_mean', 'body.stage1_10.0.1.running_var',
#                 'body.stage1_10.0.1.num_batches_tracked', 'body.stage2_0.0.0.weight', 'body.stage2_0.0.1.weight',
#                 'body.stage2_0.0.1.bias', 'body.stage2_0.0.1.running_mean', 'body.st\
# age2_0.0.1.running_var', 'body.stage2_0.0.1.num_batches_tracked', 'body.stage2_1.0.0.weight',
#                 'body.stage2_1.0.1.weight', 'body.stage2_1.0.1.bias', 'body.stage2_1.0.1.running_mean',
#                 'body.stage2_1.0.1.running_var', 'body.stage2_1.0.1.num_batches_tr\
# acked', 'body.stage2_2.0.0.weight', 'body.stage2_2.0.1.weight', 'body.stage2_2.0.1.bias',
#                 'body.stage2_2.0.1.running_mean', 'body.stage2_2.0.1.running_var',
#                 'body.stage2_2.0.1.num_batches_tracked', 'body.stage2_3.0.0.weight', 'body.stage2_3.0.1.wei\
# ght', 'body.stage2_3.0.1.bias', 'body.stage2_3.0.1.running_mean', 'body.stage2_3.0.1.running_var',
#                 'body.stage2_3.0.1.num_batches_tracked', 'body.stage2_4.0.0.weight', 'body.stage2_4.0.1.weight',
#                 'body.stage2_4.0.1.bias', 'body.stage2_4.0.1.running\
# _mean', 'body.stage2_4.0.1.running_var', 'body.stage2_4.0.1.num_batches_tracked', 'body.stage2_5.0.0.weight',
#                 'body.stage2_5.0.1.weight', 'body.stage2_5.0.1.bias', 'body.stage2_5.0.1.running_mean',
#                 'body.stage2_5.0.1.running_var', 'body.stage2_5.0.\
# 1.num_batches_tracked', 'body.stage2_6.0.0.weight', 'body.stage2_6.0.1.weight', 'body.stage2_6.0.1.bias',
#                 'body.stage2_6.0.1.running_mean', 'body.stage2_6.0.1.running_var',
#                 'body.stage2_6.0.1.num_batches_tracked', 'body.stage2_7.0.0.weight', 'body.\
# stage2_7.0.1.weight', 'body.stage2_7.0.1.bias', 'body.stage2_7.0.1.running_mean', 'body.stage2_7.0.1.running_var',
#                 'body.stage2_7.0.1.num_batches_tracked', 'body.stage2_8.0.0.weight', 'body.stage2_8.0.1.weight',
#                 'body.stage2_8.0.1.bias', 'body.stag\
# e2_8.0.1.running_mean', 'body.stage2_8.0.1.running_var', 'body.stage2_8.0.1.num_batches_tracked',
#                 'body.stage2_9.0.0.weight', 'body.stage2_9.0.1.weight', 'body.stage2_9.0.1.bias',
#                 'body.stage2_9.0.1.running_mean', 'body.stage2_9.0.1.running_var', '\
# body.stage2_9.0.1.num_batches_tracked', 'body.stage2_10.0.0.weight', 'body.stage2_10.0.1.weight',
#                 'body.stage2_10.0.1.bias', 'body.stage2_10.0.1.running_mean', 'body.stage2_10.0.1.running_var',
#                 'body.stage2_10.0.1.num_batches_tracked', 'body.stage2\
# _11.0.0.weight', 'body.stage2_11.0.1.weight', 'body.stage2_11.0.1.bias', 'body.stage2_11.0.1.running_mean',
#                 'body.stage2_11.0.1.running_var', 'body.stage2_11.0.1.num_batches_tracked', 'body.stage3_0.0.0.weight',
#                 'body.stage3_0.0.1.weight', 'body.st\
# age3_0.0.1.bias', 'body.stage3_0.0.1.running_mean', 'body.stage3_0.0.1.running_var',
#                 'body.stage3_0.0.1.num_batches_tracked', 'body.stage3_1.0.0.weight', 'body.stage3_1.0.1.weight',
#                 'body.stage3_1.0.1.bias', 'body.stage3_1.0.1.running_mean', 'body.\
# stage3_1.0.1.running_var', 'body.stage3_1.0.1.num_batches_tracked', 'body.stage3_2.0.0.weight',
#                 'body.stage3_2.0.1.weight', 'body.stage3_2.0.1.bias', 'body.stage3_2.0.1.running_mean',
#                 'body.stage3_2.0.1.running_var', 'body.stage3_2.0.1.num_batches_\
# tracked', 'body.stage3_3.0.0.weight', 'body.stage3_3.0.1.weight', 'body.stage3_3.0.1.bias',
#                 'body.stage3_3.0.1.running_mean', 'body.stage3_3.0.1.running_var',
#                 'body.stage3_3.0.1.num_batches_tracked']
#
#     i = 0  # 当前的debug进度，里面的参数shape是对的，应该是顺序问题pop的方式是不对的，要插回去原有的位置
#
#     # python的name是不可以更改的；
#     # 前三层： 162层
#     stage_num = 162
#     # 当前的方式就是重新再写一个网络的字典去更新；
#     new_dict = {}
#     for k, v in pretrained_dict.items():  # type class: ditct
#         if i < 162:
#             new_dict[new_name[i]] = v
#         else:
#             new_dict[k] = v
#         i = i + 1
#     # pretrained_dict = new_state_dict #超级字典，键值对
#     # for i in range(len(ori_name)):
#     #   pretrained_dict.update({new_name[i]: pretrained_dict.pop(ori_name[i])})
#     model.load_state_dict(new_dict, strict=False)
#     return model


if __name__ == '__main__':

    torch.set_grad_enabled(False)
    cfg = None
    cfg = cfg_mnet
    # net and model
    # rf_model = RetinaFace(cfg=cfg, phase='test')
    # rf_model = load_model(rf_model, args.trained_model, args.cpu)
    # rf_model.eval()
    # rf_mode = rf_model.to("cpu")

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)

    net_ori = net
    net.eval()
    print('Finished loading model!')
    # cudnn.benchmark = True
    # device = torch.device("cpu" if args.cpu else "cuda")
    # net = net.to(device)
    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # net_fused = fuse_bn_recursively(copy.deepcopy(net))
    # net_fused = torch.quantization.fuse_modules(net, [  # stage1
    #     ['body.stage1_0.0.0', 'body.stage1_0.0.1', 'body.stage1_0.0.2'],
    #     ['body.stage1_1.0.0', 'body.stage1_1.0.1', 'body.stage1_1.0.2'],
    #     ['body.stage1_2.0.0', 'body.stage1_2.0.1', 'body.stage1_2.0.2'],
    #     ['body.stage1_3.0.0', 'body.stage1_3.0.1', 'body.stage1_3.0.2'],
    #     ['body.stage1_4.0.0', 'body.stage1_4.0.1', 'body.stage1_4.0.2'],
    #     ['body.stage1_5.0.0', 'body.stage1_5.0.1', 'body.stage1_5.0.2'],
    #     ['body.stage1_6.0.0', 'body.stage1_6.0.1', 'body.stage1_6.0.2'],
    #     ['body.stage1_7.0.0', 'body.stage1_7.0.1', 'body.stage1_7.0.2'],
    #     ['body.stage1_8.0.0', 'body.stage1_8.0.1', 'body.stage1_8.0.2'],
    #     ['body.stage1_9.0.0', 'body.stage1_9.0.1', 'body.stage1_9.0.2'],
    #     ['body.stage1_10.0.0', 'body.stage1_10.0.1', 'body.stage1_10.0.2'],
    #     # stage2
    #     ['body.stage2_0.0.0', 'body.stage2_0.0.1', 'body.stage2_0.0.2'],
    #     ['body.stage2_1.0.0', 'body.stage2_1.0.1', 'body.stage2_1.0.2'],
    #     ['body.stage2_2.0.0', 'body.stage2_2.0.1', 'body.stage2_2.0.2'],
    #     ['body.stage2_3.0.0', 'body.stage2_3.0.1', 'body.stage2_3.0.2'],
    #     ['body.stage2_4.0.0', 'body.stage2_4.0.1', 'body.stage2_4.0.2'],
    #     ['body.stage2_5.0.0', 'body.stage2_5.0.1', 'body.stage2_5.0.2'],
    #     ['body.stage2_6.0.0', 'body.stage2_6.0.1', 'body.stage2_6.0.2'],
    #     ['body.stage2_7.0.0', 'body.stage2_7.0.1', 'body.stage2_7.0.2'],
    #     ['body.stage2_8.0.0', 'body.stage2_8.0.1', 'body.stage2_8.0.2'],
    #     ['body.stage2_9.0.0', 'body.stage2_9.0.1', 'body.stage2_9.0.2'],
    #     ['body.stage2_10.0.0', 'body.stage2_10.0.1', 'body.stage2_10.0.2'],
    #     ['body.stage2_11.0.0', 'body.stage2_11.0.1', 'body.stage2_11.0.2'],
    #     # stage3
    #     ['body.stage3_0.0.0', 'body.stage3_0.0.1', 'body.stage3_0.0.2'],
    #     ['body.stage3_1.0.0', 'body.stage3_1.0.1', 'body.stage3_1.0.2'],
    #     ['body.stage3_2.0.0', 'body.stage3_2.0.1', 'body.stage3_2.0.2'],
    #     ['body.stage3_3.0.0', 'body.stage3_3.0.1', 'body.stage3_3.0.2'],
    #     # fpn
    #     ['fpn.output1.0', 'fpn.output1.1', 'fpn.output1.2'],
    #     ['fpn.output2.0', 'fpn.output2.1', 'fpn.output2.2'],
    #     ['fpn.output3.0', 'fpn.output3.1', 'fpn.output3.2'],
    #     ['fpn.merge1.0', 'fpn.merge1.1', 'fpn.merge1.2'],
    #     ['fpn.merge2.0', 'fpn.merge2.1', 'fpn.merge2.2'],
    #     # ssh1
    #     ['ssh1.conv3X3.0', 'ssh1.conv3X3.1'],
    #     ['ssh1.conv5X5_1.0', 'ssh1.conv5X5_1.1', 'ssh1.conv5X5_1.2'],
    #     ['ssh1.conv5X5_2.0', 'ssh1.conv5X5_2.1'],
    #     ['ssh1.conv7X7_2.0', 'ssh1.conv7X7_2.1', 'ssh1.conv7X7_2.2'],
    #     ['ssh1.conv7x7_3.0', 'ssh1.conv7x7_3.1'],
    #     # ssh2
    #     ['ssh2.conv3X3.0', 'ssh2.conv3X3.1'],
    #     ['ssh2.conv5X5_1.0', 'ssh2.conv5X5_1.1', 'ssh2.conv5X5_1.2'],
    #     ['ssh2.conv5X5_2.0', 'ssh2.conv5X5_2.1'],
    #     ['ssh2.conv7X7_2.0', 'ssh2.conv7X7_2.1', 'ssh2.conv7X7_2.2'],
    #     ['ssh2.conv7x7_3.0', 'ssh2.conv7x7_3.1'],
    #     # ssh3
    #     ['ssh3.conv3X3.0', 'ssh3.conv3X3.1'],
    #     ['ssh3.conv5X5_1.0', 'ssh3.conv5X5_1.1', 'ssh3.conv5X5_1.2'],
    #     ['ssh3.conv5X5_2.0', 'ssh3.conv5X5_2.1'],
    #     ['ssh3.conv7X7_2.0', 'ssh3.conv7X7_2.1', 'ssh3.conv7X7_2.2'],
    #     ['ssh3.conv7x7_3.0', 'ssh3.conv7x7_3.1']
    # ])  # fuse_modules
    net_fused = torch.quantization.fuse_modules(net, [  # stage1
        ['body.stage1.0.0', 'body.stage1.0.1', 'body.stage1.0.2'],
        ['body.stage1.1.0', 'body.stage1.1.1', 'body.stage1.1.2'],
        ['body.stage1.1.3', 'body.stage1.1.4', 'body.stage1.1.5'],
        ['body.stage1.2.0', 'body.stage1.2.1', 'body.stage1.2.2'],
        ['body.stage1.2.3', 'body.stage1.2.4', 'body.stage1.2.5'],
        ['body.stage1.3.0', 'body.stage1.3.1', 'body.stage1.3.2'],
        ['body.stage1.3.3', 'body.stage1.3.4', 'body.stage1.3.5'],
        ['body.stage1.4.0', 'body.stage1.4.1', 'body.stage1.4.2'],
        ['body.stage1.4.3', 'body.stage1.4.4', 'body.stage1.4.5'],
        ['body.stage1.5.0', 'body.stage1.5.1', 'body.stage1.5.2'],
        ['body.stage1.5.3', 'body.stage1.5.4', 'body.stage1.5.5'],
        # stage2
        ['body.stage2.0.0', 'body.stage2.0.1', 'body.stage2.0.2'],
        ['body.stage2.0.3', 'body.stage2.0.4', 'body.stage2.0.5'],
        ['body.stage2.1.0', 'body.stage2.1.1', 'body.stage2.1.2'],
        ['body.stage2.1.3', 'body.stage2.1.4', 'body.stage2.1.5'],
        ['body.stage2.2.0', 'body.stage2.2.1', 'body.stage2.2.2'],
        ['body.stage2.2.3', 'body.stage2.2.4', 'body.stage2.2.5'],
        ['body.stage2.3.0', 'body.stage2.3.1', 'body.stage2.3.2'],
        ['body.stage2.3.3', 'body.stage2.3.4', 'body.stage2.3.5'],
        ['body.stage2.4.0', 'body.stage2.4.1', 'body.stage2.4.2'],
        ['body.stage2.4.3', 'body.stage2.4.4', 'body.stage2.4.5'],
        ['body.stage2.5.0', 'body.stage2.5.1', 'body.stage2.5.2'],
        ['body.stage2.5.3', 'body.stage2.5.4', 'body.stage2.5.5'],
        # # stage3
        ['body.stage3.0.0', 'body.stage3.0.1', 'body.stage3.0.2'],
        ['body.stage3.0.3', 'body.stage3.0.4', 'body.stage3.0.5'],
        ['body.stage3.1.0', 'body.stage3.1.1', 'body.stage3.1.2'],
        ['body.stage3.1.3', 'body.stage3.1.4', 'body.stage3.1.5'],
        # fpn
        ['fpn.output1.0', 'fpn.output1.1', 'fpn.output1.2'],
        ['fpn.output2.0', 'fpn.output2.1', 'fpn.output2.2'],
        ['fpn.output3.0', 'fpn.output3.1', 'fpn.output3.2'],
        ['fpn.merge1.0', 'fpn.merge1.1', 'fpn.merge1.2'],
        ['fpn.merge2.0', 'fpn.merge2.1', 'fpn.merge2.2'],

        # # ssh1
        ['ssh1.conv3X3.0', 'ssh1.conv3X3.1'],
        ['ssh1.conv5X5_1.0', 'ssh1.conv5X5_1.1', 'ssh1.conv5X5_1.2'],
        ['ssh1.conv5X5_2.0', 'ssh1.conv5X5_2.1'],
        ['ssh1.conv7X7_2.0', 'ssh1.conv7X7_2.1', 'ssh1.conv7X7_2.2'],
        ['ssh1.conv7x7_3.0', 'ssh1.conv7x7_3.1'],
        # # ssh2
        ['ssh2.conv3X3.0', 'ssh2.conv3X3.1'],
        ['ssh2.conv5X5_1.0', 'ssh2.conv5X5_1.1', 'ssh2.conv5X5_1.2'],
        ['ssh2.conv5X5_2.0', 'ssh2.conv5X5_2.1'],
        ['ssh2.conv7X7_2.0', 'ssh2.conv7X7_2.1', 'ssh2.conv7X7_2.2'],
        ['ssh2.conv7x7_3.0', 'ssh2.conv7x7_3.1'],
        # ssh3
        ['ssh3.conv3X3.0', 'ssh3.conv3X3.1'],
        ['ssh3.conv5X5_1.0', 'ssh3.conv5X5_1.1', 'ssh3.conv5X5_1.2'],
        ['ssh3.conv5X5_2.0', 'ssh3.conv5X5_2.1'],
        ['ssh3.conv7X7_2.0', 'ssh3.conv7X7_2.1', 'ssh3.conv7X7_2.2'],
        ['ssh3.conv7x7_3.0', 'ssh3.conv7x7_3.1'],
    ])  # fuse_modules
    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    net_prepared = torch.quantization.prepare(net_fused, inplace=True)  # quantize
    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    # input_fp32 = torch.randn(1, 3, 375, 500)
    net_prepared.eval()
    # calibration start
    DATA_PATH = '../data/widerface/train/label.txt'
    BATCH_SIZE = 16
    N_CALIBRATION = 1
    # train_loader, test_loader = get_dataloader( DATA_PATH, BATCH_SIZE, N_CPU )
    training_dataset = '../data/widerface/train/label.txt'
    img_dim = cfg['image_size']
    rgb_mean = (104, 117, 123)  # bgr order
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    calibrate(net_prepared, dataset, N_CALIBRATION, BATCH_SIZE)
    print('calibration done')
    # calibration end
    # net_prepared(input_fp32)
    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    net = torch.quantization.convert(net_prepared, inplace=True) # quantize
    # torch.save(net.state_dict(), "./quant_model/mobilenet0.25_quant.pth")
    # torch.jit.save(torch.jit.script(net), "./quant_model/mobilenet0.25_quant0406.pth")
    # print(net)
    # for i in (net.state_dict().keys()):
    #     print(net.state_dict()[i])"./quant_model/mobilenet0.25_quant.pth"


    net.eval()
    '''
    for i in (net.state_dict().keys()):
        if 'weight' in i:
            np.save('./layer_output/' + i +'_weight_int8.npy', net.state_dict()[i].int_repr())
            np.save('./layer_output/' + i +'_weight_scale.npy', net.state_dict()[i].q_scale())
            np.save('./layer_output/' + i +'_weight_zero_point.npy', net.state_dict()[i].q_zero_point())
        else:
            np.save('./layer_output/' + i + '_else.npy', net.state_dict()[i])
        print(i + 'done!')
    # '''
    # 输入图片的路径

    # image_path = '../curve/test.jpg'

    image_path = ["./pc/1.jpg", "./pc/2.jpg", "./pc/3.jpg", "./pc/4.jpg", "./pc/5.jpg"
                  , "./pc/6.jpg", "./pc/7.jpg", "./pc/8.jpg", "./pc/9.jpg"]
    for i in image_path:
        print(i)
        image = i
        img_raw = cv2.imread(image, cv2.IMREAD_COLOR)
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img_raw = img_raw[0:719, 67:471]
        # img = np.float32(img_raw)
        # (540,720,3)
        # 原来的540 --> 405
        # testing scale
        resize = 1
        target_size = 1600
        max_size = 2150  # 480 270
        # img = cv2.resize(img_raw, (480, 270))
        img = cv2.resize(img_raw, (1024, 680))
        # img_raw = cv2.resize(img_raw, (480, 270))
        img_raw = cv2.resize(img_raw, (1024, 680))
        img = np.float32(img)
        print(img.shape)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to("cpu")
        scale = scale.to("cpu")
        # im_shape = img.shape
        # # cv2.imwrite("./harry_480_270.jpg", img_raw)
        # im_size_min = np.min(im_shape[0:2])
        # im_size_max = np.max(im_shape[0:2])
        # resize = float(target_size) / float(im_size_min)
        # # prevent bigger axis from being more than max_size:
        # if np.round(resize * im_size_max) > max_size:
        #     resize = float(max_size) / float(im_size_max)
        # if args.origin_size:
        #     resize = 1
        # if resize != 1:
        #     img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        # im_height, im_width, _ = img.shape
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1) #gbr
        # # temp = img[0, :, :]
        # # img[0, :, :] = img[2, :, :]
        # # img[2, :, :] = temp #rbg
        # img = torch.from_numpy(img).unsqueeze(0)
        print(img)
        _t = {'forward_pass': Timer(), 'misc': Timer()}
        _t['forward_pass'].tic()
        #################### network ##################
        net_ori.eval()
        np.save('./layer_output/' + 'input.npy', img)
        tic = time.time()
        # rf_time = time.time()
        # b_loc_, b_conf_, loc_, conf_, v_ = net(img)
        # print("原模型推理时间：", time.time()-rf_time)
        # net_time = time.time()
        b_loc, b_conf, loc, conf= net(img)# forward pass ,  landms
        # print("量化模型推理时间：", time.time()-net_time)

        # net_name = ["input","body0","body1","body2","fpn0","fpn1","fpn2","feature1","feature2","feature3","bbox_regressions0","bbox_regressions1","bbox_regressions2",
        #             "classification0","classification1","classification2"]

        # for i,j in enumerate(v_):
        #     # print(j)
        #     j = torch.Tensor(j)
        #
        #     a = j.reshape(-1)
        #
        #
        #     b = v[i].reshape(-1)
        #     b =torch.Tensor(b)
        #     # print(a)
        #     # print(b)
        #     # print("ends")
        #     corret = torch.cosine_similarity(a, b, dim=0)
        #     print(net_name[i]+": ")
        #     print(corret)
        #     print(a[0:10])
        #     print(b[0:10])
        # print(type(loc))
        # print(type(conf))
        '''
        f = open('nn_loc_int8.txt', 'w')  # 清空文件内容再写
        for iii in (np.array(b_loc.int_repr()[0])):
            for jjj in (iii):
                f.write(str(jjj))  # 只能写字符串
                f.write('\t')
            f.write('\n')
        f.close()
    
        f = open('nn_conf_int8.txt', 'w')  # 清空文件内容再写
        for iii in (np.array(b_conf.int_repr()[0])):
            for jjj in (iii):
                f.write(str(jjj))  # 只能写字符串
                f.write('\t')
            f.write('\n')
        f.close()
    
        f = open('nn_loc_dquan.txt', 'w')
        f.write(str(b_loc.q_scale()))  # 只能写字符串
        f.write('\n')
        f.write(str(b_loc.q_zero_point()))  # 只能写字符串
        f.write('\n')
        f.close()
    
        f = open('nn_conf_dquan.txt', 'w')  # 清空文件内容再写
        f.write(str(b_conf.q_scale()))  # 只能写字符串
        f.write('\n')
        f.write(str(b_conf.q_zero_point()))  # 只能写字符串
        f.write('\n')
        f.close()
    
        f = open('nn_loc_output.txt', 'w')  # 清空文件内容再写
        for iii in (loc[0]):
            for jjj in (iii):
                f.write(str(jjj)[7:-1])  # 只能写字符串
                f.write('\t')
            f.write('\n')
        f.close()
        f = open('nn_conf_output.txt', 'w')  # 清空文件内容再写
        for iii in (conf[0]):
            for jjj in (iii):
                f.write(str(jjj)[7:-1])  # 只能写字符串
                f.write('\t')
            f.write('\n')
        f.close()
        '''
        _t['forward_pass'].toc()
        _t['misc'].tic()

        # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # inds = np.where(scores > args.confidence_threshold)[0]  # 废物！！！
        # # priorbox = PriorBox(cfg, inds, image_size=(im_height, im_width))
        # priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        # # priors = priorbox.fast_box()
        # priors_ori = priorbox.forward()
        #
        #
        #
        # # boxes_ori = decode(loc[0], priors_ori.data, cfg['variance'])  # 这里函数1
        # boxes_ori = decode(loc[0], priors_ori.data, cfg['variance'])  # 这里函数1
        # boxes_ori = boxes_ori * scale / resize
        #
        # prior_data = priors_ori.data
        # # landms = landms[0][inds]
        # scores = scores[inds]
        # loc = loc[0][inds]
        # # (16082,4)
        #
        # boxes = decode(loc, prior_data, cfg['variance'])  # 这里函数1
        #
        # boxes = boxes * scale / resize
        # boxes = boxes.cpu().numpy()
        #
        # # keep top-K before NMS
        # order = scores.argsort()[::-1]  # 这里函数3，是一个排序
        # # order = scores.argsort()[::-1][:args.top_k]
        # boxes = boxes[order]
        # # landms = landms[order]
        # scores = scores[order]
        #
        # dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)  # 这里函数4
        # dets = dets[keep, :]
        # # landms = landms[keep]
        #
        # for i in range(len(dets)):
        #     if dets[i][0] < 0 or dets[i][1] < 0:
        #         dets[i][4] = -9
        #     if dets[i][0] > 480 or dets[i][1] > 270 or dets[i][2] > 480 or dets[i][3] > 270:
        #         dets[i][4] = -9
        #
        # _t['misc'].toc()
        # # pdb.set_trace()
        # aaa = -0.5
        # # pdb.set_trace()
        # # save image
        # # 正图片
        # # dets[0][0:5] = [185.551498, 55.972622, 251.458405, 131.926250, 3]
        # # dets[1][0:5] = [348.720642, 66.965881, 401.087982, 128.509323, 3]
        # # dets[2][0:5] = [70.013405, 75.125427, 122.895416, 137.884491, 3]
        # # 对称图片
        # # 70.907646    61.629433    123.544853    125.629433    0.248800
        # # 352.849762    75.395210    404.467407    136.045410   -0.213200
        # if args.save_image:
        #     for b in dets:
        #         # b :15个元素
        #         # if b[4] < args.vis_thres:
        #         if b[4] < aaa:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #         # bbb = b
        #         # print( bbb[0],bbb[1], bbb[2], bbb[3] )
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #     # save image
        #     if not os.path.exists("./results/"):
        #         os.makedirs("./results/")
        #     name = "./results/" + 'harry_test_ori' + ".jpg"
        #     cv2.imwrite(name, img_raw)
        #     cv2.imshow('harry_test', img_raw)
        #     cv2.waitKey(0)
        print('net forward time: {:.4f}'.format(time.time() - tic))
        print(conf)
        conf = F.softmax(conf, dim=-1)

        print(loc)
        print(conf)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to("cpu")
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to("cpu")
        # landms = landms * scale1 / resize
        # landms = landms.cpu().numpy()

        # ignore low scores
        print(scores)
        inds = np.where(scores > args.confidence_threshold)[0]
        print(inds)
        boxes = boxes[inds]
        # landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        # landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        # landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        print(dets)
        # show image
        if args.save_image:
            for b in dets:
                print(b[4])
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


                # landms
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            # name = "test.jpg"
            # cv2.imwrite(name, img_raw)
            cv2.imshow("TEST", img_raw)
            print("hi")
            cv2.waitKey(100)
            name = "./pc2/"+str(i.split("/")[-1][0:5])
            cv2.imwrite(name, img_raw)