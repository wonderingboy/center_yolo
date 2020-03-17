import sys
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
# import _init_paths
import resnet_backbone
import centernet_head
# import centernet_loss
import yolov3_head
import union_dataset
from union_dataset import affine_transform
from union_dataset import get_affine_transform
from config import *
from union_dataset import pre_process
from yolov3_head import yl_nms
from union_utils import merge_yl_ct_dets, parse_data_label, update_stastics


class CombinationModel(nn.Module):
    def __init__(self, backbone_layer, num_classes):
        super(CombinationModel, self).__init__()
        self.backbone = resnet_backbone.get_pose_net(backbone_layer)
        self.ct_head = centernet_head.CenterNetHead(num_classes)
        self.yolov3_head = yolov3_head.YOLOHead(
            anchors, strides, num_classes, nx, ny)

    def forward(self, inp):
        bko = self.backbone(inp)
        return self.ct_head(bko[-1]), self.yolov3_head(bko[::-1])


model = CombinationModel(18, 1)
model.load_state_dict(torch.load(sys.argv[1]))
model.eval().cuda()
f = open(sys.argv[2])
dataset = f.readlines()
f.close()

statics = {}
for i in range(class_nums):
    statics[i] = {'ground_nums': 0, 'recall_ground_nums': 0, 'detection_nums': 0, 'correct_detection_nums': 0}

for data in dataset:
    data = data.strip('\n')
    data_label = data[:-3]+'txt'
    img = cv2.imread(data)
    height, width, _ = img.shape
    final_labels = parse_data_label(data_label, height, width)
    img, meta = pre_process(img)
    _ = model(img.cuda())
    cd_dets = model.ct_head.inference(0.3, meta)
    yl_dets = model.yolov3_head.inference(0.3)
    yl_dets = yl_nms(yl_dets, meta)
    # print(data)
    final_dets = merge_yl_ct_dets(yl_dets, cd_dets)
    statics = update_stastics(statics, final_labels, yl_dets)
print(statics)