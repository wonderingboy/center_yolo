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

class CombinationModel(nn.Module):
    def __init__(self, backbone_layer, num_classes):
        super(CombinationModel, self).__init__()
        self.backbone = resnet_backbone.get_pose_net(backbone_layer)
        self.ct_head = centernet_head.CenterNetHead(num_classes)
        self.yolov3_head = yolov3_head.YOLOHead(anchors, stride, num_classes, nx, ny)

    def forward(self, inp):
        bko = self.backbone(inp)
        return self.ct_head(bko[-1]), self.yolov3_head(bko[::-1])


model = CombinationModel(18, 1)
model.load_state_dict(torch.load('params.pkl'))
model.eval().cuda()

img_ = cv2.imread(sys.argv[1])
img, yl_image, meta = pre_process(img_)

outputs = model(img.cuda())
cd_dets = model.ct_head.inference(0.3, meta)

for bbox in cd_dets:
    print(bbox)
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
    cv2.putText(img_, str( bbox[5] )+':'+str(bbox[4])[:3], (bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),1)

yl_dets = model.yolov3_head.inference(0.3)
yl_dets = yl_nms(yl_dets, meta)
for bbox in yl_dets:
    print(bbox)
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 4)
    cv2.putText(img_, str( bbox[5] )+':'+str(bbox[4])[:3], (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),1)
cv2.imshow('hhh', img_)
cv2.waitKey(-1)
