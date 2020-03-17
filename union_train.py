import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# import _init_paths
import resnet_backbone
import centernet_head
# import centernet_loss
import yolov3_head
import union_dataset

from config import *

class CombinationModel(nn.Module):
    def __init__(self, backbone_layer, num_classes, anchors, nx, ny):
        super(CombinationModel, self).__init__()
        self.backbone = resnet_backbone.get_pose_net(backbone_layer)
        self.ct_head = centernet_head.CenterNetHead(num_classes)
        self.yolov3_head = yolov3_head.YOLOHead(anchors, num_classes, nx, ny)
    def forward(self, inp):
        bko = self.backbone(inp)
        return self.ct_head(bko[-1]), self.yolov3_head(bko[::-1])


model = CombinationModel(arch, class_nums, anchors, nx, ny)
Closs = centernet_head.CenterNetLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 1.25e-4)
dataset = union_dataset.CTDetDatasetTxt('../train.txt', class_nums)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=union_dataset.CTDetDatasetTxt.collate_fn
)
model.train().cuda()

for j in range(60):
    for i, batch in enumerate(train_loader):
        if i == len(train_loader) - 1:
            continue
        outputs = model(batch['input'].cuda())
        ctloss, ctloss_stats = Closs(outputs[0][0], batch)
        ylloss, ylloss_stats = yolov3_head.compute_loss(outputs[1], batch['yolov3'].cuda(), paras)
        optimizer.zero_grad()
        tloss = ctloss+ylloss
        tloss.backward()
        optimizer.step()
        if i%10 == 0:
            print(i, "------>", tloss,ctloss,ylloss)
    if j%10==0:
        torch.save(model.state_dict(), 'params_'+ str(j) +'.pkl')
