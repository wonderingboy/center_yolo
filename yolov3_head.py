import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from union_dataset import xywh2xyxy
from union_dataset import affine_transform

def yl_nms(yl_outputs, meta):
    all_boxes = yl_outputs[:, :4]
    all_boxes = xywh2xyxy(all_boxes)
    scores = yl_outputs[:, 4]
    keep = torchvision.ops.nms(all_boxes, scores, 0.4)

    all_boxes = all_boxes.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
    for p in range(all_boxes.shape[0]):
        all_boxes[p, 0:2] = affine_transform(
            all_boxes[p, 0:2], meta['inv_tran'])
        all_boxes[p, 2:] = affine_transform(all_boxes[p, 2:], meta['inv_tran'])
    final_results = []
    yl_outputs = yl_outputs.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    for i in range(keep.shape[0]):
        max_score = np.max(yl_outputs[keep[i], 5:])
        max_score_index = np.argwhere(yl_outputs[keep[i], 5:] == max_score)
        final_results.append([all_boxes[keep[i], 0], all_boxes[keep[i], 1], all_boxes[keep[i],
                             2], all_boxes[keep[i], 3], max_score*scores[i], max_score_index[0][0] ])
    return final_results

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        # non-zero power for gradient stability
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, paras, giou_flag=True):  # predictions, targets, model

    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(paras, targets)
    h = {'cls_pw': 1.0, 'obj_pw': 1.0, 'fl_gamma': 0.5}  # hyperparameters
    arc = 'default'  # # (default, uCE, uBCE) detection architectures
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    BCE = nn.BCEWithLogitsLoss(reduction=red)
    CE = nn.CrossEntropyLoss(reduction=red)  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(
            BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pxy = torch.sigmoid(ps[:, 0:2])
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i].cuda()
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            # giou computation
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 -
                                                             giou).mean()  # giou loss
            tobj[b, a, gj, gi] = giou.detach().clamp(
                0).type(tobj.dtype) if giou_flag else 1.0

            # cls loss (only if multiple classes)
            if 'default' in arc and paras['nc'] > 1:
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= 3.54
    lobj *= 64.3
    lcls *= 37.4
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None].cuda()  # [N,1,2]
    wh2 = wh2[None].cuda()  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # iou = inter / (area1 + area2 - inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def build_targets(params, targets):
    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = False
    reject, use_all_anchors = True, True
    for i in range(len(params['nx'])):
        # get number of grid points and anchor vec for this yolo layer
        ng = [params['nx'][i], params['ny'][i]]
        ng = torch.Tensor(ng).cuda()
        stride = params['stride'][i]
        anchor_vec = params['anchors'][i]/stride
        anchor_vec = torch.Tensor(anchor_vec)
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = wh_iou(torch.Tensor(anchor_vec), gwh)
            if use_all_anchors:
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                # iou threshold hyperparameter
                j = iou.view(-1) > 0.225
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < params['nc'], 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           params['nc'], params['nc'] - 1, c.max())

    return tcls, tbox, indices, av


def create_grids(self, img_size=(512, 512), ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)
    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(
        device).type(type).view((1, 1, ny, nx, 2))
    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(
        1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


class YOLOLayer(nn.Module):
    def __init__(self, anchors, strides, nc,  nx, ny, training=True, img_size=(512, 512)):
        super(YOLOLayer, self).__init__()
        # print(anchors,strides)
        self.anchors = torch.Tensor(anchors).cuda()
        self.strides = torch.Tensor(np.array([strides])).cuda()
        # print(77777,self.strides)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs
        self.nx = nx # initialize number of x gridpoints
        self.ny = ny  # initialize number of y gridpoints
        self.training = training
        self.img_size = img_size

    def forward(self, p, var=None):
        img_size = self.img_size
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(
            0, 1, 3, 4, 2).contiguous()  # prediction
        self.p = p
        return p

    def inference(self,thresh):
        anchor_vec = torch.tensor(self.anchors).cuda() / torch.tensor(self.strides).cuda()
        anchor_wh = anchor_vec.cuda().view(1, 3, 1, 1, 2).type(torch.float)
        yv, xv = torch.meshgrid([torch.arange(self.nx), torch.arange(self.ny)])
        grid_xy = torch.stack((xv, yv), 2).to('cuda').type( torch.float32).view((1, 1, self.nx, self.ny, 2))
        self.p[..., :2] = torch.sigmoid(self.p[..., :2]) + grid_xy
        self.p[..., 2:4] = torch.exp(self.p[..., 2:4]) * anchor_wh
        self.p[..., :4] *= self.strides
        torch.sigmoid_(self.p[..., 4:])
        if self.nc == 1:
            self.p[..., 5] = 1
        min_wh, max_wh = 2, 4096
        nc = self.p.shape[-1] - 5  # number of classes
        self.p = self.p[self.p[..., 4] > 0.1]
        self.p = self.p[(self.p[..., 2:4] > min_wh).all(1) & (self.p[..., 2:4] < max_wh).all(1)]
        return self.p.view(-1, self.p.shape[-1])
    
    def parse(self):
        pass


class YOLOHead(nn.Module):
    def __init__(self, anchors, strides, nc,  nx, ny, training=True, img_size=(512, 512)):
        super(YOLOHead, self).__init__()
        # print('555',strides)

        self.yolo1_1 = nn.Sequential()
        self.yolo1_1.add_module('Conv2d', nn.Conv2d(in_channels=512, out_channels=256,
                                                    kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo1_1.add_module(
            'BatchNorm2d', nn.BatchNorm2d(256, momentum=0.1))
        self.yolo1_1.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        self.yolo1_2 = nn.Sequential()
        self.yolo1_2.add_module('Conv2d_1', nn.Conv2d(in_channels=256, out_channels=128,
                                                      kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo1_2.add_module(
            'BatchNorm2d', nn.BatchNorm2d(128, momentum=0.1))
        self.yolo1_2.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
        self.yolo1_2.add_module('Conv2d_2', nn.Conv2d(in_channels=128, out_channels=(nc+5)*3,
                                                      kernel_size=3, stride=1, padding=1, groups=1, bias=True))
        self.yolo1_layer = YOLOLayer(
            anchors[0], strides[0], nc, nx[0], ny[0], training, img_size)

        self.yolo2_1 = nn.Sequential()
        self.yolo2_1.add_module('Conv2d', nn.Conv2d(in_channels=512, out_channels=256,
                                                    kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo2_1.add_module(
            'BatchNorm2d', nn.BatchNorm2d(256, momentum=0.1))
        self.yolo2_1.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        self.yolo2_2 = nn.Sequential()
        self.yolo2_2.add_module('Conv2d_1', nn.Conv2d(in_channels=256, out_channels=128,
                                                      kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo2_2.add_module(
            'BatchNorm2d', nn.BatchNorm2d(128, momentum=0.1))
        self.yolo2_2.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
        self.yolo2_2.add_module('Conv2d_2', nn.Conv2d(in_channels=128, out_channels=(nc+5)*3,
                                                      kernel_size=3, stride=1, padding=1, groups=1, bias=True))
        self.yolo2_layer = YOLOLayer(
            anchors[1], strides[1], nc,  nx[1], ny[1], training, img_size)

        self.yolo3 = nn.Sequential()
        self.yolo3.add_module('Conv2d_1', nn.Conv2d(in_channels=384, out_channels=256,
                                                    kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo3.add_module(
            'BatchNorm2d_1', nn.BatchNorm2d(256, momentum=0.1))
        self.yolo3.add_module('activation_1', nn.LeakyReLU(0.1, inplace=True))

        self.yolo3.add_module('Conv2d_2', nn.Conv2d(in_channels=256, out_channels=128,
                                                    kernel_size=3, stride=1, padding=1, groups=1, bias=False))
        self.yolo3.add_module(
            'BatchNorm2d_2', nn.BatchNorm2d(128, momentum=0.1))
        self.yolo3.add_module('activation_2', nn.LeakyReLU(0.1, inplace=True))
        self.yolo3.add_module('Conv2d_3', nn.Conv2d(in_channels=128, out_channels=(nc+5)*3,
                                                    kernel_size=3, stride=1, padding=1, groups=1, bias=True))
        self.yolo3_layer = YOLOLayer(
            anchors[2], strides[2], nc, nx[2], ny[2], training, img_size)

    def forward(self, x):
        x1_, x2_, x3_ = x
        x1_1 = self.yolo1_1(x1_)
        x1_2 = self.yolo1_2(x1_1)

        x2 = nn.Upsample(scale_factor=2)(x1_1)
        x2 = torch.cat([x2_, x2], dim=1)

        x2_1 = self.yolo2_1(x2)
        x2_2 = self.yolo2_2(x2_1)

        x3 = nn.Upsample(scale_factor=2)(x2_1)
        x3 = torch.cat([x3_, x3], dim=1)
        x3_2 = self.yolo3(x3)
        self.out1, self.out2, self.out3 = self.yolo1_layer(x1_2), self.yolo2_layer(x2_2), self.yolo3_layer(x3_2)
        return self.out1, self.out2, self.out3

    def inference(self, thresh):
        dets=[self.yolo1_layer.inference(thresh), self.yolo2_layer.inference(thresh), self.yolo3_layer.inference(thresh)]
        return  torch.cat(dets,0)
