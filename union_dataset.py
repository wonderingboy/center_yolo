from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math


import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data



def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def pre_process(image, scale=1, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    inp_height, inp_width = 512, 512
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inv_trans = get_affine_transform(
        c, s, 0, [inp_width, inp_height], inv=True)
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image_ = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    inp_image = inp_image_
    inp_image = inp_image / 255.
    inp_image = inp_image.transpose(
        2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(inp_image)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4,
            'inv_tran': inv_trans}
    return images.float(), inp_image_, meta

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]    

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

class COCO(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        # self.data_dir = os.path.join(opt.data_dir, 'coco')
        # self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        # if split == 'test':
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'annotations',
        #         'image_info_test-dev2017.json').format(split)
        # else:
        #     if opt.task == 'exdet':
        #         self.annot_path = os.path.join(
        #             self.data_dir, 'annotations',
        #             'instances_extreme_{}2017.json').format(split)
        #     else:
        #         self.annot_path = os.path.join(
        self.class_name = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.mean = np.array([0.485, 0.456, 0.406],
                             np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split

        print('==> initializing coco 2017 {} data.'.format(split))
        # self.coco = coco.COCO(self.annot_path)
        # self.images = self.coco.getImgIds()
        # self.num_samples = len(self.all_images)

        # print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
               * x).clip(None, 255).astype(np.uint8)
    # inplace hue clip (0 - 179 deg)
    np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


class CTDetDatasetTxt(COCO):
    def _coco_box_to_bbox(self, box, height, width):
        bbox = np.array([(box[1]-box[3]/2)*width, (box[2]-box[4]/2)*height,
                         (box[1]+box[3]/2)*width, (box[2]+box[4]/2)*height], dtype=np.float32)
        return bbox

    def load_labels(self):
        self.all_labels = []
        for image in self.all_images:
            f = open(image[:-3]+'txt')
            lines = f.readlines()
            tmp_labels = []
            for line in lines:
                line = line.strip('\n').split(' ')
                tmp_labels.append([int(line[0]), float(line[1]), float(
                    line[2]), float(line[3]), float(line[4])])
            self.all_labels.append(tmp_labels)

    def __len__(self):
        return len(self.all_labels)

    def __init__(self, list_file, num_classes, scale=0.4, shift=0.1, input_h=512, input_w=512, down_ratio=4):
        super(CTDetDatasetTxt, self).__init__(None, list_file[:-4])
        self.list_file = list_file
        self.num_classes=num_classes
        self.shift = shift
        self.scale = scale
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio

        self.max_objs = 20
        self.all_labels = []
        f = open(list_file)
        self.all_images = f.readlines()
        self.all_images = [image.strip('\n') for image in self.all_images]
        self.load_labels()

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_path = self.all_images[index]
        anns = self.all_labels[index]
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([width / 2., height / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.input_h, self.input_w

        flipped = False
        if 'train' in self.list_file:
            sf = self.scale  # 0.4
            cf = self.shift  # 0.1
            c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        # cv2.imwrite("result.jpg", inp)

        if 'train' in self.list_file:
            augment_hsv(inp)
        inp = (inp.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        gt_det = []
        yolov3_labels = torch.zeros((num_objs, 6))
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann, height, width)
            cls_id = ann[0]
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                yolov3_labels[k, 1] = cls_id
                yolov3_labels[k, 2] = (bbox[0] + bbox[2])/2/output_w
                yolov3_labels[k, 3] = (bbox[1] + bbox[3])/2/output_h
                yolov3_labels[k, 4] = w/output_w
                yolov3_labels[k, 5] = h/output_h
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1

                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        ret = {'input': inp, 'hm': hm,
               'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'yolov3': yolov3_labels}
        ret.update({'reg': reg})
        return ret

    @staticmethod
    def collate_fn(batch):
        inputs = []
        hms = []
        reg_masks = []
        inds = []
        whs = []
        yolov3s = []
        regs = []
        for i, item in enumerate(batch):
            inputs.append(torch.from_numpy(item['input']).unsqueeze(0))
            hms.append(torch.from_numpy(item['hm']).unsqueeze(0))
            reg_masks.append(torch.from_numpy(item['reg_mask']).unsqueeze(0))
            inds.append(torch.from_numpy(item['ind']).unsqueeze(0))
            whs.append(torch.from_numpy(item['wh']).unsqueeze(0))
            regs.append(torch.from_numpy(item['reg']).unsqueeze(0))
            l = item['yolov3']
            l[:, 0] = i
            yolov3s.append(l)

        return {'input': torch.cat(inputs),
                'hm': torch.cat(hms),
                'reg_mask': torch.cat(reg_masks),
                'ind': torch.cat(inds),
                'wh': torch.cat(whs),
                'yolov3': torch.cat(yolov3s),
                'reg': torch.cat(regs)
                }
