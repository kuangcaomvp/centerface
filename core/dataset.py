#! /usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import random

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg
import random


class Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.height = self.train_input_size[0]
        self.width = self.train_input_size[1]
        self.classes = utils.read_class_names(cfg.BASE.CLASSES)
        self.num_classes = len(self.classes)
        self.max_objs = cfg.TRAIN.topK

        self.annotations = self.load_annotations()
        self.nid = self.init_nid()
        self.num_samples = len(self.annotations)
        self.batch_num = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.dataset = tf.data.Dataset.from_generator(self.generate_values,
                                                      (tf.string, tf.float32)) \
            .map(self.wrapped_complex_calulation, num_parallel_calls=8) \
            .filter(self.filter_fn) \
            .padded_batch(self.batch_size,
                          padded_shapes=(
                              [self.height, self.width, 3],
                              [self.height // 4, self.width // 4, self.num_classes],
                              [self.max_objs],
                              [self.max_objs],
                              [self.max_objs, 2],
                              [self.max_objs, 2],
                              [self.max_objs],
                              [self.max_objs, 4],
                              [None],
                              [self.max_objs, 10],),
                          padding_values=(0.0, 0.0, 0, 0, 0.0, 0.0, -1, 0.0, 0, 0.0)) \
            .prefetch(8)

        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        # 每次结束后sess,run初始化下这个变量 可以重置数据集 不然数据只能跑一次
        self.init_op = self.iterator.initializer

    def filter_fn(self, img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, lenth, pts):

        return tf.greater(lenth, 0)

    def load_annotations(self):
        L = []
        for i in self.annot_path:
            with open(i, 'r') as f:
                txt = f.readlines()
                annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
                L.extend(annotations)
        return L

    def init_nid(self):
        L = []
        for i in self.annotations:
            bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in i.strip().split()[1:]])
            tids = bboxes[:, 1]
            ind = np.argmax(tids)
            L.append(tids[ind])
        L = np.array(L)
        ind_max = np.argmax(L)
        return L[ind_max] + 2

    def __len__(self):
        return self.batch_num

    def __iter__(self):
        return self

    def generate_values(self):
        n_ = [i for i in range(len(self.annotations))]
        random.shuffle(n_)
        for j in n_:
            line = self.annotations[j].strip().split()
            image_path = line[0]
            bboxes = np.array([list(map(lambda x: float(x), box.split(','))) for box in line[1:]])
            yield image_path, bboxes

    def wrapped_complex_calulation(self, image_path, bboxes):
        img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, lenth, pts \
            = tf.numpy_function(func=self.parse_annotation,
                                inp=(image_path, bboxes),
                                Tout=(tf.float32, tf.float32, tf.int32,
                                      tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32))
        return img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, lenth, pts

    def draw_msra_gaussian(self, heatmap, center, sigma):
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[1], heatmap.shape[0]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] < 0 or ul[1] < 0 or br[0] >= w or br[1] >= h:
            return heatmap
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def xyxy2xywh(self, x):
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = np.zeros_like(x[:, :4])
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return np.concatenate([y, x[:, 4:]], axis=-1)

    def xywh2xyxy(self, x):
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = np.zeros_like(x)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2)
        y[:, 1] = (x[:, 1] - x[:, 3] / 2)
        y[:, 2] = (x[:, 0] + x[:, 2] / 2)
        y[:, 3] = (x[:, 1] + x[:, 3] / 2)
        return y

    # bboxes x,y,w,h
    def crop_bbox(self, bboxes, image):
        h, w, _ = image.shape
        box_1 = self.xywh2xyxy(bboxes[:, :4])
        box_1[:, [0, 2]] = box_1[:, [0, 2]] * w
        box_1[:, [1, 3]] = box_1[:, [1, 3]] * h
        box_1[:, [0, 2]] = np.clip(box_1[:, [0, 2]], 0, w - 1)
        box_1[:, [1, 3]] = np.clip(box_1[:, [1, 3]], 0, h - 1)
        box_2 = bboxes[:, 4:]
        box_2[:, 0::2] = box_2[:, 0::2] * w
        box_2[:, 1::2] = box_2[:, 1::2] * h
        return np.concatenate([box_1, box_2], axis=-1)

    def random_affine(self, img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      borderValue=(127.5, 127.5, 127.5)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        if targets is not None:
            if len(targets) > 0:
                n = targets.shape[0]
                points = targets[:, 2:6].copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 4)

                targets = targets[i]
                targets[:, 2:6] = xy[i]

            return imw, targets, M
        else:
            return imw

    def parse_annotation(self, image_path, bboxes):

        if type(image_path) is not str:
            image_path = str(image_path, 'utf-8')

        if not os.path.isfile(image_path):
            raise ValueError('"{}" does not exist.'.format(image_path))

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError('File corrupt {}'.format(image_path))

        # h, w, _ = image.shape
        img, ratio, padw, padh = self.letterbox(image, height=self.height, width=self.width)

        # xywh 转xyxy 超出图片部分剪切
        bb = self.crop_bbox(bboxes[:, 2:], image)
        # Normalized xyxy
        labels = bboxes.copy()
        labels[:, 2::2] = ratio * bb[:, 0::2] + padw
        labels[:, 3::2] = ratio * bb[:, 1::2] + padh
        # if cfg.TRAIN.DATA_AUG and random.uniform(0, 1) > 0.5:
        #     img, labels, M = self.random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10),
        #                                         scale=(0.50, 1.20))
        labels[:, 2::2] = np.clip(labels[:, 2::2], 0, self.width - 1)
        labels[:, 3::2] = np.clip(labels[:, 3::2], 0, self.height - 1)

        labels[:, 2:] = self.xyxy2xywh(labels[:, 2:])
        labels[:, 2::2] = labels[:, 2::2] / self.width
        labels[:, 3::2] = labels[:, 3::2] / self.height
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        img = img.astype(np.float32)
        img /= 255.0
        # 下采样4倍
        output_h = self.height // 4
        output_w = self.width // 4
        # heatmap 中心点分布
        hm = np.zeros((output_h, output_w, self.num_classes), dtype=np.float32)
        # wh长宽比例
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # offset 中心点偏执
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        # 5个关键点相对于中心点的偏置
        pts = np.zeros((self.max_objs, 10), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int32)
        # len(bboxs) < self.max_objs 有box标记1 没有0
        reg_mask = np.zeros((self.max_objs,), dtype=np.int32)
        # 类别
        ids = np.ones((self.max_objs,), dtype=np.int32) * (-1)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)
        num_objs = labels.shape[0]
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:6]
            pt = label[6:]
            pt[0::2] = pt[0::2] * output_w
            pt[1::2] = pt[1::2] * output_h
            pt = np.reshape(pt, (-1, 2))
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox[0] = np.clip(bbox[0], 0, output_w)
            bbox[1] = np.clip(bbox[1], 0, output_h)
            bbox_xy = np.copy(bbox)
            bbox_xy[0] = (bbox_xy[0] - bbox_xy[2] / 2)
            bbox_xy[1] = (bbox_xy[1] - bbox_xy[3] / 2)
            bbox_xy[2] = (bbox_xy[0] + bbox_xy[2])
            bbox_xy[3] = (bbox_xy[1] + bbox_xy[3])
            bbox_xy[0::2] = np.clip(bbox_xy[0::2], 0, output_w - 1)
            bbox_xy[1::2] = np.clip(bbox_xy[1::2], 0, output_h - 1)
            bbox[2] = bbox_xy[2] - bbox_xy[0]
            bbox[3] = bbox_xy[3] - bbox_xy[1]
            bbox[0] = bbox_xy[0] + bbox[2] / 2
            bbox[1] = bbox_xy[1] + bbox[3] / 2
            if bbox[3] > 4 and bbox[2] > 4 and bbox[1] > 0 and bbox[0] > 0:
                radius = 1
                # xy 中心点 float
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                # xy 中心点 int
                ct_int = ct.astype(np.int32)
                pts[k] = np.reshape(pt - ct, (-1))
                self.draw_msra_gaussian(hm[..., cls_id], ct_int, radius)
                wh[k] = 1. * bbox[2], 1. * bbox[3]
                # h*w展开后的坐标
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # offset 中心点偏执
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = int(label[1])  # if int(label[1]) != -1 else self.nid + 10
                bbox_xys[k] = bbox_xy
        lenth = np.array([bbox_xys[reg_mask > 0].shape[0]], np.int32)
        return img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, lenth, pts

    def letterbox(self, img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh
