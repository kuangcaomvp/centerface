#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg


def image_progress(img0, size=[416, 416]):
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img, _, _, _ = utils.letterbox(img0, height=size[0], width=size[1])
    img = img.astype(np.float32)
    img /= 255.0
    img = img[np.newaxis, ...]

    width = img0.shape[1]
    height = img0.shape[0]
    inp_height = img.shape[1]
    inp_width = img.shape[2]
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = [width / 2., height / 2., s,
            inp_height // 4,
            inp_width // 4]
    return img, img0, np.array(meta, dtype=np.float32)


pb_file = "checkpoint_dcn2/face.pb"
num_classes = len(utils.read_class_names(cfg.BASE.CLASSES))

return_elements = ["input/input_data:0", "decode/concat_2:0",
                   "decode/Reshape_18:0", "decode/BatchGatherND_6/Reshape_3:0"]

graph = tf.Graph()


return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(0)
    index = 0
    while True:
        index += 1
        return_value, frame = vid.read()
        if return_value:
            img, img0, meta = image_progress(frame, [416, 416])
        else:
            raise ValueError("No image!")

        dets, points, id_feature = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: img})

        remain_inds = (dets[..., 4] > 0.6) & (dets[..., 5] == 0.0)
        dets = dets[remain_inds]
        points = points[remain_inds]
        # 原始图片展示
        dets[..., :2] = utils.transform_preds(
            dets[..., 0:2], np.array([meta[0], meta[1]]), meta[2], (meta[3], meta[4]))
        dets[..., 2:4] = utils.transform_preds(
            dets[..., 2:4], np.array([meta[0], meta[1]]), meta[2], (meta[3], meta[4]))

        for i in range(5):
            points[..., 2 * i:2 * (i + 1)] = utils.transform_preds(
                points[..., 2 * i:2 * (i + 1)], np.array([meta[0], meta[1]]), meta[2], (meta[3], meta[4]))

        im = img0.copy()
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(im, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0), 2)
            pt = points[i]
            p = np.reshape(pt, (-1, 2))
            for k in p:
                cv2.circle(im, (int(k[0]), int(k[1])), 1, (255, 0, 0), 4)
        cv2.imshow('dets', im)
        cv2.waitKey(1)

