#! /usr/bin/env python
# coding=utf-8

import math

import tensorflow.compat.v1 as tf

import core.utils as utils
from core.config import cfg
from core.dcnv2.hrnet import seg_hrnet
from core.tool_giou import bbox
from core.tool_giou.hungarian_matching import hungarian_matching, bbox_giou


class Face(object):
    def __init__(self, input_data, trainable, nid=572, drop=0.0):

        self.trainable = trainable
        self.dropout = drop
        self.classes = utils.read_class_names(cfg.BASE.CLASSES)
        self.num_class = len(self.classes)
        self.nid = nid
        self.emb_scale = math.sqrt(2) * math.log(self.nid - 1) if self.nid > 1 else 1
        self.s_det = -1.85
        self.s_id = -1.05
        self.hm_weight = 1
        self.wh_weight = 1
        self.off_weight = 1
        self.out_size = cfg.TRAIN.INPUT_SIZE
        self.hm, self.wh, self.reg, self.pts, self.rid = self.__build_nework(input_data)
        with tf.variable_scope('decode'):
            self.dets, self.inds, self.id_feature, self.points = self.decode(self.hm, self.wh, self.reg, self.rid,
                                                                             self.pts)

    def loss_boxes(self, p_bbox, t_bbox, t_indices, p_indices):
        p_bbox = tf.gather(p_bbox, p_indices)
        t_bbox = tf.gather(t_bbox, t_indices)
        l1_loss = self.smooth_l1_loss(t_bbox, p_bbox)
        l1_loss = tf.reduce_sum(l1_loss) / (tf.cast(tf.shape(p_bbox)[0], tf.float32) + 1e-6)

        _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox, t_bbox)
        giou = bbox_giou(_p_bbox_xy, _t_bbox_xy)
        loss_giou = 1 - tf.linalg.diag_part(giou)
        loss_giou = tf.reduce_sum(loss_giou) / (tf.cast(tf.shape(p_bbox)[0], tf.float32) + 1e-6)

        return loss_giou, l1_loss

    def handel(self, target_bbox, reg_mask):
        t_offset = 0
        p_offset = 0
        all_target_bbox = []
        all_predicted_bbox = []
        all_target_indices = []
        all_predcted_indices = []
        for i in range(cfg.TRAIN.BATCH_SIZE):
            p_bbox, t_bbox = self.dets[i][..., :4], target_bbox[i][reg_mask[i] > 0]
            t_indices, p_indices, t_selector, p_selector = hungarian_matching(t_bbox, p_bbox)
            t_indices = t_indices + tf.cast(t_offset, tf.int64)
            p_indices = p_indices + tf.cast(p_offset, tf.int64)
            all_target_bbox.append(t_bbox)
            all_predicted_bbox.append(p_bbox)
            all_target_indices.append(t_indices)
            all_predcted_indices.append(p_indices)
            t_offset += tf.shape(t_bbox)[0]
            p_offset += tf.shape(p_bbox)[0]
        all_target_bbox = tf.concat(all_target_bbox, axis=0)
        all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
        all_target_indices = tf.concat(all_target_indices, axis=0)
        all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
        giou_loss, l1_loss = self.loss_boxes(
            all_predicted_bbox,
            all_target_bbox,
            all_target_indices,
            all_predcted_indices,
        )
        return giou_loss, l1_loss

    # feat: bhwc  ind: bm
    def tranpose_and_gather_feat(self, feat, ind):
        shape = tf.shape(feat)
        dim = feat.shape[-1]
        rid = tf.reshape(feat, [shape[0], -1, dim])
        con = tf.expand_dims(ind, axis=-1)
        out = tf.gather_nd(rid, con, batch_dims=1)
        return out

    # reid 后处理
    def enc_rid(self, ind, reg_mask):
        out = self.tranpose_and_gather_feat(self.rid, ind)
        # out = out[reg_mask > 0]
        out = self.emb_scale * tf.nn.l2_normalize(out, axis=1)
        out = tf.layers.dense(out, self.nid)
        return out

    def __build_nework(self, input_data):
        hr = seg_hrnet(input_data)
        hm = tf.layers.conv2d(hr, self.num_class, 1, 1)
        wh = tf.layers.conv2d(hr, 2, 1, 1)
        reg = tf.layers.conv2d(hr, 2, 1, 1)
        pts = tf.layers.conv2d(hr, 10, 1, 1)
        rid = tf.layers.conv2d(hr, 128, 1, 1)
        # rid = tf.layers.dropout(rid,
        #                         rate=self.dropout,
        #                         training=tf.convert_to_tensor(self.trainable))
        return hm, wh, reg, pts, rid

    def loss(self, hm, wh, reg_mask, ind, reg, ids, bbox, pts):
        with tf.name_scope(name='loss'):
            loss_hm = self.focal_loss(self.hm, hm)
            loss_wh = self.RegL1Loss(self.wh, reg_mask, ind, wh)
            loss_offset = self.RegL1Loss(self.reg, reg_mask, ind, reg)
            loss_pts = self.RegL1Loss(self.pts, reg_mask, ind, pts)
            id_out = self.enc_rid(ind, reg_mask)
            loss_ids = self.compute_loss(id_out, ids)
            det_loss = self.hm_weight * loss_hm + self.wh_weight * loss_wh + self.off_weight * loss_offset + loss_pts
            giou_loss, l1_loss = self.handel(bbox, reg_mask)
            # loss = tf.exp(-self.s_det) * det_loss + tf.exp(-self.s_id) * loss_ids + (self.s_det + self.s_id)
            # loss *= 0.5
            # giou可以修改为ciou或者diou 具体效果需要测试才知道 不要什么都听论文说的 要动手测试看看  操
            det_loss += giou_loss + l1_loss
            loss = det_loss + loss_ids
            #不好用 loss会变负的 但是测试效果还不错 就是loss负的看的不舒服
            # loss_hander = MultiLossLayer([det_loss, loss_ids])
            # loss = loss_hander.get_loss()
        return loss, loss_hm, loss_wh, loss_offset, loss_ids, giou_loss, l1_loss, loss_pts

    def _nms(self, heat, kernel=3):
        hmax = tf.nn.max_pool2d(heat, kernel, strides=1, padding='SAME')
        heat_z = tf.zeros_like(heat, tf.float32)
        out = tf.where(tf.equal(hmax, heat), hmax, heat_z)
        return out

    def _topk(self, scores, K=40):
        shape = tf.shape(scores)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        cat = shape[-1]
        scores = tf.reshape(scores, (batch, -1, cat))
        scores = tf.transpose(scores, (0, 2, 1))
        topk_scores, topk_inds = tf.nn.top_k(scores, K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width)
        topk_ys = tf.cast(topk_ys, tf.int32)
        topk_ys = tf.cast(topk_ys, tf.float32)
        topk_xs = (topk_inds % width)
        topk_xs = tf.cast(topk_xs, tf.int32)
        topk_xs = tf.cast(topk_xs, tf.float32)

        topk_score, topk_ind = tf.nn.top_k(tf.reshape(topk_scores, (batch, -1)), K)
        topk_clses = (topk_ind / K)
        topk_clses = tf.cast(topk_clses, tf.int32)
        topk_ind = tf.expand_dims(topk_ind, axis=-1)
        topk_inds = tf.reshape(topk_inds, (batch, -1, 1))
        topk_inds = tf.gather_nd(topk_inds, topk_ind, batch_dims=1)
        topk_inds = tf.reshape(topk_inds, (batch, K))
        topk_ys = tf.reshape(topk_ys, (batch, -1, 1))
        topk_ys = tf.gather_nd(topk_ys, topk_ind, batch_dims=1)
        topk_ys = tf.reshape(topk_ys, (batch, K))
        topk_xs = tf.reshape(topk_xs, (batch, -1, 1))
        topk_xs = tf.gather_nd(topk_xs, topk_ind, batch_dims=1)
        topk_xs = tf.reshape(topk_xs, (batch, K))
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def decode(self, hm, wh, reg, rid, pts, k=cfg.TRAIN.topK):
        out_hm = tf.sigmoid(hm)
        id_feature = tf.nn.l2_normalize(rid, axis=1)
        shape = tf.shape(out_hm)
        batch = shape[0]
        heat = self._nms(out_hm)
        scores, inds, clses, ys, xs = self._topk(heat, K=k)

        reg = self.tranpose_and_gather_feat(reg, inds)
        reg = tf.reshape(reg, (batch, k, 2))
        xs = tf.reshape(xs, (batch, k, 1)) + reg[:, :, 0:1]
        ys = tf.reshape(ys, (batch, k, 1)) + reg[:, :, 1:2]
        wh = self.tranpose_and_gather_feat(wh, inds)
        wh = tf.reshape(wh, (batch, k, 2))
        clses = tf.reshape(clses, (batch, k, 1))
        clses = tf.cast(clses, tf.float32)
        scores = tf.reshape(scores, (batch, k, 1))
        bboxes = tf.concat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], axis=2)
        pts = self.tranpose_and_gather_feat(pts, inds)
        pts = tf.reshape(pts, (batch, k, 5, 2))
        points = tf.expand_dims(tf.concat([xs, ys], axis=2), axis=-2) + pts
        points = tf.reshape(points, (batch, k, 10))
        dets = tf.concat([bboxes, scores, clses], axis=2)
        id_feature = self.tranpose_and_gather_feat(id_feature, inds)
        return dets, inds, id_feature, points

    def label_smoothing(self, inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / (K + 1e-6))

    def compute_loss(self, logits, y):
        y_smoothed = self.label_smoothing(tf.one_hot(y, depth=self.nid))
        istarget = tf.to_float(tf.not_equal(y, -1))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
        mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget) + 1e-6)
        return mean_loss

    def mse_loss(self, pred, target):
        pred = tf.sigmoid(pred)
        loss = tf.reduce_sum(tf.pow(pred - target, 2))
        return loss

    # 计算 heatmap loss
    def focal_loss(self, pred, gt, esp=1e-4):
        pred = tf.sigmoid(pred)
        pred = tf.clip_by_value(pred, esp, 1 - esp)
        pos_inds = tf.cast(tf.equal(gt, 1), tf.float32)
        neg_inds = tf.cast(tf.less(gt, 1), tf.float32)

        neg_weights = tf.pow(1 - gt, 4)

        loss = 0
        pos_loss = tf.log(pred) * tf.pow(1 - pred, 2) * pos_inds
        neg_loss = tf.log(1 - pred) * tf.pow(pred, 2) * neg_weights * neg_inds

        num_pos = tf.reduce_sum(pos_inds)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def RegL1Loss(self, output, mask, ind, target):
        pred = self.tranpose_and_gather_feat(output, ind)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, (1, 1, pred.shape[-1]))
        mask = tf.cast(mask, tf.float32)
        loss = self.smooth_l1_loss(target * mask, pred * mask)
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-4)
        return loss

    def smooth_l1_loss(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        return loss


