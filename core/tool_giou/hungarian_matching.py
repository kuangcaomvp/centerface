import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from core.tool_giou import bbox


def np_tf_linear_sum_assignment(matrix):
    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]
    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)
    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    return target_indices.astype(np.int64), pred_indices.astype(np.int64), target_selector, pred_selector


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-6)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-6)

    return giou


def hungarian_matching(t_bbox, p_bbox):
    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = bbox.merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox, t_bbox)
    cost_giou = -bbox_giou(_p_bbox_xy, _t_bbox_xy)
    cost_matrix = cost_bbox + cost_giou

    target_indices, pred_indices, target_selector, pred_selector = tf.numpy_function(np_tf_linear_sum_assignment,
                                                                                     inp=[cost_matrix],
                                                                                     Tout=[tf.int64, tf.int64, tf.bool,
                                                                                           tf.bool])
    return pred_indices, target_indices, pred_selector, target_selector
