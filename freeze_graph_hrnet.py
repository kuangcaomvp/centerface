#! /usr/bin/env python
# coding=utf-8

import tensorflow.compat.v1 as tf

from core.model import Face

pb_file = "checkpoint_dcn2/face.pb"
ckpt_file = ".\checkpoint_dcn2\loss=1.4921.ckpt-20"
output_node_names = ["input/input_data", "decode/concat_2", "decode/Reshape_18", "decode/BatchGatherND_6/Reshape_3"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='input_data')
    # con = tf.placeholder(dtype=tf.int32, shape=(None, 150), name='con')
    # mask = tf.placeholder(dtype=tf.int32, shape=(None, None), name='reg_mask')
    # hm = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name='hm')
    # wh = tf.placeholder(dtype=tf.float32, shape=(None, None, 2), name='wh')
    # reg = tf.placeholder(dtype=tf.float32, shape=(None, None, 2), name='reg')
    # ids = tf.placeholder(dtype=tf.float32, shape=(None, 150), name='ids')
    # meta = tf.placeholder(dtype=tf.float32, name='meta')
    # bbox = tf.placeholder(dtype=tf.float32, shape=(None, None, 4), name='box')
model = Face(input_data, trainable=False, nid=572)
print(model.dets)
print(model.id_feature)
print(model.points)
# out = model.handel(bbox, mask)
# print(out)

#
#
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())

#
# import  cv2
# import numpy as np
#
#
# def letterbox(img, height=608, width=1088,
#               color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
#     shape = img.shape[:2]  # shape = [height, width]
#     ratio = min(float(height) / shape[0], float(width) / shape[1])
#     new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
#     dw = (width - new_shape[0]) / 2  # width padding
#     dh = (height - new_shape[1]) / 2  # height padding
#     top, bottom = round(dh - 0.1), round(dh + 0.1)
#     left, right = round(dw - 0.1), round(dw + 0.1)
#     img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
#     return img, ratio, dw, dh
#
# original_image = cv2.imread('1.jpg')
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# img, _, _, _ = letterbox(original_image, height=416, width=608)
# # Normalize RGB
# img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
# img = img.astype(np.float32)
# img /= 255.0
# img = img[np.newaxis, ...]
# out = sess.run('wh/LeakyRelu:0', {input_data:img})
# print(out)
# print(1/(1+np.exp(out)))
# testset = Dataset('train')
# sess.run(testset.init_op)
# while True:
#     try:
#         img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, _ = sess.run(testset.next_element)
#         out, a, b = sess.run(['add_29:0', 'truediv_7:0', 'truediv_4:0'],
#                              {input_data: img, bbox: bbox_xys, mask: reg_mask})
#         print(out.shape)
#         if out.shape[1] == 0:
#             print(_)
#             print(out)
#             o = np_tf_linear_sum_assignment(out)
#             print(o[0])
#             print(a)
#             print(b)
#             print(reg_mask)
#     except tf.errors.OutOfRangeError:
#         # print("End of dataset")
#         break
