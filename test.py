import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from core.dataset import Dataset

if __name__ == "__main__":
    testset = Dataset('test')
    print(testset.nid)
    sess = tf.InteractiveSession()
    count = 0
    sess.run(testset.init_op)
    t1 = time.time()
    while True:
        try:
            img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, lenth, pts = sess.run(testset.next_element)
            for j in range(hm.shape[0]):
                bbox = bbox_xys[j][reg_mask[j] > 0]

                pt = pts[j][reg_mask[j] > 0]
                print(bbox)
                print(pt)
                plt.subplot(111)
                plt.imshow(hm[j][..., 0])
                plt.colorbar(cax=None, ax=None, shrink=0.5)
                plt.show()
                image = img[j]
                for index, i in enumerate(bbox):
                    p = np.reshape(pt[index], (-1, 2))
                    box = i[:4]
                    ct = np.array([(i[2] + i[0]) / 2, (i[3] + i[1]) / 2])
                    p = ct + p
                    p = p * 4
                    box = box * 4
                    image = cv2.rectangle(image, (int(float(box[0])),
                                                  int(float(box[1]))),
                                          (int(float(box[2])),
                                           int(float(box[3]))), (255, 0, 0), 2)
                    for k in p:
                        cv2.circle(image, (int(k[0]), int(k[1])), 1, (255, 0, 0), 4)

                cv2.imshow('image', image)
                cv2.waitKey()
        except tf.errors.OutOfRangeError:
            break
    t2 = time.time()
    print(t2 - t1)
