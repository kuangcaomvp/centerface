#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict

__C = edict()

cfg = __C
__C.BASE = edict()

# Set the class name
__C.BASE.CLASSES = "./data/classes/class.names"

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = ["2.txt"]#["mot_16-2.txt", 'mot_20-2.txt', 'human_3.txt']
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.INPUT_SIZE = [224, 224]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.INITIAL_WEIGHT = "./checkpoint_dcn2/loss=1.7240.ckpt-3"
__C.TRAIN.topK = 5
__C.TRAIN.epochs = 20
__C.TRAIN.warmup_periods = 5
# 学习率
__C.TRAIN.lr = 1e-3
__C.TRAIN.lr_deep = 1e-5
# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = ["test.txt"]
__C.TEST.BATCH_SIZE = 4
__C.TEST.INPUT_SIZE = [224, 224]
__C.TEST.INPUT_SIZES = [416, 608]
__C.TEST.DATA_AUG = False
__C.TEST.SCORE_THRESHOLD = 0.5
__C.TEST.IOU_THRESHOLD = 0.45

__C.PREDICT = edict()
__C.PREDICT.INPUT_SIZE = 416
__C.PREDICT.SCORE = 0.4
__C.PREDICT.IOU = 0.45
