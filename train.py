#! /usr/bin/env python
# coding=utf-8

import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

from core.config import cfg
from core.dataset import Dataset
from core.model import Face

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_available_gpus():
    r"""
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def calculate_mean_edit_distance_and_loss(iterator, trainable, nid, dropout):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    img, hm, reg_mask, ind, wh, reg, ids, bbox_xys, _, pts = iterator.get_next()
    model = Face(img, trainable, nid, dropout)
    loss, loss_hm, loss_wh, loss_offset, loss_ids, giou_loss, l1_loss, pts_loss = model.loss(hm, wh, reg_mask, ind, reg,
                                                                                             ids,
                                                                                             bbox_xys, pts)
    return loss, loss_hm, loss_wh, loss_offset, loss_ids, giou_loss, l1_loss, pts_loss


def get_tower_results(iterator, optimizer, trainable, nid, dropout):
    # To calculate the mean of the losses
    tower_avg_losses = []
    # Tower gradients to return
    tower_gradients = []
    losses_hm = []
    losses_wh = []
    losses_off = []
    losses_ids = []
    losses_giou = []
    losses_l1 = []
    losses_pts = []

    available_devices = get_available_gpus()
    if len(available_devices) == 0:
        available_devices = ['/cpu:0']
    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(available_devices)):
            # Execute operations of tower i on device i
            device = available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    loss, loss_hm, loss_wh, loss_offset, loss_ids, giou_loss, l1_loss, pts_loss = calculate_mean_edit_distance_and_loss(
                        iterator,
                        trainable,
                        nid, dropout)
                    tf.get_variable_scope().reuse_variables()
                    tower_avg_losses.append(loss)
                    losses_hm.append(loss_hm)
                    losses_wh.append(loss_wh)
                    losses_off.append(loss_offset)
                    losses_ids.append(loss_ids)
                    losses_giou.append(giou_loss)
                    losses_l1.append(l1_loss)
                    losses_pts.append(pts_loss)
                    gradients = optimizer.compute_gradients(loss)
                    tower_gradients.append(gradients)
    avg_loss = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    avg_loss_hm = tf.reduce_mean(input_tensor=losses_hm, axis=0)
    avg_loss_wh = tf.reduce_mean(input_tensor=losses_wh, axis=0)
    avg_loss_off = tf.reduce_mean(input_tensor=losses_off, axis=0)
    avg_loss_ids = tf.reduce_mean(input_tensor=losses_ids, axis=0)
    avg_loss_giou = tf.reduce_mean(input_tensor=losses_giou, axis=0)
    avg_loss_l1 = tf.reduce_mean(input_tensor=losses_l1, axis=0)
    avg_loss_pts = tf.reduce_mean(input_tensor=losses_pts, axis=0)
    return tower_gradients, avg_loss, avg_loss_hm, avg_loss_wh, avg_loss_off, avg_loss_ids, avg_loss_giou, avg_loss_l1, avg_loss_pts


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []
    # Run this on cpu_device to conserve GPU memory
    with tf.device('/cpu:0'):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)
            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])
            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def train():
    trainset = Dataset('train')
    testset = Dataset('dev')
    nid = trainset.nid
    global_step = tf.train.get_or_create_global_step()

    steps_per_period = len(trainset)
    train_steps = tf.constant(cfg.TRAIN.epochs * steps_per_period,
                              dtype=tf.float32, name='train_steps')
    warmup_steps = tf.constant(cfg.TRAIN.warmup_periods * steps_per_period,
                               dtype=tf.float32, name='warmup_steps')
    warmup_lr = tf.to_float(global_step) / tf.to_float(warmup_steps) \
                * cfg.TRAIN.lr
    decay_lr = tf.train.cosine_decay(
        cfg.TRAIN.lr,
        global_step=tf.to_float(global_step) - warmup_steps,
        decay_steps=train_steps - warmup_steps,
        alpha=0.01)
    learn_rate = tf.where(tf.to_float(global_step) < warmup_steps, warmup_lr, decay_lr)
    # learn_rate = noam_scheme(cfg.TRAIN.lr, global_step, warmup_steps)
    optimizer = tf.train.AdamOptimizer(learn_rate)

    iterator = tf.data.Iterator.from_structure(trainset.dataset.output_types,
                                               trainset.dataset.output_shapes,
                                               output_classes=trainset.dataset.output_classes)
    train_init_op = iterator.make_initializer(trainset.dataset)
    test_init_op = iterator.make_initializer(testset.dataset)

    trainable = tf.placeholder(dtype=tf.bool, name='training')
    dropout = tf.placeholder(tf.float32, shape=(), name='drop_out')
    gradients, loss, loss_hm, loss_wh, loss_off, loss_ids, loss_giou, loss_l1, loss_pts = get_tower_results(iterator,
                                                                                                            optimizer,
                                                                                                            trainable,
                                                                                                            nid,
                                                                                                            dropout)

    avg_tower_gradients = average_gradients(gradients)

    grads, all_vars = zip(*avg_tower_gradients)
    clipped, gnorm = tf.clip_by_global_norm(grads, 0.25)
    grads_and_vars = list(zip(clipped, all_vars))
    apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        loader = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % cfg.TRAIN.INITIAL_WEIGHT)
            loader.restore(sess, cfg.TRAIN.INITIAL_WEIGHT)
        except:
            print('=> %s does not exist !!!' % cfg.TRAIN.INITIAL_WEIGHT)
            print('=> Now it starts to train LM from scratch ...')

        lenth = len(get_available_gpus())
        if lenth == 0:
            lenth = 1
        for epoch in range(1, 1 + cfg.TRAIN.epochs):
            # 初始化数据集
            sess.run(train_init_op)
            train_epoch_loss, test_epoch_loss = [], []
            train_epoch_hm = []
            train_epoch_wh = []
            train_epoch_off = []
            train_epoch_ids = []
            train_epoch_giou = []
            train_epoch_l1 = []
            train_epoch_pts = []
            pbar = tqdm(range(len(trainset) // lenth + 1))
            for i in pbar:
                try:
                    _, train_step_loss, train_step_hm, train_step_wh, train_step_off, train_step_ids, train_step_giou, train_step_l1, train_step_pts, \
                    global_step_val = sess.run(
                        [apply_gradient_op, loss, loss_hm, loss_wh, loss_off, loss_ids, loss_giou, loss_l1, loss_pts,
                         global_step], feed_dict={
                            trainable: True, dropout: 0.1

                        })
                    train_epoch_loss.append(train_step_loss)
                    train_epoch_hm.append(train_step_hm)
                    train_epoch_wh.append(train_step_wh)
                    train_epoch_off.append(train_step_off)
                    train_epoch_ids.append(train_step_ids)
                    train_epoch_giou.append(train_step_giou)
                    train_epoch_l1.append(train_step_l1)
                    train_epoch_pts.append(train_step_pts)
                    pbar.set_description("loss:%.2f hm:%.2f wh:%.2f off:%.2f ids:%.2f giou: %.2f l1: %.2f pts: %.2f"
                                         % (
                                             train_step_loss, train_step_hm, train_step_wh, train_step_off,
                                             train_step_ids, train_step_giou, train_step_l1, train_step_pts))
                except tf.errors.OutOfRangeError:
                    break


            sess.run(test_init_op)
            while True:
                try:
                    test_step_loss = sess.run([loss], feed_dict={
                        trainable: False, dropout: 0.0
                    })
                    test_epoch_loss.append(test_step_loss)
                except tf.errors.OutOfRangeError:
                    break
                except:
                    break
            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            train_epoch_hm = np.mean(train_epoch_hm)
            train_epoch_wh = np.mean(train_epoch_wh)
            train_epoch_off = np.mean(train_epoch_off)
            train_epoch_ids = np.mean(train_epoch_ids)
            train_epoch_giou = np.mean(train_epoch_giou)
            train_epoch_l1 = np.mean(train_epoch_l1)
            train_epoch_pts = np.mean(train_epoch_pts)
            ckpt_file = "./checkpoint_dcn2/loss=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s hm: %.2f wh: %.2f off: %.2f ids: %.2f giou: %.2f l1: %.2f pts: %.2f"
                  % (
                      epoch, log_time, train_epoch_hm, train_epoch_wh, train_epoch_off, train_epoch_ids,
                      train_epoch_giou,
                      train_epoch_l1, train_epoch_pts))
            print("=> Epoch: %2d Time: %s Train_loss: %.2f Test_loss: %.2f"
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss))
            saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    train()
