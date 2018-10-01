# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     balance_data_train
   Description :
   Author :       'li'
   date：          2018/9/7
-------------------------------------------------
   Change Activity:
                   2018/9/7:
-------------------------------------------------
"""
from data_provider.training_data_provider import get_training_data

__author__ = 'li'

"""
Train shadow net script
"""
import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config

logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--weights_path', type=str, help='Where you store the pretrained weights')

    return parser.parse_args()


def train_shadownet(dataset_dir, weights_path=None):
    """
    :param dataset_dir:
    :param weights_path:
    :return:
    """
    # decode the tf records to get the training data

    # initializa the net model
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
    inputdata = tf.placeholder(dtype=tf.float32, shape=(32, 32, 100, 3))
    input_labels = tf.sparse_placeholder(tf.int32, shape=(None, -1))
    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=25 * np.ones(32)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32), merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)
    # Set tf summary
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()
    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)
    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)
    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)
    # Set the training parameters
    train_epochs = config.cfg.TRAIN.EPOCHS
    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(train_epochs):
            try:
                training_img, training_label = get_training_data()
                feed_dict = {inputdata: training_img, input_labels: training_label}
                _, c, seq_distance, preds, gt_labels, summary = sess.run(
                    [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op], feed_dict=feed_dict)
                # calculate the precision
                preds_sequence = preds[0].values.tolist()
                gt_value = gt_labels.values.tolist()
                pre_count = len(preds_sequence)
                accu_num = 0
                gt_count = len(gt_value)
                for index in range(gt_count):
                    if index < pre_count:
                        if gt_value[index] is not None and preds_sequence[index] is not None:
                            if gt_value[index] == preds_sequence[index]:
                                accu_num += 1
                accuracy = accu_num * 1.0 / pre_count

                if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy))
                if epoch % 1000 == 0 and epoch != 0:
                    summary_writer.add_summary(summary=summary, global_step=epoch)
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                    logger.info('save_model!!!!!!!!!!!!!!!!!!!___________________')
            except Exception as e:
                print(e)
        coord.request_stop()
        coord.join(threads=threads)
    sess.close()

    return


if __name__ == '__main__':
    # # init args
    # args = init_args()
    # if not ops.exists(args.dataset_dir):
    #     raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))
    # /gpu_data/code/text_recognition/model/shadownet/shadownet_2018-08-03-20-50-16.ckpt-6000
    # train_shadownet('./save_dir',
    #                 None)
    train_shadownet('./save_dir',
                    '/gpu_data/back/code/vertical_text_recognition/model/shadownet/shadownet_2018-10-01-11-20-26.ckpt-29000')
    print('Done')
