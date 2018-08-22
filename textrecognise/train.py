#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper
import natsort

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)

def main(_):
    model = cnn_lstm_otc_ocr.LSTMOCR('train')
    model.build_graph()

    print('loading train data, please wait---------------------')
    train_feeder = utils.DataIterator(data_dir='train')
    print('get image: ', type(train_feeder.image[0].shape),train_feeder.labels)

    print('loading validation data, please wait---------------------')
    val_feeder = utils.DataIterator(data_dir='val')
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  # 100000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 100000/100

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # example: 10000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.device('/gpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                shuffle_idx = np.random.permutation(num_train_samples)
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()

                # the tracing part
                for cur_batch in range(num_batches_per_epoch):
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    indexs = [shuffle_idx[i % num_train_samples] for i in
                              range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                    batch_inputs, batch_seq_len, batch_labels = \
                        train_feeder.input_index_generate_batch(indexs)
                    # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels,
                            model.seq_len: batch_seq_len}

                    # if summary is needed
                    # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)
                    # calculate the cost
                    train_cost += batch_cost * FLAGS.batch_size
                    print('batch_cost is: ', batch_cost)
                    #print 'train_cost is: ', train_cost
                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)

                    # train_err += the_err * FLAGS.batch_size
                    # do validation
                    if step % FLAGS.validation_steps == 0:
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        for j in xrange(num_batches_per_epoch_val):
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                          range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                            val_inputs, val_seq_len, val_labels = \
                                val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels,
                                        model.seq_len: val_seq_len}

                            dense_decoded, lastbatch_err, lr = \
                                sess.run([model.dense_decoded, model.lrn_rate],
                                         val_feed)

                            # print the decode result
                            ori_labels = val_feeder.the_label(indexs_val)
                            acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                             ignore_value=-1, isPrint=True)
                            acc_batch_total += acc

                        accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples

                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                        # train_err /= num_train_samples
                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                         lastbatch_err, time.time() - start_time, lr))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
