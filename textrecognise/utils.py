#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import cv2
import natsort
import scipy.io as sio
# +-* + () + 10 digit + blank + space
#num_classes = 3 + 2 + 10 + 1 + 1
num_classes = 32 
maxPrintLen = 100

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 180, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('max_stepsize', 64, 'max stepsize in lstm, as well as '
                                                'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 1, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 5000000, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', './data/sign/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './data/sign/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './data/sign/train/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')


FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)


'''
#charset = '0123456789+-*()'
#charset = '0123456789ABCHILNOVYk主体侧关刀制华压变号合器场子容富开抗控智有本机构柜母汇电相石端箱线聂联能莲调贾载迤长闸+'

encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
'''
#decode_maps = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','']
#decode_maps = ['0', '1', '2', '']
decode_maps = ['1','0','k','V','2','号','主','变','有','载','调','压','控','制','箱','进','避','雷','器','段','母','I','实','训','II','C','线','B','A','相',' ','']
#words = sio.loadmat('/home/zdyd/software/text_recognition/data/text/words.mat')
#print words['new_words'][0][64]
#print charset[63]

class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        if data_dir == 'train':
            name = '/media/zdyd/Mango/textrecognition/swf0818/data/' + data_dir + '/'
            fname = os.listdir(name)
            for file in fname:
                if file[-3:] == 'jpg':
                    im = cv2.imread(name+file, 0).astype(np.float32) / 255.
                    im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
                    #print(im.shape)
                    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                    f = open(name+file[:-3]+'txt','r')
                    label_line = f.read()[:]
                    label_line = label_line.strip('\n')
                    f.close()

                    #label_line = np.loadtxt(name+file[:-3]+'txt',dtype=str)
                    print(label_line)
                    #print(content)
                    labels = []
                    #for i in range(label_line.size):
                        #print(i)
                    #for i in range(len(label_line)):
                    for j in str(label_line):
                        index = decode_maps.index(j)
                        labels.append(int(index))
                    #print(labels)
                    self.image.append(im)
                    self.labels.append(labels)
                        #print(type(label[0]),label[1])
                #print(name+file)
            '''
            fname = sio.loadmat(name + 'name.mat')
            #print('name:', fname)
            labels = sio.loadmat(name + 'label.mat')
            for i in range(len(fname['trainname'][0])):
                image_name = '/media/zdyd/code/yuanfei/ocr/textrecognition/data/text/image_train/' + str(fname['trainname'][0][i][0]) + '.jpg'
                #print(image_name)
                label = labels['trainlabel'][0][i]
                label = list(label[0,:])
                #print image_name, label

                #print image_name, txt.decode('utf8')
                im = cv2.imread(image_name, 0).astype(np.float32)/255.
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                #code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                self.labels.append(label)
                #print image_name, label
            '''
            '''
            elif data_dir == 'val':
                name = '/media/zdyd/code/yuanfei/ocr/textrecognition/data/text/' + data_dir + '/'
                fname = sio.loadmat(name + 'name.mat')
                labels = sio.loadmat(name + 'label.mat')
                for i in range(len(fname['valname'][0])):
                    image_name = '/media/zdyd/code/yuanfei/ocr/textrecognition/data/text/image_train/' + str(fname['valname'][0][i][0]) + '.jpg'
                    label = labels['vallabel'][0][i]
                    label = list(label[0,:])
                    #print image_name, label

                    #print image_name, txt.decode('utf8')
                    im = cv2.imread(image_name, 0).astype(np.float32)/255.
                    # resize to same height, different width will consume time on padding
                    # im = cv2.resize(im, (image_width, image_height))
                    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                    self.image.append(im)

                    # image is named as /.../<folder>/00000_abcd.png
                    #code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                    self.labels.append(label)
                    #print image_name, label
            '''
        elif data_dir == 'val':
            name = '/media/zdyd/Mango/textrecognition/swf0818/data/' + data_dir + '/'
            fname = os.listdir(name)
            for file in fname:
                if file[-3:] == 'jpg':
                    im = cv2.imread(name + file, 0).astype(np.float32) / 255.
                    im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
                    # print(im.shape)
                    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                    label_line = np.loadtxt(name + file[:-3] + 'txt', dtype=str)
                    print(label_line)
                    labels = []
                    # for i in range(label_line.size):
                    # print(i)
                    for j in str(label_line):
                        index = decode_maps.index(j)
                        labels.append(int(index))
                        # print(j)
                    print(labels)
                    self.image.append(im)
                    self.labels.append(labels)
    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in xrange(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
