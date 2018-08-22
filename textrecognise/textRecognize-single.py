# -*- coding:utf-8 -*-
import logging
import os
import cv2
import numpy as np
import tensorflow as tf
import cnn_lstm_otc_ocr
import utils
import natsort
import base64

checkpoint_dir = '/media/zdyd/dujing/yjx/textrecognition/checkpoint'
Dir = '/media/zdyd/dujing/yjx/textrecognition/digital/'

FLAGS = utils.FLAGS
logger = logging.getLogger('Testing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)

class textrecognize(object):
	def __init__(self):
		self.model = cnn_lstm_otc_ocr.LSTMOCR('infer')
		self.graph = self.model.build_graph()

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.6

		self.sess = tf.Session(config = config)
		self.saver = tf.train.Saver()
		self._load_weights('/media/zdyd/dujing/yjx/textrecognition/checkpoint', self.sess, self.saver)


	def _load_weights(self,checkpoint_dir, sess, saver):

		ckpt = tf.train.latest_checkpoint(checkpoint_dir)

		if ckpt:
			saver.restore(sess, ckpt)
			print('restore from ckpt{}'.format(ckpt))
		else:
			print('cannot restore')


	def main(self,filename):

		#imgList = sorted(os.listdir('/media/zdyd/code/yuanfei/ocr/textrecognition/digital/test/'))
		#imgList = natsort.natsorted(imgList)
		#total_steps = len(imgList) / FLAGS.batch_size

		os.environ["CUDA_VISIBLE_DEVICES"] = '0'
		decoded_expression = []

		#for curr_step in xrange(total_steps):
		imgs_input = []
		seq_len_input = []

		#for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
			#filename = Dir + 'test/'+ img
		im = cv2.imread(filename, 0).astype(np.float32) / 255.
		im = cv2.resize(im, (180, 60))
		im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

		def get_input_lens(seqs):
			length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)
			return seqs, length

		inp, seq_len = get_input_lens(np.array([im]))
		imgs_input.append(im)
		seq_len_input.append(seq_len)

		imgs_input = np.asarray(imgs_input)
		seq_len_input = np.asarray(seq_len_input)
		seq_len_input = np.reshape(seq_len_input, [-1])

		feed = {self.model.inputs: imgs_input,self.model.seq_len: seq_len_input}
		dense_decoded_code = self.sess.run(self.model.dense_decoded, feed)
		#print(type(dense_decoded_code))
		for item in dense_decoded_code:
			print(item)
			expression = ''
			for i in item:
				expression += utils.decode_maps[i]
			decoded_expression.append(expression)

		with open('./result.txt', 'a') as f:
			for code in decoded_expression:
				f.write(code + '\n')

		result = base64.b64encode(code)

		print("解码结果：")
		r = base64.b64decode(result)
		print(r)

		print("编码结果：")
		return result


if __name__ == '__main__':
	print('@@')
	textrecognize = textrecognize()
	textrecognize.main('/media/zdyd/dujing/yjx/zone1_subimg/sign1/sign1_0.jpg')

