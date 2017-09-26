import numpy as np
import string


# ----- CURENTLY - functions to handle text embedding conversions
# TODO - merge with brad_w2v.py or brad_input.py or both

# GLOBAL Variables
MAX_LENGTH = 36																											# max scentence length is 36
_data_path = 'asset/data/' 																								# default data path


index2byte = np.load('trained_w2v_embedding/wordsList.npy').tolist()													# note there is no space! you don't need one since each vector is a standalone word
index2byte.append('<pad>')
index2byte.append('<EOS>')																								# note this means end of scentence however using here for symbol error when word outside of vocab


# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
	byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)


# convert sentence to index list
def str2index(str_):			# !!! now word2index !!!

	# clean white space
	str_ = ' '.join(str_.split())
	# remove punctuation and make lower case
	str_ = str_.translate(string.punctuation).lower()

	res = []
	for word in str_.split(' '):
		try:
			res.append(byte2index[bytes(word, encoding='utf-8')])		# word2index
		except KeyError:
			res.append(byte2index['<EOS>'])
			pass
	for _ in range(len(res), MAX_LENGTH):
		res.append(byte2index['<pad>'])
	return res


# convert index list to string
def index2str(index_list):
	# transform label index to character
	str_ = ''
	for index in index_list:
		str_ += index2byte[index]
	return str_


# print list of index list
def print_index(indices):
	for index_list in indices:
		print(index2str(index_list))




"""		NOT IN USE
import sugartensor as tf
import csv


# real-time wave to mfcc conversion function
@tf.sg_producer_func
def _load_mfcc(src_list):

	# label, wave_file
	label, mfcc_file = src_list

	# decode string to integer
	label = np.fromstring(label, np.int)

	# load mfcc
	mfcc = np.load(mfcc_file, allow_pickle=False)

	# speed perturbation augmenting
	mfcc = _augment_speech(mfcc)

	return label, mfcc


def _augment_speech(mfcc):

	# random frequency shift ( == speed perturbation effect on MFCC )
	r = np.random.randint(-2, 2)

	# shifting mfcc
	mfcc = np.roll(mfcc, r, axis=0)

	# zero padding
	if r > 0:
		mfcc[:r, :] = 0
	elif r < 0:
		mfcc[r:, :] = 0

	return mfcc


# Speech Corpus
class SpeechCorpus(object):

	def __init__(self, batch_size=16, set_name='train'):

		# load meta file
		label, mfcc_file = [], []
		with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:
				# mfcc file
				mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
				# label info ( convert to string object for variable-length support )
				label.append(np.asarray(row[1:], dtype=np.int).tostring())

		# to constant tensor
		label_t = tf.convert_to_tensor(label)
		mfcc_file_t = tf.convert_to_tensor(mfcc_file)

		# create queue from constant tensor
		label_q, mfcc_file_q \
			= tf.train.slice_input_producer([label_t, mfcc_file_t], shuffle=True)

		# create label, mfcc queue
		label_q, mfcc_q = _load_mfcc(source=[label_q, mfcc_file_q],
									 dtypes=[tf.sg_intx, tf.sg_floatx],
									 capacity=256, num_threads=64)

		# create batch queue with dynamic pad
		batch_queue = tf.train.batch([label_q, mfcc_q], batch_size,
									 shapes=[(None,), (20, None)],
									 num_threads=64, capacity=batch_size*32,
									 dynamic_pad=True)

		# split data
		self.label, self.mfcc = batch_queue
		# batch * time * dim
		self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])
		# calc total batch count
		self.num_batch = len(label) // batch_size

		# print info
		tf.sg_info('%s set loaded.(total data=%d, total batch=%d)'
				   % (set_name.upper(), len(label), self.num_batch))


"""