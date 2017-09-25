import tensorflow as tf
import brad_w2v as w2v



# ----------- GLOBAL VARIABLES ------------ #

# Shared Global Variables
BATCH_SIZE = 10
ENCODER_MAX_TIME = 20							# max length of input signal (in vectorised form)
DECODER_MAX_TIME = 36							# max scentence length (in wordvectors)

# Unique Global Variables
ENCODER_INPUT_DEPTH = 20						# size of a sound feature vector
DECODER_INPUT_DEPTH = 50  						# len(a_word_vector)
OUTPUT_VOCAB_SIZE = w2v.vocabSize 				# how many words in your vocab
encoder_hidden_units = 100						# arbitary choice atm
decoder_hidden_units = 100



# ----------- MODEL FUNCTIONS ------------- #

tf.reset_default_graph()

def io():		# The input to our model (encoder) is the pre-embedding sound vector. The input into the decoder is the time shifted word embeddings of the text
	encoder_inputs_embedded = tf.placeholder(shape=(BATCH_SIZE, ENCODER_MAX_TIME, ENCODER_INPUT_DEPTH), dtype=tf.float32, name='encoder_inputs')  # [batch_size*length_of_sequence*20]
	decoder_targets_indicies = tf.placeholder(shape=(BATCH_SIZE, DECODER_MAX_TIME), dtype=tf.int64, name='decoder_targets')	# [batch_size, max_time36]
	decoder_inputs_embedded = tf.placeholder(shape=(BATCH_SIZE, DECODER_MAX_TIME, DECODER_INPUT_DEPTH), dtype=tf.float32, name='decoder_inputs')
	embed_normed = tf.constant(w2v.wordVectorsNormalised, dtype=tf.float32)

	return encoder_inputs_embedded, decoder_inputs_embedded, decoder_targets_indicies, embed_normed


def inference(encoder_inputs_embedded, decoder_inputs_embedded):

	#encoder_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([single_lstm_cell] * numLayersLSTM)

	# encoder
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
	_, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=False)  # replace later with bidirectional dynamic rnn

	# decoder
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32, time_major=False, scope="plain_decoder")  # replace later with bidirectional dynamic rnn

	# output
	decoder_logits = tf.contrib.layers.linear(decoder_outputs, DECODER_INPUT_DEPTH)										# (BATCH_SIZE, DECODER_MAX_TIME, OUTPUT_VOCAB_SIZE)


	return decoder_outputs, decoder_logits


def loss(decoder_targets, decoder_logits, embed_normed):

	def cs(logits):
		shifted_cosine = tf.matmul(logits, tf.transpose(embed_normed)) + 1	# 36*50 x 50*400002 = 36*400002
		return -1* shifted_cosine**4


	with tf.name_scope('performance_metrics'):
		# loss
		decoder_logits_normalised = tf.nn.l2_normalize(decoder_logits, dim=2)
		cosine_similarity = tf.map_fn(cs, decoder_logits_normalised)
		#cosine_similarity = tf.losses.cosine_distance(embed_normed, normed_embedding, dim=2)

		decoder_prediction = tf.argmax(cosine_similarity, 2)  # shape [batch_size, DECODER_MAX_TIME], type int64   # (BATCH_SIZE, DECODER_MAX_TIME)		- pick maximum liklihood words for each position (the index)


		# ------------ Tensorboard --------------- #
		total_loss = tf.reduce_mean(cosine_similarity)

		tf.summary.scalar('summaries/total_loss', total_loss)


		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(decoder_prediction, decoder_targets)
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)  # log model output

	return total_loss, decoder_prediction


def optimise(total_loss):
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

	return train_step