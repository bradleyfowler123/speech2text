import tensorflow as tf
import helpers
import numpy as np

def temp():
	# variables
	input_vocab_size = 20
	input_embedding_size = 65

	output_vocab_size = 10 # how many words in your vocab
	output_embedding_size = 65#50  # = len(a_word_vector)

	encoder_hidden_units = 65				# mmm
	decoder_hidden_units = 65#50


	# placeholders
	encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')    # encoder_inputs int32 tensor is shaped [batch_size, encoder_max_time]
	decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
	decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
	print(encoder_inputs)

	# embedding
	embeddings_encoder = tf.Variable(tf.random_uniform([input_vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
	encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_encoder, encoder_inputs)
	print(encoder_inputs_embedded)								# [batch_size, max_time, depth] [batch_size*20*length_of_sequence]
	embeddings_decoder = tf.Variable(tf.random_uniform([output_vocab_size, output_embedding_size], -1.0, 1.0), dtype=tf.float32)
	decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_decoder, decoder_inputs)
	print(decoder_inputs_embedded)								# [batch_size, max_time, depth]

	# encoder
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
	_, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=False)			# replace later with bidirectional dynamic rnn
	print(encoder_final_state)
									# encoder_final_state is also called "thought vector". We will use it as initial state for the Decoder. In seq2seq without attention this is the only point where Encoder passes information to Decoder


	# decoder
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32, time_major=False, scope="plain_decoder")			# replace later with bidirectional dynamic rnn
	print(decoder_outputs)

	# output
	decoder_logits = tf.contrib.layers.linear(decoder_outputs, output_vocab_size)
	decoder_prediction = tf.argmax(decoder_logits, 2)
	print(decoder_prediction)

	#optimiser
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		labels=tf.one_hot(decoder_targets, depth=output_vocab_size, dtype=tf.float32),
		logits=decoder_logits,
	)

	# loss
	loss = tf.reduce_mean(stepwise_cross_entropy)
	train_op = tf.train.AdamOptimizer().minimize(loss)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch_ = [[6], [3, 4], [9, 8, 7]]

		batch_, batch_length_ = helpers.batch(batch_)
		print('batch_encoded:\n' + str(batch_))

		din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
									max_sequence_length=4)
		print('decoder inputs:\n' + str(din_))

		pred_ = sess.run(decoder_prediction,
						 feed_dict={
							 encoder_inputs: batch_,
							 decoder_inputs: din_,
						 })
		print('decoder predictions:\n' + str(pred_))







if __name__ == '__main__':
	temp()


# Hyperparamters
maxEncoderLength = 36# somthing bigger
maxDecoderLength = 36
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3



tf.reset_default_graph()


def io():
	# Create the placeholders
	encoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name='enc_inps') for _ in range(maxEncoderLength)]
	decoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name='dec_inps') for _ in range(maxDecoderLength)]
	decoder_labels = [tf.placeholder(tf.int32, shape=(None,), name='dec_lbs') for _ in range(maxDecoderLength)]
	feed_previous = tf.placeholder(tf.bool)

	return encoder_inputs, decoder_inputs, decoder_labels, feed_previous


def inference(encoder_inputs, decoder_inputs, feed_previous, vocabSize):
	single_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
	encoder_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([single_lstm_cell] * numLayersLSTM)

	decoder_outputs = tf.contrib.legacy_seq2seq.rnn_seq2seq(encoder_inputs,	decoder_inputs, single_lstm_cell,
																						vocabSize, vocabSize,
																						feed_previous=feed_previous)
	decoderPrediction = tf.argmax(decoder_outputs, 2, name='dec_prd')

	return decoder_outputs, decoderPrediction



def loss(decoder_outputs, decoder_labels, vocabSize):
	# take output vector and map to words using pre-trained word2vec
	with tf.name_scope('performance_metrics'):
		loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in decoder_labels]

		prediction_loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs, decoder_labels, loss_weights, vocabSize)

		total_loss = prediction_loss
		tf.summary.scalar('summaries/total_loss', total_loss)

	return total_loss


def optimise(total_loss):
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

	return train_step