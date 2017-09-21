import tensorflow as tf



# Hyperparamters
maxEncoderLength = 72
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3





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

	decoder_outputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs,
																						decoder_inputs, single_lstm_cell,
																						vocabSize, vocabSize,
																						embeddingDim,
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