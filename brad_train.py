import tensorflow as tf
import numpy as np
import datetime
from random import randint
import brad_model as model
import brad_w2v as w2v
import brad_input as data_input


"""
# Some test strings that we'll use as input at intervals during training
encoderTestStrings = ["whats up bro",
					  "hi",
					  "hey how are you",
					  "that girl was really cute tho",
					  "that dodgers game was awesome"
					  ]

zeroVector = np.zeros((1), dtype='int32')
"""

# ----------------- START ---------------

# global variables
numIterations = 500000
BATCH_SIZE = 10
maxEncoderLength = 25 #arb
maxDecoderLength = 36


# input
encoder_inputs_embedded, decoder_inputs_embedded, decoder_targets_indicies = model.io()				# using the feed dictionary
# seq2seq model
decoder_outputs, decoder_logits, decoder_prediction = model.inference(encoder_inputs_embedded, decoder_inputs_embedded)
# loss
loss = model.loss(decoder_targets_indicies, decoder_logits)
# training operation
train_step = model.optimise(loss)




saver = tf.train.Saver()


# initialisations
init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer(),
				   tf.local_variables_initializer())  # Create the graph, etc.

with tf.Session() as sess:

	sess.run(init_op)

	# Tensorboard - merge all the summaries and write them out to directory
	merged = tf.summary.merge_all()
	LOG_DIR = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(LOG_DIR, sess.graph)


	for i in range(numIterations):

		encoderTrain, decoderTargetTrain, decoderInputTrain, label_inds = data_input.getTrainingBatch(data_input.xTrain, data_input.yTrain, BATCH_SIZE, data_input.label_indicies)
		# encoder train [batch_size*length_of_sequence*20] !! need to pad		this one is 153
		temp = [encoderTrain[t][0:maxEncoderLength] for t in range(len(encoderTrain))]
		feedDict = {encoder_inputs_embedded: temp}
		feedDict.update({decoder_targets_indicies: label_inds})
		feedDict.update({decoder_inputs_embedded: decoderTargetTrain})

		try:
			curLoss, _, pred = sess.run([loss, train_step, decoder_prediction], feed_dict=feedDict)
		except ValueError:
			print('EEERRRROOORRRRR!')




		if i % 50 == 0:
			print('Current loss:', curLoss, 'at iteration', i)
			summary = sess.run(merged, feed_dict=feedDict)
			writer.add_summary(summary, i)

		if i % 10000 == 0 and i != 0:
			savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)