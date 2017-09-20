import tensorflow as tf
import numpy as np
import datetime
from random import randint
import brad_model as model
import brad_w2v as w2v
import brad_input as data_input



# Some test strings that we'll use as input at intervals during training
encoderTestStrings = ["whats up bro",
					  "hi",
					  "hey how are you",
					  "that girl was really cute tho",
					  "that dodgers game was awesome"
					  ]

zeroVector = np.zeros((1), dtype='int32')


# ----------------- START ---------------

# global variables
numIterations = 500000
BATCH_SIZE = 24
maxEncoderLength = 15
maxDecoderLength = maxEncoderLength


# input
encoder_inputs, decoder_inputs, decoder_labels, feed_previous = model.io()				# using the feed dictionary
# seq2seq model
decoder_outputs, decoderPrediction = model.inference(encoder_inputs, datetime, feed_previous, w2v.vocabSize)
# loss
loss = model.loss(decoder_outputs, decoder_labels)
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

		encoderTrain, decoderTargetTrain, decoderInputTrain = data_input.getTrainingBatch(data_input.xTrain, data_input.yTrain, BATCH_SIZE, maxEncoderLength)
		feedDict = {encoder_inputs[t]: encoderTrain[t] for t in range(maxEncoderLength)}
		feedDict.update({decoder_labels[t]: decoderTargetTrain[t] for t in range(maxDecoderLength)})
		feedDict.update({decoder_inputs[t]: decoderInputTrain[t] for t in range(maxDecoderLength)})
		feedDict.update({feed_previous: False})

		curLoss, _, pred = sess.run([loss, train_step, decoderPrediction], feed_dict=feedDict)

		if i % 50 == 0:
			print('Current loss:', curLoss, 'at iteration', i)
			summary = sess.run(merged, feed_dict=feedDict)
			writer.add_summary(summary, i)

		if i % 10000 == 0 and i != 0:
			savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)


		if i % 25 == 0 and i != 0:
			num = randint(0, len(encoderTestStrings) - 1)
			print()
			encoderTestStrings[num]
			inputVector = data_input.getTestInput(encoderTestStrings[num], w2v.wordList, maxEncoderLength);
			feedDict = {encoder_inputs[t]: inputVector[t] for t in range(maxEncoderLength)}
			feedDict.update({decoder_labels[t]: zeroVector for t in range(maxDecoderLength)})
			feedDict.update({decoder_inputs[t]: zeroVector for t in range(maxDecoderLength)})
			feedDict.update({feed_previous: True})
			ids = (sess.run(decoderPrediction, feed_dict=feedDict))
			print()
			w2v.idsToSentence(ids, w2v.wordList)