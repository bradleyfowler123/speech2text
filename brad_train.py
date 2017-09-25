import tensorflow as tf
import datetime
import brad_model as model
import brad_input as data_input



# ----------- GLOBAL VARIABLES ------------ #

# Shared Global Variables
BATCH_SIZE = 10
ENCODER_MAX_TIME = 20 			#arbitary but less than shortest input vector of sound
DECODER_MAX_TIME = 36

# Unique Global variables
NUM_INTERATIONS = 500000



# ----------- SEQ2SEQ MODEL -------------- #

# input
encoder_inputs_embedded, decoder_inputs_embedded, decoder_targets_indicies, embed_normed = model.io()				# using the feed dictionary
# seq2seq model
decoder_outputs, decoder_logits = model.inference(encoder_inputs_embedded, decoder_inputs_embedded)
# loss
loss, decoder_prediction = model.loss(decoder_targets_indicies, decoder_logits, embed_normed)
# training operation
train_step = model.optimise(loss)



# ----------- initialisations ------------ #
init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer(),tf.local_variables_initializer())  		# Create the graph, etc.
saver = tf.train.Saver()

with tf.Session() as sess:

	sess.run(init_op)

	# Tensorboard - merge all the summaries and write them out to directory
	merged = tf.summary.merge_all()
	LOG_DIR = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	writer = tf.summary.FileWriter(LOG_DIR, sess.graph)


# ----------- TRAINING LOOP -------------- #
	for i in range(NUM_INTERATIONS):

		encoderTrain, decoderTargetTrain, decoderInputTrain, label_inds = data_input.getTrainingBatch(data_input.xTrain, data_input.yTrain, BATCH_SIZE, data_input.label_indicies)
		# encoder train [batch_size*length_of_sequence*20] !! need to pad		this one is 153
		temp = [encoderTrain[t][0:ENCODER_MAX_TIME] for t in range(len(encoderTrain))]
		feedDict = {encoder_inputs_embedded: temp}
		feedDict.update({decoder_targets_indicies: label_inds})
		feedDict.update({decoder_inputs_embedded: decoderTargetTrain})


		try:
			curLoss, _ = sess.run([loss, train_step], feed_dict=feedDict)
		except ValueError:
			print('EEERRRROOORRRRR!')


		if i % 50 == 0:
			print('Current loss:', curLoss, 'at iteration', i)
			summary, pred = sess.run([merged, decoder_prediction], feed_dict=feedDict)
			writer.add_summary(summary, global_step=i)

			for j in range(BATCH_SIZE):
				print(data_input.w2v.idsToSentence(pred[j]))


		if i % 10000 == 0 and i != 0:
			print('Saving Checkpoint...')
			savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)