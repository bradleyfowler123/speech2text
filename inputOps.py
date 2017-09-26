import numpy as np
from random import randint
import os
import wordEmbedding as w2v
import csv


# ----------- GLOBAL VARIABLES ------------ #

# Shared Global Variables
ENCODER_MAX_TIME = 20							# max length of input signal (in vectorised form)
ENCODER_INPUT_DEPTH = 20



def getTrainingBatch(batch_size):
	num = randint(0, numTrainingExamples - batch_size - 1)
	arr = X_TRAIN[num:num + batch_size]
	labels = Y_TRAIN[num:num + batch_size]
	label_inds = Y_TRAIN_IND[num:num + batch_size]

	# Reversing the order of encoder string apparently helps as per 2014 paper
	#reversedList = list(arr)
	#for index,example in enumerate(reversedList):
	#	reversedList[index] = list(reversed(example))


	# Lagged labels are for the training input into the decoder
	laggedLabels = [np.roll(example,-1) for example in labels]

	# Need to transpose these
	#reversedList = np.asarray(reversedList).T.tolist()
	#labels = np.asarray(labels).T.tolist()
	#laggedLabels = np.asarray(laggedLabels).T.tolist()
	return arr, labels, laggedLabels, label_inds


def getTestBatch(batch_size):
	# sound data
	arr = X_TRAIN[0:batch_size]

	# labels
	labels = Y_TRAIN[0:batch_size]
	label_inds = Y_TRAIN_IND[0:batch_size]
	laggedLabels = [np.roll(example,-1) for example in labels]


	return arr, labels, laggedLabels, label_inds

def createTrainingMatrices():
	# load meta file
	label, label_indicies, mfcc_file = [], [], []
	with open('asset/data/preprocess/meta/train.csv') as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for row in reader:
			# mfcc file
			mfcc2D = np.load('asset/data/preprocess/mfcc/' + row[0] + '.npy').T

			# pad end with zeros
			if len(mfcc2D) < ENCODER_MAX_TIME:
				temp = np.zeros((ENCODER_MAX_TIME-len(mfcc2D),ENCODER_INPUT_DEPTH))
				mfcc2D = np.append(mfcc2D, temp, axis=0)
			else:
				mfcc2D = mfcc2D[0:ENCODER_MAX_TIME, :]

			mfcc_file.append(mfcc2D)
			label_indicies.append([int(index) for index in row[1:]])
			label.append([w2v.wordVectors[int(index)] for index in row[1:]])


	return mfcc_file, label, label_indicies			# each is a list of 43663 items: labels item is 36 length list of vectors of size 50
									# input item is an array 20xn of floats. equvilent to length 20 list of size(sequence_length) vectors






if os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy'):
	X_TRAIN = np.load('Seq2SeqXTrain.npy')
	Y_TRAIN = np.load('Seq2SeqYTrain.npy')
	numTrainingExamples = X_TRAIN.shape[0]

	print('Finished loading training matrices')
else:
	# get input audio as vectors and store in xtrain
	# get labels for audio and store in ytrain
	X_TRAIN, Y_TRAIN, Y_TRAIN_IND = createTrainingMatrices()
	numTrainingExamples = len(Y_TRAIN_IND)
	#np.save('Seq2SeqXTrain.npy', X_TRAIN)
	#np.save('Seq2SeqYTrain.npy', Y_TRAIN)

	print('Finished creating training matrices')



