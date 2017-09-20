import numpy as np
from random import randint
import os
import brad_w2v as w2v

maxEncoderLength = 15

def createTrainingMatrices(conversationFileName, wList, maxLen):
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary)
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.iteritems()):
		# Will store integerized representation of strings here (initialized as padding)
		encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		# Getting all the individual words in the strings
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Throw out sequences that are too long
		if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1)):
			continue
		# Integerize the encoder string
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				# TODO: This isnt really the right way to handle this scenario
				encoderMessage[keyIndex] = 0
		encoderMessage[keyIndex + 1] = wList.index('<EOS>')
		# Integerize the decoder string
		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wList.index(word)
			except ValueError:
				decoderMessage[valueIndex] = 0
		decoderMessage[valueIndex + 1] = wList.index('<EOS>')
		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage
	# Remove rows with all zeros
	yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
	xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
	numExamples = xTrain.shape[0]
	return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen):
	num = randint(0,numTrainingExamples - localBatchSize - 1)
	arr = localXTrain[num:num + localBatchSize]
	labels = localYTrain[num:num + localBatchSize]
	# Reversing the order of encoder string apparently helps as per 2014 paper
	reversedList = list(arr)
	for index,example in enumerate(reversedList):
		reversedList[index] = list(reversed(example))

	# Lagged labels are for the training input into the decoder
	laggedLabels = []
	EOStokenIndex = w2v.wordList.index('<EOS>')
	padTokenIndex = w2v.wordList.index('<pad>')
	for example in labels:
		eosFound = np.argwhere(example==EOStokenIndex)[0]
		shiftedExample = np.roll(example,1)
		shiftedExample[0] = EOStokenIndex
		# The EOS token was already at the end, so no need for pad
		if (eosFound != (maxLen - 1)):
			shiftedExample[eosFound+1] = padTokenIndex
		laggedLabels.append(shiftedExample)

	# Need to transpose these
	reversedList = np.asarray(reversedList).T.tolist()
	labels = labels.T.tolist()
	laggedLabels = np.asarray(laggedLabels).T.tolist()
	return reversedList, labels, laggedLabels

def translateToSentences(inputs, wList, encoder=False):
	EOStokenIndex = wList.index('<EOS>')
	padTokenIndex = wList.index('<pad>')
	numStrings = len(inputs[0])
	numLengthOfStrings = len(inputs)
	listOfStrings = [''] * numStrings
	for mySet in inputs:
		for index,num in enumerate(mySet):
			if (num != EOStokenIndex and num != padTokenIndex):
				if (encoder):
					# Encodings are in reverse!
					listOfStrings[index] = wList[num] + " " + listOfStrings[index]
				else:
					listOfStrings[index] = listOfStrings[index] + " " + wList[num]
	listOfStrings = [string.strip() for string in listOfStrings]
	return listOfStrings

def getTestInput(inputMessage, wList, maxLen):
	encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		try:
			encoderMessage[index] = wList.index(word)
		except ValueError:
			continue
	encoderMessage[index + 1] = wList.index('<EOS>')
	encoderMessage = encoderMessage[::-1]
	encoderMessageList=[]
	for num in encoderMessage:
		encoderMessageList.append([num])
	return encoderMessageList



if os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy'):
	xTrain = np.load('Seq2SeqXTrain.npy')
	yTrain = np.load('Seq2SeqYTrain.npy')
	print('Finished loading training matrices')
	numTrainingExamples = xTrain.shape[0]
else:
	numTrainingExamples, xTrain, yTrain = createTrainingMatrices('trained_w2v_embedding/idsMatrix.npy', w2v.wordList, maxEncoderLength)
	np.save('Seq2SeqXTrain.npy', xTrain)
	np.save('Seq2SeqYTrain.npy', yTrain)
	print('Finished creating training matrices')