import numpy as np


# Loading in all the data structures
wordList = np.load("trained_w2v_embedding/wordsList.npy")
wordList = wordList.tolist()

wordVectors = np.load('trained_w2v_embedding/wordVectors.npy')
vocabSize = len(wordList)
wordVecDimensions = wordVectors.shape[1]

# Add two entries to the word vector matrix. One to represent padding tokens,
# and one to represent an end of sentence token
padVector = np.zeros((1, wordVecDimensions), dtype='int32')
EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
wordVectors = np.concatenate((wordVectors,padVector), axis=0)
wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

# Need to modify the word list as well
wordList.append('<pad>')
wordList.append('<EOS>')
vocabSize = vocabSize + 2





def idsToSentence(ids):
	scentence = ""
	for word_id in ids:
		if wordList[word_id] == '<EOS>':
			scentence = scentence + " " + "err"
		elif wordList[word_id] != '<pad>':
			scentence = scentence + " " + wordList[word_id].decode("utf-8")
	return scentence