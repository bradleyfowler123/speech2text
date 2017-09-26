import numpy as np
import string


# --------- Functions to setup and handle text embeddings of output ---------#


# GLOBAL Variables
DECODER_MAX_TIME = 36																											# max scentence length is 36
_data_path = 'asset/data/' 																								# default data path




# Loading in all the data structures
wordList = np.load("trained_w2v_embedding/wordsList.npy").tolist() 														# note there is no space! you don't need one since each vector is a standalone word

wordVectors = np.load('trained_w2v_embedding/wordVectors.npy')
vocabSize = len(wordList)
wordVecDimensions = wordVectors.shape[1]

# Add two entries to the word vector matrix. One to represent padding tokens,
# and one to represent an end of sentence token
padVector = np.zeros((1, wordVecDimensions), dtype='int32')
EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
wordVectors = np.concatenate((wordVectors,padVector), axis=0)
wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

wordVectorsNormalised = np.zeros(wordVectors.shape)
norms = np.linalg.norm(wordVectors,axis=1)
for i in range(len(wordVectors)):
	if norms[i] != 0:
		wordVectorsNormalised[i] = wordVectors[i]/norms[i]


# Need to modify the word list as well
wordList.append('<pad>')
wordList.append('<EOS>')																								# note this means end of scentence however using here for symbol error when word outside of vocab
vocabSize = vocabSize + 2




# byte to index mapping
byte2index = {}
for i, ch in enumerate(wordList):
	byte2index[ch] = i


def idsToSentence(ids):
	scentence = ""
	for word_id in ids:
		if wordList[word_id] == '<EOS>':
			scentence = scentence + " " + "err"
		elif wordList[word_id] != '<pad>':
			scentence = scentence + " " + wordList[word_id].decode("utf-8")
	return scentence


# convert sentence to index list
def word2index(str_):			# !!! now word2index !!!

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
	for _ in range(len(res), DECODER_MAX_TIME):
		res.append(byte2index['<pad>'])
	return res