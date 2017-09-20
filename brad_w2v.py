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





def idsToSentence(ids, wList):
	EOStokenIndex = wList.index('<EOS>')
	padTokenIndex = wList.index('<pad>')
	myStr = ""
	listOfResponses=[]
	for num in ids:
		if num[0] == EOStokenIndex or num[0] == padTokenIndex:
			listOfResponses.append(myStr)
			myStr = ""
		else:
			myStr = myStr + wList[num[0]] + " "
	if myStr:
		listOfResponses.append(myStr)
	listOfResponses = [i for i in listOfResponses if i]
	return listOfResponses