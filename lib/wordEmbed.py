import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
from scipy import spatial
import re
import sys

class Embedding	:
	def __init__(self,embedDim):
		self.fileName = "/home/hansd410/data/glove/glove.6B."+str(embedDim)+"d.txt"
		self.wordIdxDic = {}
		self.wordEmbedding = None

		embeddingList = []
		wordIdx = 0
		fin = open(self.fileName,'r')
		while True:
			if(not((wordIdx+1) % 100000)):
			   print ("reading embedding "+str(wordIdx+1))
			line = fin.readline()

			# end of line, stop read
			if not line: break

			token = line.split()
			vector = token[1:]
			embeddingList.append(vector)
			self.wordIdxDic[token[0]] = wordIdx
			wordIdx += 1
		fin.close()

		embedNumpy = np.asarray(embeddingList).astype(float)

		# Out of vocabulary processing
		self.wordIdxDic["<unk>"]=wordIdx
		#embedNumpy = np.append(embedNumpy,[(np.random.rand(embedDim)-0.5)*2],axis=0)
		embedNumpy = np.append(embedNumpy,[(np.zeros(embedDim))],axis=0)
		embedShape = embedNumpy.shape
		
		embedTensor = torch.zeros(embedShape,dtype=torch.float)
		embedTensor =embedTensor.new_tensor(embedNumpy)

		self.wordEmbedding = torch.nn.Embedding.from_pretrained(embedTensor)
	
	def getEmbed(self):
		return self.wordEmbedding
	
	def getWordIdxDic(self):
		return self.wordIdxDic

	def getEmbedTensor(self,senList, maxLen):
		wholeIdxList = []
		for i in range(len(senList)):
			tokenList=senList[i].split(' ')
			senIdxList = []
			for token in tokenList:
				if token not in self.wordIdxDic.keys():
					token = '<unk>'

				if(len(senIdxList)==maxLen):
					break
				else:
					senIdxList.append(self.wordIdxDic[token])
			while(len(senIdxList)<maxLen):
				senIdxList.append(self.wordIdxDic['<unk>'])
			senIdxArray = np.array(senIdxList)
			wholeIdxList.append(senIdxArray)
		wholeIdxArray = np.array(wholeIdxList)
		wholeIdxTensor = torch.tensor(wholeIdxArray)
		return self.wordEmbedding(wholeIdxTensor)
