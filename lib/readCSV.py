
import csv

class csvReader:
	def __init__(self, csvDir):
		self.contextList = []
		with open(csvDir) as csvFile:
			data = csv.reader(csvFile, delimiter = '\t',quotechar=None)
			for row in data:
				self.contextList.append(row[0])
	def getContextList(self):
		return self.contextList
