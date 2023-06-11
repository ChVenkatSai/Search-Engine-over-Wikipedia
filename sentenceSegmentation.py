import nltk.data

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		text = text.replace('!','.')
		text = text.replace('?','.')

		segmentedText = text.split(".")[:-1]

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

		segmentedText = sent_detector.tokenize(text.strip())

		return segmentedText
