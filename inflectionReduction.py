from nltk.stem import PorterStemmer

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		reducedText = []
		for s in text:
			words = []
			for word in s:
				words.append(PorterStemmer().stem(word))
			reducedText.append(words)

		return reducedText
