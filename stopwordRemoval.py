from util import *

from nltk.corpus import stopwords
from nltk.corpus import wordnet



class StopwordRemoval():

	def fromList(self, text, doQueryExpansion=False):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		arg2 : boolean
			It says whether to perform queryexpansion or not
		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed

			- Query expansion is also done if the boolean argument (arg2) is true
		"""

		stopwordRemovedText = None

		#Fill in code here
		stopwordRemovedText = []
		stop_words = set(stopwords.words('english'))
		for sentence in text:
			filtered_sentence = [word.lower() for word in sentence if not word in stop_words] # changing the case
			stopwordRemovedText.append(filtered_sentence)

		if(doQueryExpansion):
			stopwordRemovedQueryExpandedText = []
			count=0
			for x in stopwordRemovedText:
				synonyms = []
				for word in x:
					synonyms.append(word)
					for syn in wordnet.synsets(word):
						for l in syn.lemmas() :
							if(count<2):
								if l.name() not in synonyms:
									synonyms.append(l.name())
									count+=1
				stopwordRemovedQueryExpandedText.append(synonyms)
				count=0
			return stopwordRemovedQueryExpandedText

		return stopwordRemovedText
