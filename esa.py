from util import Utilities

import wikipedia
import json
import numpy as np

# Parser warning might occur :
# The code that caused this warning is on line 389 of the file /anaconda3/lib/python3.8/site-packages/wikipedia/wikipedia.py.
# To get rid of this warning, pass the additional argument 'features="lxml"' to the BeautifulSoup constructor.

class ESA():

	index = None # class variable - available to all instances

	def __init__(self):
		self.utilities = Utilities()

	# def GetConceptsFromWikipedia(self, doc_titles):
	# """
	# - Run this function first by passing the doc titles to get the concepts from wikipedia and store them in json
	# - This takes about 3 hours, this is already done
	# - So this part is commented
	#
	# Parameters
	# ----------
	# arg1 : list
	# 	A list of doc-titles from the cranfield dataset
	# Returns
	# -------
	# None
	# """
	# 	pages = []
	# 	for doc_title in doc_titles:
	# 		if(doc_title == ""):
	# 			continue
	# 		search_results = wikipedia.search(doc_title)
	# 		for item in search_results:
	# 			try:
	# 				# page = wikipedia.summary(item, auto_suggest = True) # only for summary
	# 				page = wikipedia.page(item, auto_suggest = True) # for whole content
	# 			except Exception:
	# 				continue
	#
	# 			# # only for summary
	# 			# pages.append({
    #             #     "title" : item,
    #             #     "summary": page
    #             # })
	# 			# for whole content
	# 			pages.append({
	# 		        "title" : page.title,
	# 		        "content": page.content
	# 			})
	# 	pages = list({page["title"]:page for page in pages}.values())
	# 	# with open("wikipedia_concepts_only_summary.json", 'w') as fout: # only for summary
	# 	with open("wikipedia_concepts_whole_content.json", 'w') as fout: # for the whole content
	# 		json.dump(pages, fout)

	def buildConceptsIndex(self, concepts):
		"""
		Builds the concept index and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			the content of a concept(wikipedia article) and each sub-sub-list is a sentence
			of the content of that concept
		Returns
		-------
		None
		"""

		index = None
		#Fill in code here

		concept_inv_index = self.utilities.concepts_inverted_index(concepts)
		corpus = self.utilities.corpus

		# idf
		idf = {}
		for word in corpus:
			if word not in concept_inv_index:
				idf[word] = 0
			else:
				idf[word] = np.log10(len(concepts)/(len(concept_inv_index[word])))

		# tfidf
		tfidf = np.zeros([len(concepts),len(corpus)])
		concept_index = 0
		for concept in concepts:
			for sentence in concept:
				words = list(set.intersection(set(corpus), set(sentence)))
				for word in words:
					word_index = corpus.index(word)
					tfidf[concept_index][word_index] += idf[word] # multiple adds -> covers tf
			concept_index += 1

		index = {
			"concept_inv_index" : concept_inv_index,
			"concept_corpus" : corpus,
			"concepts_tfidf" : tfidf
		}

		ESA.index = index

	def map_docs_to_concept_space(self, docs_tfidf):
		"""
		Parameters
		----------
		arg1 : matrix
			The tfidf matrix of all the docs with all the terms in the corpus
		Returns
		-------
			The docs mapped to the concept space
		"""
		docs_concepts = np.matmul(docs_tfidf, np.transpose(ESA.index["concepts_tfidf"]))
		return docs_concepts

	def map_query_to_concept_space(self, query_tfidf):
		"""
		Parameters
		----------
		arg1 : A vector
			The tfidf vector of a query with all the terms in the corpus
		Returns
		-------
			The query mapped to the concept space
		"""
		query_concepts = np.matmul(query_tfidf, np.transpose(ESA.index["concepts_tfidf"]))
		return query_concepts
