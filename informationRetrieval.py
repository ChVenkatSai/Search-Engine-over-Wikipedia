from util import Utilities

import numpy as np
from lsa import LSA
from esa import ESA

class InformationRetrieval():

	def __init__(self):
		self.index = None

		self.lsa = LSA()
		self.esa = ESA()
		self.utilities = Utilities()

	def buildIndex(self, docs, docIDs, isLSA):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		arg3: boolean
			Says whether LSA is being performed or not
		Returns
		-------
		None
		"""

		index = None

		inv_index, corpus = self.utilities.inverted_index(docs)

		# idf
		idf = {}
		for word in corpus:
			idf[word] = np.log10(len(docs)/(len(inv_index[word])))

		# tfidf
		tfidf = np.zeros([len(docs),len(corpus)])
		doc_index = 0
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					if word in corpus:
						word_index = corpus.index(word)
						tfidf[doc_index][word_index] += idf[word] # multiple adds -> covers tf
			doc_index += 1

		if(isLSA == True):
			tfidf_k, u_k, s_values_k, vt_k = self.lsa.reduced_tfidf(tfidf)
			index = {
				"corpus" : corpus,
				"idf" : idf,
				"tfidf" : tfidf_k,
				"T": u_k, # txs
				"S": s_values_k, # sxs
				"D": np.transpose(vt_k) # dxs
			}
		else:
			index = {
				"corpus" : corpus,
				"idf" : idf,
				"tfidf" : tfidf
			}

		self.index = index

	def rank(self, queries, isLSA, isESA):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		arg2: boolean
			Says whether LSA is being performed or not

		arg3: boolean
			Says whether ESA is being performed or not

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		# class variables
		corpus = self.index["corpus"]
		idf = self.index["idf"]
		tfidf_docs = self.index["tfidf"]

		if(isESA == True):
			docs_concepts = self.esa.map_docs_to_concept_space(tfidf_docs)
			tfidf_docs = docs_concepts

		tfidf = np.zeros([len(queries), len(corpus)])
		query_index = 0
		all_cosine_sims = []

		# ifidf for queries
		for query in queries:
			for sentence in query:
				for word in sentence:
					word = word.lower()
					if word != '.' and word in corpus:
						word_index = corpus.index(word)
						tfidf[query_index][word_index] += idf[word] # multiple adds -> covers tf

			query_tfidf = tfidf[query_index]

			# WITH ESA
			if(isESA == True):
				query_tfidf = self.esa.map_query_to_concept_space(tfidf[query_index])

			# WITHOUT LSA
			# cosine similarities
			if(isLSA == False):
				cosine_sims = []
				for tfidf_doc in tfidf_docs:
					if np.linalg.norm(tfidf_doc) == 0:
						cosine_sim = 0
					else:
						cosine_sim = np.dot(query_tfidf, tfidf_doc)/((np.linalg.norm(query_tfidf)) * (np.linalg.norm(tfidf_doc)))
					cosine_sims.append(cosine_sim)
				all_cosine_sims.append(cosine_sims)
				query_index += 1
			# WITH LSA
			else:
				T = self.index["T"] # txs
				S = self.index["S"] # sxs
				D = self.index["D"] # dxs
				cosine_sims = self.lsa.cosine_similarity(T,S,D,query_tfidf)
				all_cosine_sims.append(cosine_sims)
				query_index += 1

		# ranking docs
		docIDs = [i+1 for i in range(len(tfidf_docs))]
		for i in all_cosine_sims:
			sorted_sim = np.argsort(i)[::-1]
			doc_IDs_ordered.append([docIDs[j] for j in sorted_sim])

		return doc_IDs_ordered
