import numpy as np

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		# fraction of retrieved documents that are relavant
		count = 0
		for i in query_doc_IDs_ordered[:k]:
			if(i in true_doc_IDs):
				count += 1
		precision = count/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		true_doc_IDs = []
		sumPrecision = 0
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			sumPrecision += self.queryPrecision(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanPrecision = sumPrecision/(len(query_ids))

		return meanPrecision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		# fraction of relavant documents that are retrieved
		count = 0
		for i in true_doc_IDs:
			if(i in query_doc_IDs_ordered[:k]):
				count += 1
		recall = count/(len(true_doc_IDs))
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		true_doc_IDs = []
		sumRecall = 0
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			sumRecall += self.queryRecall(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanRecall = sumRecall/(len(query_ids))

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		P = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		R = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if(P==0 and R==0):
			fscore = 0
		else:
			fscore = (2*P*R)/(P+R) # 2PR/(P+R)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		true_doc_IDs = []
		sumFscore = 0
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			sumFscore += self.queryFscore(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanFscore = sumFscore/(len(query_ids))

		return meanFscore


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		true_doc_ids = true_doc_IDs[0]
		true_doc_relavances = true_doc_IDs[1]

		relavance_list = []

		# DCG
		DCG = 0
		for i, doc_ID in enumerate(query_doc_IDs_ordered[:k]):
			if doc_ID in true_doc_ids:
				doc_index = true_doc_ids.index(doc_ID)
				relavance = true_doc_relavances[doc_index]
				relavance_list.append([relavance,i+1])
				DCG += relavance/(np.log2(i + 2))

		# ---------- CORRECTED THE IDCG CALCULATION -----------
		ideal_relavance_list = []

		for i, doc_ID in enumerate(query_doc_IDs_ordered):
			if doc_ID in true_doc_ids:
				doc_index = true_doc_ids.index(doc_ID)
				relavance = true_doc_relavances[doc_index]
				ideal_relavance_list.append([relavance,i+1])
		ideal_relavance_list = list(sorted(ideal_relavance_list, key = lambda item : item[0], reverse = True))
		ideal_relavance_list = ideal_relavance_list[:k]

		# IDCG
		IDCG = 0
		for i in range(len(ideal_relavance_list)):
			relavance = ideal_relavance_list[i][0]
			IDCG += relavance/(np.log2(i + 2))

		# nDCG
		if IDCG == 0:
			nDCG = 0
		else:
			nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		true_doc_IDs = []
		sumNDCG = 0
		for i in query_ids:
			true_doc_IDs_ids = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			true_doc_IDs_rels = [qrel["position"] for qrel in qrels if qrel["query_num"]==str(i)] # need relavances also
			true_doc_IDs = [true_doc_IDs_ids, true_doc_IDs_rels]
			sumNDCG += self.queryNDCG(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanNDCG = sumNDCG/(len(query_ids))

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		sumP = 0
		countP = 0
		for i in range(k):
			if(query_doc_IDs_ordered[i] in true_doc_IDs):
				countP += 1
				sumP += countP/(i + 1)
		if(countP == 0):
			avgPrecision = 0
		else:
			avgPrecision = sumP/countP

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		true_doc_IDs = []
		sumAveragePrecision = 0
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in q_rels if qrel["query_num"]==str(i)]
			sumAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanAveragePrecision = sumAveragePrecision/(len(query_ids))

		return meanAveragePrecision
