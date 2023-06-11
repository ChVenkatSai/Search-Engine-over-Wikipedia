class Utilities():

    corpus = None # accessible to all class instances

    def inverted_index(self, docs):
        """
		Representing the terms in corpus in terms of docs as inverted index

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document

		Returns
		-------
		dict, list
			- dict: Each key is a term in the corpus.
                    Each value is an array (arr) of length 2
                    Where arr[0] represents the doc_ID in which the term is present
                    And arr[1] represents the number of times the term appeared in that document

			- list: A list of all the terms in the corpus
		"""
        inv_index={}
        corpus = []
        doc_index = 1
        for doc in docs:
            for sentence in doc:
                for word in sentence:
                    if word != '.' and word not in corpus:
                        corpus.append(word)
                    if word != '.':
                        if word not in inv_index:
                            inv_index[word] = [[doc_index,0]]
                        if inv_index[word][-1][0] == doc_index:
                            inv_index[word][-1][1] += 1
                        else:
                            inv_index[word].append([doc_index,1])
            doc_index += 1

            Utilities.corpus = corpus

        return inv_index, corpus

    def concepts_inverted_index(self, concepts):
        """
		Representing the terms in corpus in terms of concepts as inverted index

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a concept (wikipedia article) and each sub-sub-list is a sentence of the concept

		Returns
		-------
		dict
			- dict: Each key is a term in the corpus.
                    Each value is an array (arr) of length 2
                    Where arr[0] represents the concept_ID in which the term is present
                    And arr[1] represents the number of times the term appeared in that concept

		"""

        corpus = Utilities.corpus
        concepts_inv_index = {}
        concept_index = 1
        for concept in concepts:
            for sentence in concept:
                words = list(set.intersection(set(corpus), set(sentence)))
                for word in words:
                    if word != '.':
                        if word not in concepts_inv_index:
                            concepts_inv_index[word] = [[concept_index,0]]
                        if concepts_inv_index[word][-1][0] == concept_index:
                            concepts_inv_index[word][-1][1] += 1
                        else:
                            concepts_inv_index[word].append([concept_index, 1])
            concept_index += 1

        return concepts_inv_index
