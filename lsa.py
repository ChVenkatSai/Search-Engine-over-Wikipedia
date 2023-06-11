import numpy as np

class LSA():

    def reduced_tfidf(self, tfidf):
        """
		Parameters
		----------
		arg1 : matrix
			The tfidf matrix of all the docs with all the terms in the corpus
		Returns
		-------
			The reduced tfidf matrix and the matrices obtained after SVD
            and reducing the dimensionality
		"""
        tfidf = np.transpose(tfidf)
        len_corpus = np.shape(tfidf)[0]
        len_docs = np.shape(tfidf)[1]

        u,s,vt = np.linalg.svd(tfidf); # u: txt, s: txd, vt = dxd
        s_size = np.size(s) # 1400 = d
        s_values = np.zeros([len_corpus, len_docs]) # txd
        s_values[:s_size, :s_size] = np.diag(s) # dxd is diagonal
        us = np.dot(u,s_values) # txd
        tfidf1 = np.dot(us, vt) # txd

        s_values_k = s_values[:300, :300] # sxs
        u_k = u[:, :300] # txs
        vt_k = vt[:300, :] # sxd
        us_k = np.dot(u_k,s_values_k) # txs
        tfidf_k = np.dot(us_k, vt_k) # txd

        return tfidf_k, u_k, s_values_k, vt_k

    def cosine_similarity(self, T, S, D, tfidf_query):
        """
		Parameters
		----------
		arg1 : matrix
			Terms in concept space, after SVD and reducing dimensionality
        arg2 : matrix
			Singular values after SVD on tfidf matrix
        arg3 : matrix
			Docs in concept space, after SVD and reducing dimensionality
        arg4 : vector
			The tfidf vector of a query with all the terms in the corpus
		Returns
		-------
			The cosine similarities of the query with all the documents in the dataset
		"""
        DS_docs_matrix = np.dot(D, S) # dxs
        DS_query = np.dot(tfidf_query,T) # 1xs

        cosine_sims = []
        for DS_doc in DS_docs_matrix: # 1xs
            if np.linalg.norm(DS_doc) == 0:
                cosine_sim = 0
            else:
                cosine_sim = np.dot(DS_query, DS_doc)/((np.linalg.norm(DS_query)) * (np.linalg.norm(DS_doc)))
            cosine_sims.append(cosine_sim)
        return cosine_sims
