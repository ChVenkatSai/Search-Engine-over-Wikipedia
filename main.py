from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from esa import ESA # ESA
from spellcheck import SpellCheck # spellcheck

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
	pass
elif version_info.major == 2:
	try:
		input = raw_input
	except NameError:
		pass
else:
	print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args, model, method):
		self.args = args
		self.isVSM = False
		self.addSpellCheck = False
		self.addQueryExpansion = False
		self.addLSA = False
		self.addESA = False
		self.isBestModel = False
		if model == 0:
			self.isVSM = True
		elif model == 1:
			self.addSpellCheck = True
		elif model == 2:
			self.addQueryExpansion = True
		elif model == 3:
			self.addLSA = True
		elif model == 4:
			self.addESA = True
		elif model == 5:
			self.isBestModel = True
			self.addLSA = True

		self.method = method

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()

		self.esa = ESA() # ESA
		self.spellcheck = SpellCheck() # SpellCheck


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text, isQueryExpansion=False):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text, isQueryExpansion)


	def preprocessQueries(self, queries, isSpellCheck, isQueryExpansion): # CHANGE AFTER COMPARISON
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Spellcheck on queries
		if(isSpellCheck): # Perform spellcheck if isSpellCheck is True
			queries = [self.spellcheck.correctQuery(query) for query in queries]

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = None
			if isQueryExpansion: # Perform Query Expansion if isQueryExpansion is True
				stopwordRemovedQuery = self.removeStopwords(query, True)
			else:
				stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs, isConcepts = False):
		"""
		Preprocess the documents
		"""

		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		if(isConcepts):
			json.dump(segmentedDocs, open(self.args.out_folder + "segmented_concepts.txt", 'w'))
		else:
			json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		if(isConcepts):
			json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_concepts.txt", 'w'))
		else:
			json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		if(isConcepts):
			json.dump(reducedDocs, open(self.args.out_folder + "reduced_concepts.txt", 'w'))
		else:
			json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		if(isConcepts):
			json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_concepts.txt", 'w'))
		else:
			json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs

	def getProcessedQueries(self, queries):
		"""
		Call the preprocessQueries function with required arguments according to the model
		"""
		if(self.addSpellCheck):
			return self.preprocessQueries(queries, True, False)
		if(self.addQueryExpansion):
			return self.preprocessQueries(queries, False, True)
		else:
			return self.preprocessQueries(queries, False, False)

	def plotEvaluationMetrics(self, doc_IDs_ordered, query_ids, qrels, title, savename):
		"""
		Calculate and plot the evaluation metrics according to the model
		"""
		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +
				str(k) + " : " + str(precision) + ", " + str(recall) +
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		#Plot the metrics and save plot
		plt.figure()
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics" + title + "- Cranfield Dataset")
		plt.savefig(args.out_folder + savename + ".png")
		return

	def plotPRCurves(self, doc_IDs_ordered_old, doc_IDs_ordered_new, query_ids, qrels):
		"""
		Calculate and plot the precisions and recalls for VSM and a new model for ease of comparison
		"""

		precisions, recalls = [], []
		for k in range(1, 101):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered_old, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered_old, query_ids, qrels, k)
			recalls.append(recall)

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions_new, recalls_new = [], []
		for k in range(1, 101):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered_new, query_ids, qrels, k)
			precisions_new.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered_new, query_ids, qrels, k)
			recalls_new.append(recall)

		# Plot the metrics and save plot
		label = None
		savename = None
		if(self.addSpellCheck):
			label = "Spell Check"
			savename = "spellcheck"
		elif(self.addQueryExpansion):
			label = "Query Expansion"
			savename = "queryexpansion"
		elif(self.addLSA):
			label = "LSA"
			savename = "lsa"
		elif(self.addESA):
			label = "ESA"
			savename = "esa"
		if(self.isBestModel):
			plt.figure()
			plt.plot(recalls, precisions, label="VSM");
			plt.plot(recalls_new, precisions_new, label="Best Model")
			plt.legend()
			plt.xlabel("R")
			plt.ylabel("P")
			plt.title("P-R curve - comparison")
			plt.savefig(args.out_folder + "pr_curves_best_model_comparison.png")
			print("PR plot generated and saved in the " + args.out_folder + " folder.")
			return
		plt.figure()
		plt.plot(recalls, precisions, label="Without " + label);
		plt.plot(recalls_new, precisions_new, label="With " + label)
		plt.legend()
		plt.xlabel("R")
		plt.ylabel("P")
		plt.title("P-R curve - comparison")
		plt.savefig(args.out_folder + "pr_curves_" + savename + "_comparison.png")
		print("PR plot generated and saved in the " + args.out_folder + " folder.")
		return


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

		# Process queries - VSM
		processedQueries_old = self.preprocessQueries(queries, False, False)

		# Process queries - new model
		processedQueries_new = self.getProcessedQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids, False) # --> Without LSA - CHECK IF PROBLEMATIC

		# --- ESA ---
		if(self.addESA):
			# Read concepts
			concepts_json = json.load(open("wikipedia_concepts_whole_content.json", 'r'))[:]

			# get summary/content of the concepts from wikipedia
			# - RAN ONCE (2.5 hr process) and stored in wikipedia_concepts_only_summary.json and wikipedia_concepts_whole_content.json
			# doc_titles = [item["title"] for item in docs_json]
			# self.esa.GetConceptsFromWikipedia(doc_titles)
			# concepts = [item["summary"] for item in concepts_json] # for summary
			concepts = [item["content"] for item in concepts_json] # for content

			# Process concepts
			processedConcepts = self.preprocessDocs(concepts, True)
			self.esa.buildConceptsIndex(processedConcepts)

		# Rank the documents for each query - for VSM
		doc_IDs_ordered_old = self.informationRetriever.rank(processedQueries_old, False, False) # without anything

		if(self.addLSA):
			# Build document index with LSA
			self.informationRetriever.buildIndex(processedDocs, doc_ids, True) # --> With LSA

		# Rank the documents for each query - for new model
		doc_IDs_ordered_new = self.informationRetriever.rank(processedQueries_new, self.addLSA, self.addESA)

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Plot the required metrics
		if(self.method == 'eval'):
			if(self.isBestModel):
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " for Best Model ", "eval_plot_bestmodel")
			elif(self.addSpellCheck):
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " with Spell Check ", "eval_plot_spellcheck")
			elif(self.addQueryExpansion):
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " with Query Expansion ", "eval_plot_queryexpansion")
			elif(self.addLSA):
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " with LSA ", "eval_plot_lsa")
			elif(self.addESA):
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " with ESA ", "eval_plot_esa")
			else:
				self.plotEvaluationMetrics(doc_IDs_ordered_new, query_ids, qrels, " ", "eval_plot_vsm")
		elif(self.method == 'comp'):
			if(self.isVSM == False):
				self.plotPRCurves(doc_IDs_ordered_old, doc_IDs_ordered_new, query_ids, qrels)

	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/",
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "demo-output/",
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true",
						help = "Take custom query as input")

	# Parse the input arguments
	args = parser.parse_args()

	# Chooose method(s) to include in the model
	print("\nChoose-")
	print("\t0 for basic Vector Space Model")
	print("\t1 to add Spellcheck to the Vector Space Model")
	print("\t2 to add QueryExpansion to the Vector Space Model")
	print("\t3 to add LSA to the Vector Space Model")
	print("\t4 to add ESA to the Vector Space Model")
	print("\t5 for our Best Model\n")
	print("Enter the model number [0/1/2/3/4/5]:", end=" ")
	model = int(input())
	method = 'eval'
	# Choose whether to evaluate the model or compare the model with the VSM
	if(model != 0):
		print("Choose what you want to do with the model-")
		print("\tEnter 'eval' for evaluating the model")
		print("\tEnter 'comp' for comparing the model with the Vector Space Model")
		print("What do you want to do with the model?:", end=" ")
		method = input()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args, model, method)

	# Either handle query from user or evaluate on the complete dataset
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
