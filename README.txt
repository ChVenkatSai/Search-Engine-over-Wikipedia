This folder contains all the code files that are used in this project - SEARCH ENGINE
This code works for both Python 2 and Python 3.

main.py - The main module that contains the outline of the SEARCH ENGINE
It first preprocesses (Stemming, Tokenization, Stopword removal, etc) the docs and queries.
And then builds the index in informationRetrieval.py for the method used while running the code,
ranks the documents in the decreasing order of their relevance to a particular query,
and finally evaluates the model (evaluation.py) and saves the corresponding plots.

To test the code, run main.py with the appropriate arguments.
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)]

Then the model number must be given as input.
For choosing the model-
        0 for basic Vector Space Model
        1 to add Spellcheck to the Vector Space Model
        2 to add QueryExpansion to the Vector Space Model
        3 to add LSA to the Vector Space Model
        4 to add ESA to the Vector Space Model
        5 for our Best Model

Then the evaluation method must be given as input.
(For the Vector Space Model, there's no comparison, so this question won't appear)
Choose what you want to do with the model-
        Enter 'eval' for evaluating the model
        Enter 'comp' for comparing the model with the Vector Space Model

eval -> precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed and printed for k=1 to 10
     -> All these evaluation metrics are plotted on a single figure and the plot is saved to a file in the OUTPUT folder

comp -> precision@k and recall@k are computed for both the vector space model and the above chosen model for k=1 to 100
     -> precision is plotted against recall (P-R plot) for both the models on the same figure and the plot is saved to a file in the OUTPUT folder

Example:
> python main.py
>
> Choose-
>       0 for basic Vector Space Model
>       1 to add Spellcheck to the Vector Space Model
>       2 to add QueryExpansion to the Vector Space Model
>       3 to add LSA to the Vector Space Model
>       4 to add ESA to the Vector Space Model
>       5 for our Best Model
>
> Enter the model number [0/1/2/3/4/5]: 2            <-- (User must provide this input)
> Choose what you want to do with the model-
>      Enter 'eval' for evaluating the model
>      Enter 'comp' for comparing the model with the Vector Space Model
> What do you want to do with the model?: comp        <-- (User must provide this input)

In this example, the model will be the one where query expansion is added to the VSM.
And comparison between the models is done (P-R plot is generated)

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the (Cranfield) dataset are considered and the evaluation metrics are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.
- (Note that these are overwritten for each run of the code) 

---------
DATASETS:
 - The dataset (for example: Cranfield) related files must be placed in the
   DATASET folder in the same directory as the code files.

 - The wikipedia_concepts_whole_content.json and wikipedia_concepts_only_summary.json files
   (provided in the google drive) must be located in the same directory as the code files.
