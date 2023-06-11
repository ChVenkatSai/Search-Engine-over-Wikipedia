import string
from nltk.corpus import wordnet
# from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords

def query_expansion(query):

    stop_words=set(stopwords.words("english"))
    query=query.lower()
    query=query.translate(str.maketrans('','',string.punctuation))
    words=query.split()
    print(words)
    # word_tokens = word_tokenize(query)

    synonyms=[]

    count=0
    for x in words:

        for syn in wordnet.synsets(x):
            for l in syn.lemmas() :
                if(count<3):
                    if l.name() not in synonyms:
                        synonyms.append(l.name())
                        count+=1

        count=0

    synonyms_string=' '.join(synonyms)
    new_query=" ".join([synonyms_string])
    return new_query

query = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."
new_query = query_expansion(query)
print(new_query)
