import operator
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt 
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from engine.tokenizers import tokenize_to_paragraphs, tokenize_to_sentences


def tfidf_matrix_generator(tokens):
    # Bag of words in vector form
    with open("data/tfidf_stem.pk", "rb") as trained_vectorizer_file:
        vectorizer = pickle.loads(trained_vectorizer_file.read())
        
    norm_matrix = vectorizer.fit_transform(tokens)
    return norm_matrix * norm_matrix.transpose()
     

def generate_summary_units(units, max_units, generate_matrix, stem=True):
    wnl = WordNetLemmatizer()
    stemmed_units = []
    for i, unit in enumerate(units):
        tokens = nltk.word_tokenize(unit)
        # pos tagging
        tagged_tokens = pos_tag(tokens)

        for tok_i, tok in enumerate(tagged_tokens):
            position = None
            if tok[1] == "NN" or tok[1] == "NNS" or tok[1] == "NNPS":
                position = wordnet.NOUN
            elif "JJ" in tok[1]:
                position = wordnet.ADJ
            elif "VB" in tok[1]:
                position = wordnet.VERB
            elif "RB" in tok[1]:
                position = wordnet.ADV

            if position:
                tokens[tok_i] = wnl.lemmatize(tok[0], position)

        stemmed_units.append("".join([("" if tok in string.punctuation else " ")+tok 
                for tok in tokens]).strip())

    # matrix creation     
    matrix = generate_matrix(stemmed_units) 

    # graph generation
    graph = nx.from_scipy_sparse_matrix(matrix)

    # PageRank
    scores = nx.pagerank_scipy(graph, max_iter=100)

    # generate summary
    pagerank = sorted(scores.items(), 
            key=operator.itemgetter(1),
            reverse=True)[:max_units]
    summary_indexes = sorted(pagerank)
    summary_units = [units[i] for i, score in summary_indexes] 

    return summary_units


def file_to_doc(filename):
    with open(filename, "r") as myfile:
        return myfile.read() 


def get_tfidf_summary_units(units, max_units, stem):
    return generate_summary_units(units, max_units, tfidf_matrix_generator, stem)
