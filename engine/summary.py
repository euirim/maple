import operator
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt 

from engine.tokenizers import tokenize_to_paragraphs, tokenize_to_sentences


def tfidf_matrix_generator(tokens):
    # Bag of words in vector form
    vectorizer = TfidfVectorizer(stop_words="english",)
    norm_matrix = vectorizer.fit_transform(tokens)
    return norm_matrix * norm_matrix.transpose()
     

# Does not assume newline characters have been eliminated
def summarize(doc, max_units, generate_matrix, paragraphs=False, stem=True):
    if paragraphs:
        units = tokenize_to_paragraphs(doc)
    else:
        doc = doc.replace("\n", " ")
        units = tokenize_to_sentences(doc)

    # stemming
    if stem:
        stemmed_units = []
        stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=True)
        for i, unit in enumerate(units):
            tokens = nltk.word_tokenize(unit)
            for i, token in enumerate(tokens):
                tokens[i] = stemmer.stem(token)

            stemmed_units.append("".join([("" if tok in string.punctuation else " ")+tok 
                for tok in tokens])[1:])
    else:
        stemmed_units = units

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
    if paragraphs:
        divider = "\n\n"
        # deal with long paragraphs
        if False:
            for i, unit in enumerate(summary_units):
                num_sents = len(tokenize_to_sentences(unit))
                if num_sents > 10:
                    summary_units[i] = summarize(unit, int(num_sents/2), generate_matrix,
                            stem=True)
    else:
        divider = " "
    summary = divider.join(summary_units)

    # plotting
#    nx.draw(graph, with_labels=True, node_size=150, node_color="c",
#           font_size=8)
#    plt.title("text1")
#    plt.savefig("figures.png", dpi=400)
#    plt.show()

    return summary


def file_to_doc(filename):
    with open(filename, "r") as myfile:
        return myfile.read() 


def tfidf_summarize(doc, max_units, paragraphs=False):
    return summarize(doc, max_units, tfidf_matrix_generator, paragraphs)
