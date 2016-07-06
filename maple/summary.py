import operator
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt 


def tfidf_matrix_generator(tokens):
    # Bag of words in vector form
    vectorizer = TfidfVectorizer(stop_words="english",)
    norm_matrix = vectorizer.fit_transform(tokens)
    return norm_matrix * norm_matrix.transpose()
     

def tokenize_sentences(doc):
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = sent_tokenizer.tokenize(doc)

    for i, sentence in enumerate(sentences):
        try:
            if sentences[i+1][0].islower():
                sentences[i:i+2] = [sentence+" "+sentences[i+1]]
        except IndexError:
            continue

    return sentences


def tokenize_paragraphs(doc):
    return doc.split("\n")
   

# Does not assume newline characters have been eliminated
def summarize(doc, max_units, generate_matrix, paragraphs=False):
    if paragraphs:
        units = tokenize_paragraphs(doc)
    else:
        doc = doc.replace("\n", " ")
        units = tokenize_sentences(doc)

    # stemming
    stemmed_units = []
    stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=True)
    for i, unit in enumerate(units):
        tokens = nltk.word_tokenize(unit)
        for i, token in enumerate(tokens):
            tokens[i] = stemmer.stem(token)

        stemmed_units.append("".join([("" if tok in string.punctuation else " ")+tok 
            for tok in tokens])[1:])

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


def tfidf_summarize(doc, max_units, paragraphs=False):
    return summarize(doc, max_units, tfidf_matrix_generator, paragraphs)
