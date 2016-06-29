import operator
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt 


def tfidf_matrix_generator(tokens):
    # Bag of words in vector form
    vectorizer = TfidfVectorizer(stop_words="english")
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
   

# assumes newline characters have been eliminated
def summarize(doc, max_sentences, generate_matrix):
    sentences = tokenize_sentences(doc)

    # matrix creation     
    matrix = generate_matrix(sentences) 

    # graph generation
    graph = nx.from_scipy_sparse_matrix(matrix)

    # PageRank
    scores = nx.pagerank_scipy(graph, max_iter=100)

    # generate summary
    pagerank = sorted(scores.items(), 
            key=operator.itemgetter(1),
            reverse=True)[:max_sentences]
    summary_indexes = sorted(pagerank)
    summary_sents = [sentences[i] for i, score in summary_indexes] 
    summary = " ".join(summary_sents)

    # plotting
    nx.draw(graph, with_labels=True, node_size=300, node_color="c")
    plt.title('<C>={}'.format(sentences[0]))
    plt.show()
    plt.savefig("~/Desktop/" + summary_sents[0] + ".png", dpi=400)

    return summary


def tfidf_summarize(doc, max_sentences):
    return summarize(doc, max_sentences, tfidf_matrix_generator)
    
    
    


