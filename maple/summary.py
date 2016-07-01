import operator
import nltk
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
   

# assumes newline characters have been eliminated
def summarize(doc, max_sentences, generate_matrix):
    sentences = tokenize_sentences(doc)

    # stemming
    stemmed_sentences = []
    stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=True)
    for i, sentence in enumerate(sentences):
        words = []
        for token in sentence.split():
           words.append(stemmer.stem(token))

        stemmed_sentences.append(" ".join(words))

    # matrix creation     
    matrix = generate_matrix(stemmed_sentences) 

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
#    nx.draw(graph, with_labels=True, node_size=300, node_color="c")
#    plt.title("text1")
#    plt.savefig("figures.png", dpi=400)
#    plt.show()

    return summary


def tfidf_summarize(doc, max_sentences):
    return summarize(doc, max_sentences, tfidf_matrix_generator)
    
    
    


