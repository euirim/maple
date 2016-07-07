import os
from engine.summary import tfidf_summarize, file_to_doc
from engine.tokenizers import tokenize_to_sentences, tokenize_to_paragraphs


def test_summarizer(filename, summarizer, max_units, paragraphs=False):
    print("\n********* " + filename.split("/")[-1] + " TEST" + " *********")
    doc = file_to_doc(filename) 
    num_words = len(doc.split())
    print("ORIGINAL TEXT STATISTICS")
    if paragraphs:
        num_units = len(tokenize_to_paragraphs(doc)) 
        print("Number of paragraphs: {}".format(num_units))
    else:
        num_units = len(tokenize_to_sentences(doc))    
        print("Number of sentences: {}".format(num_units))

    print("Number of words: {}\n".format(num_words))

    print("SUMMARY")
    summary = summarizer(doc, max_units, paragraphs) 

    print(summary)
    print("\nSummary Stats: {} units | {} words\n".format(
        max_units, len(summary.split())))

    return 0


def tests_simple():
    # get directory of script calling test
    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    test_summarizer(cdir+"data/test0.txt", tfidf_summarize, 10)
    test_summarizer(cdir+"data/test1.txt", tfidf_summarize, 5)
    test_summarizer(cdir+"data/test2.txt", tfidf_summarize, 5)
    test_summarizer(cdir+"data/test3.txt", tfidf_summarize, 6)
    test_summarizer(cdir+"data/test4.txt", tfidf_summarize, 13)
    test_summarizer(cdir+"data/test5.txt", tfidf_summarize, 10)
    test_summarizer(cdir+"data/test6.txt", tfidf_summarize, 4)
    
    return 0


def tests_diverse():
    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    test_summarizer(cdir+"data/speech1.txt", tfidf_summarize, 10, paragraphs=True)
    test_summarizer(cdir+"data/speech2.txt", tfidf_summarize, 10, paragraphs=True)
    test_summarizer(cdir+"data/debate.txt", tfidf_summarize, 10, paragraphs=True)
    test_summarizer(cdir+"data/floor-short.txt", tfidf_summarize, 10)
    test_summarizer(cdir+"data/floor-long.txt", tfidf_summarize, 10,
        paragraphs=True)
    test_summarizer(cdir+"data/interview.txt", tfidf_summarize, 10, paragraphs=True)

    return 0
