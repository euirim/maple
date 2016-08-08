#!/usr/bin/env python
"""Maple: automatically summarizes given text using
a modified version of the TextRank algorithm."""

import sys
import codecs
import pickle
import string

import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from tests.tests_visual import test_summarizer, tests_simple, tests_diverse 
from tests.tests_field import generate_test_files
from tests import tests_alpha


def field_test():
    generate_test_files("~/Documents/summ_test_files/selected")

def alpha_test():
    tests_alpha.generate_test_files("output")


def test(simple=True):
    if simple:
        print("********* MAPLE'S SIMPLE TESTS *********\n")
        tests_simple()
    else:
        print("********* MAPLE'S DIVERSE TESTS *********\n")
        tests_diverse()

    print("********* TESTS COMPLETED *********")


def train(filename, stem=True):
    """
    Given file to use as unsupervised data, train tfidfvectorizer and punkt
    sentence tokenizer and output to pickle in data directory.
    """
    text = codecs.open(filename, "rb", "utf8").read()

    abbreviations = [
            "u.s.a", "fig", "gov", "sen", "jus", "jdg", "rep", "pres",
            "mr", "mrs", "ms", "h.r", "s.", "h.b", "s.b", "u.k", "u.n",
            "u.s.s.r", "u.s",
    ]

    print("TRAINING SENTENCE TOKENIZER...")
    pst = PunktSentenceTokenizer()
    pst.train(text.replace("\n\n", " "))
    # add extra abbreviations
    pst._params.abbrev_types.update(abbreviations)    
    print("TRAINED ABBREVIATIONS: \n{}".format(pst._params.abbrev_types))
    
    # stemming
    if stem:
        wnl = WordNetLemmatizer()
        print("WORD TOKENIZING TEXT")
        tokens = nltk.word_tokenize(text)
        
        # pos tagging
        print("POS TAGGING TEXT...")
        tagged_tokens = pos_tag(tokens)

        print("STEMMING TRAINING TEXT...")
        for i, tok in enumerate(tagged_tokens):
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
                tokens[i] = wnl.lemmatize(tok[0], position)

            if i % 1000000 == 0:
                print("TOKEN: {}".format(i))

        text = "".join([("" if tok in string.punctuation else " ")+tok 
                for tok in tokens])
        text = text.strip() 
    
    print("TRAINING VECTORIZER...")
    tfv = TfidfVectorizer()
    tfv.fit(pst.tokenize(text))

    # export trained tokenizer + vectorizer
    print("EXPORTING TRAINED TOKENIZER + VECTORIZER...")
    if stem:
        punkt_out_filename = "data/punkt_stem.pk"
        tfidf_out_filename = "data/tfidf_stem.pk"
    else:
        punkt_out_filename = "data/punkt.pk"
        tfidf_out_filename = "data/tfidf.pk"

    with open(punkt_out_filename, "wb") as pst_out:
        pickle.dump(pst, pst_out)
    with open(tfidf_out_filename, "wb") as tfv_out:
        pickle.dump(tfv, tfv_out)

    print("EXPORTING COMPLETED")
    return


def main(argv):       
    if argv[0] == "-t":
        try:
            test(bool(int(argv[1])))
            return 0
        except:
            print("Enter True or False as second parameter for testing.\n")
            return 1
    elif (len(argv) > 3) or (len(argv) < 3) or (argv[0] == "-h"):
        print("./maple.py (optional -test true or false) <filename>" 
                " <max_units> <units (-p or -s)>")
        return 1
    
    if argv[2] == "-p":
        paragraphs = True
    else:
        paragraphs = False

    test_summarizer(filename)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
