#!/usr/bin/env python
"""Maple: automatically summarizes given text using
a modified version of the TextRank algorithm."""

import sys
import codecs
import pickle

from nltk.tokenize.punkt import PunktSentenceTokenizer
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


def train(filename):
    """
    Given file to use as unsupervised data, train tfidfvectorizer and punkt
    sentence tokenizer and output to pickle in data directory.
    """
    text = codecs.open(filename, "rb", "utf8").read()

    abbreviations = [
            "u.s.a", "fig", "gov", "sen", "jus", "jdg", "rep", "pres",
            "mr", "mrs", "ms", "h.r", "s.", "h.b", "s.b", "u.k", "u.n",
            "u.s.s.r",
    ]

    print("TRAINING SENTENCE TOKENIZER...")
    pst = PunktSentenceTokenizer()
    pst.train(text.replace("\n\n", " "))
    # add extra abbreviations
    pst._params.abbrev_types.update(abbreviations)    
    print("TRAINED ABBREVIATIONS: \n{}".format(pst._params.abbrev_types))

    print("TRAINING VECTORIZER...")
    tfv = TfidfVectorizer()
    tfv.fit(text.split("\n\n"))


    # export trained tokenizer + vectorizer
    print("EXPORTING TRAINED TOKENIZER + VECTORIZER...")
    with open("data/punkt.pk", "wb") as pst_out:
        pickle.dump(pst, pst_out)
    with open("data/tfidf.pk", "wb") as tfv_out:
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
