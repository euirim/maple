#!/usr/bin/env python
"""Maple: automatically summarizes given text using
a modified version of the TextRank algorithm."""

import sys
import summary
    
def file_to_doc(filename):
    with open(filename, "r") as myfile:
        return myfile.read().replace("\n", " ")

def main():
    doc0 = file_to_doc("../tests/test0.txt")
    doc2 = file_to_doc("../tests/test2.txt")
    doc3 = file_to_doc("../tests/test3.txt")
    doc4 = file_to_doc("../tests/test4.txt")
    doc5 = file_to_doc("../tests/test5.txt")
    doc6 = file_to_doc("../tests/test6.txt")

    print("test0.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc0))))
    print(summary.tfidf_summarize(doc0, 10)) 

    print("test2.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc2))))
    print(summary.tfidf_summarize(doc2, 5)) 

    print("test3.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc3))))
    print(summary.tfidf_summarize(doc3, 6)) 
        
    print("test4.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc4))))
    print(summary.tfidf_summarize(doc4, 13)) 

    print("test5.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc5))))
    print(summary.tfidf_summarize(doc5, 10))    

    print("test6.txt test\n")
    print("Number of sentences:"
            " {}".format(len(summary.tokenize_sentences(doc6))))
    print(summary.tfidf_summarize(doc6, 6))       

if __name__ == "__main__":
    sys.exit(main())
