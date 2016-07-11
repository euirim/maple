import os
from engine.documents import Document

def test_summarizer(filename):
    doc = Document(filename=filename)
    doc.build()
    doc.pprint()

    return 0


def tests_simple():
    # get directory of script calling test
    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    test_summarizer(cdir+"data/test0.txt")
    test_summarizer(cdir+"data/test1.txt")
    test_summarizer(cdir+"data/test2.txt")
    test_summarizer(cdir+"data/test3.txt")
    test_summarizer(cdir+"data/test4.txt")
    test_summarizer(cdir+"data/test5.txt")
    test_summarizer(cdir+"data/test6.txt")
    test_summarizer(cdir+"data/test7.txt")   
    test_summarizer(cdir+"data/test8.txt")   
    test_summarizer(cdir+"data/test9.txt")   
    test_summarizer(cdir+"data/test10.txt")   
    test_summarizer(cdir+"data/test11.txt")   
    test_summarizer(cdir+"data/test12.txt")   
    test_summarizer(cdir+"data/test13.txt") 
    test_summarizer(cdir+"data/test14.txt") 
    test_summarizer(cdir+"data/test15.txt") 
    test_summarizer(cdir+"data/test16.txt") 

    return 0


def tests_diverse():
    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    test_summarizer(cdir+"data/speech1.txt")
    test_summarizer(cdir+"data/speech2.txt")
    test_summarizer(cdir+"data/debate.txt")
    test_summarizer(cdir+"data/floor-short.txt")
    test_summarizer(cdir+"data/floor-long.txt")
    test_summarizer(cdir+"data/interview.txt")

    #doc = file_to_doc(cdir+"data/debate.txt")
    #tokenize_to_remark(doc)

    return 0
