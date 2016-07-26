#!/usr/bin/env python
"""Maple: automatically summarizes given text using
a modified version of the TextRank algorithm."""

import sys

from tests.tests_visual import test_summarizer, tests_simple, tests_diverse 
from tests.tests_field import generate_test_files


def csv_test():
    generate_test_files("~/Documents/summ_test_files/selected")


def test(simple=True):
    if simple:
        print("********* MAPLE'S SIMPLE TESTS *********\n")
        tests_simple()
    else:
        print("********* MAPLE'S DIVERSE TESTS *********\n")
        tests_diverse()

    print("********* TESTS COMPLETED *********")


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
