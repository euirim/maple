import re
import nltk
from . import tokenizers as tok
from . import summary as summ


class Document(object):
    
    def __init__(self, filename=None, text=None):
        self.filename = filename
        self.text = text 
        
        # original text
        self.words = None
        self.sentences = None
        self.paragraphs = None
        
        self.num_words = None
        self.num_paragraphs = None
        self.num_sentences = None
        
        self.genre = None
        self.summary = None

    def build(self):
        self.load()
        self.tokenize()
        self.get_count()
        self.get_summary()

    def load(self):
        if self.filename:
            self.text = summ.file_to_doc(self.filename)
        else:
            print("No associated filename.")
 
    def tokenize(self):
        self.words = self.text.split() 
        self.sentences = tok.tokenize_to_sentences(
            self.text.replace("\n", " "))
        self.paragraphs = tok.tokenize_to_paragraphs(self.text)

    def get_count(self):
        self.num_words = len(self.words)
        self.num_sentences = len(self.sentences)
        self.num_paragraphs = len(self.paragraphs)

    # both unit_type and num_units must be given to get a fixed summary
    def get_summary(self, unit_type=None, max_units=None, stem=True):
        if unit_type and max_units:
            if unit_type == 0:
                units = self.sentences
                divider = " "
            else:
                units = self.paragraphs 
                # for proper printing
                divider = "\n\n"

        else:
            if self.num_words > 1000 and self.num_paragraphs > 10:
                units = self.paragraphs
                unit_type = 1
                unit_count = self.num_paragraphs
                divider = "\n\n"
            else:
                units = self.sentences
                unit_type = 0
                unit_count = self.num_sentences
                divider = " " 

            max_units = round(6 * unit_count ** (1/9))

        summary_units = summ.get_tfidf_summary_units(units, max_units, stem)             

        # for long paragraphs
        if unit_type == 1:
            for i, unit in enumerate(summary_units):
                if re.match("\((Applause|APPLAUSE|Laughter|LAUGHTER)\.\))",
                        unit):
                    del summary_units[i]
                    continue
                print(i)
                doc = Document(text=unit)
                doc.build()
                summary_units[i] = doc.summary

        self.summary = divider.join(summary_units) 
        
    def pprint(self):
        print("********* {} *********\n".format(self.filename))
        print("TEXT STATISTICS:")
        print("Word #: {}; Sentence #: {}; Paragraph #: {};\n".format(
            self.num_words, self.num_paragraphs, self.num_sentences))

        print("SUMMARY:\n")
        print(self.summary)
        print("\nSUMMARY STATISTICS:")
        print("Word #: {}: Sentence #: {}; Paragraph #: {};\n".format(
            len(self.summary.split()),
            len(tok.tokenize_to_sentences(self.text)),
            len(tok.tokenize_to_paragraphs(self.text))))
