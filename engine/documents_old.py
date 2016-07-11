import re
import nltk
from . import tokenizers as tok
from . import summary as summ


# mod = moderate
def mod_max_unit_func(unit_num):
    return 5 * unit_num ** (1/12) 


class Document(object):
    
    def __init__(self, filename=None, text=None,
            max_unit_func=lambda x: 5*x**(1/12)):
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

        self.max_unit_func = max_unit_func
        self.recursive = False

    def build(self):
        self.load()
        self.tokenize()
        self.get_count()
        self.get_summary()

    def load(self):
        if self.filename:
            self.text = summ.file_to_doc(self.filename)
        else:
            pass 

    def tokenize(self):
        regex = re.compile(r"\((Applause|APPLAUSE|Laughter|LAUGHTER)\.\) ?",
                re.IGNORECASE)
        cleaned_text = regex.sub("", self.text)

        self.words = cleaned_text.split() 
        self.sentences = tok.tokenize_to_sentences(
            cleaned_text.replace("\n", " "))
        self.paragraphs = tok.tokenize_to_paragraphs(cleaned_text)

    def get_count(self):
        self.num_words = len(self.words)
        self.num_sentences = len(self.sentences)
        self.num_paragraphs = len(self.paragraphs)

    # both unit_type and num_units must be given to get a fixed summary
    def get_summary(self, unit_type=None, max_units=None, stem=True):
        if unit_type is not None and max_units is not None:
            print("Hello!")
            if unit_type == 0:
                units = self.sentences
                divider = " "
            else:
                units = self.paragraphs 
                # for proper printing
                divider = "\n"

        else:
            if self.num_words > 500 and self.num_paragraphs > 5:
                units = self.paragraphs
                unit_type = 1
                unit_count = self.num_paragraphs
                divider = "\n"
            else:
                units = self.sentences
                unit_type = 0
                unit_count = self.num_sentences
                divider = " " 

            max_units = round(self.max_unit_func(unit_count))

        summary_units = summ.get_tfidf_summary_units(units, max_units, stem)             

        # for long paragraphs
        if unit_type == 1:
            for i, unit in enumerate(summary_units):
                doc = Document(text=unit,)
                doc.max_unit_func = lambda x: 3*x**(1/12)
                doc.recursive = True
                doc.build()
                summary_units[i] = doc.summary

        self.summary = divider.join(summary_units) 

        degree = 1 
        while len(self.summary.split()) > 500:
            self.shorten_summary(degree)
            degree += 1

    def shorten_summary(self, degree):
        doc = Document(text=self.summary, 
                max_unit_func=lambda x: (5-degree)*x**(1/12))
        doc.build()
        self.summary = doc.summary

    def pprint(self):
        print("********* {} *********\n".format(self.filename))
        print("TEXT STATISTICS:")
        print("Word #: {}; Sentence #: {}; Paragraph #: {};\n".format(
            self.num_words, self.num_sentences, self.num_paragraphs))

        print("SUMMARY:\n")
        summary_paragraphs = tok.tokenize_to_paragraphs(self.summary) 
        for sent in summary_paragraphs:
            print(sent)
            print("\n")

        print("SUMMARY STATISTICS:")
        print("Word #: {}: Sentence #: {}; Paragraph #: {};\n".format(
            len(self.summary.split()),
            len(tok.tokenize_to_sentences(self.summary)),
            len(summary_paragraphs),))
