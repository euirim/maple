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
        self.process_text()
        self.tokenize()
        self.get_count()
        self.get_summary()

    def load(self):
        if self.filename:
            self.text = summ.file_to_doc(self.filename)
        else:
            pass 

    # requires tokenize to have already been called
    def process_text(self):
        """
        Apply quotes to reduce ambiguity.
        """
        regex = re.compile(r"\((Applause|APPLAUSE|Laughter|LAUGHTER)\.\) ?",
                re.IGNORECASE)
        cleaned_text = regex.sub("", self.text)
        # ex: here is what I said:
        cleaned_text = re.sub(r": ?\n", " ", cleaned_text)
        cleaned_text = re.sub(r"^[A-Za-z0-9 \.'\-&]*[A-Za-z0-9]+\n", "", cleaned_text)
        self.text = cleaned_text

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
        # 4 sentence minimum for summarization
        if self.num_sentences < 4 and not self.recursive:
            self.summary = "" 
            return 0
        elif unit_type is not None and max_units is not None:
            if unit_type == 0:
                units = self.sentences
                divider = " "
            else:
                units = self.paragraphs 
                # for proper printing
                divider = "\n\n"
        else:
            if self.num_words > 250 and self.num_paragraphs > 5:
                units = self.paragraphs
                unit_type = 1
                unit_count = self.num_paragraphs
                divider = "\n\n"
            else:
                units = self.sentences
                unit_type = 0
                unit_count = self.num_sentences
                divider = " " 

            max_units = round(self.max_unit_func(unit_count))
        
        # quotation ambiguity fixing
#        if unit_type == 0:
#            quoted = False
#            for i, unit in enumerate(units):
#                unit = unit.strip()
#                if unit[0] == "\"":
#                    quoted = True
#                
#                if unit[-1] == "\"":
#                    quoted = False
#
#                if quoted:
#                    if unit[0] == "\"":
#                        unit = unit + "\""
#                    elif unit[-1] == "\"":
#                        unit = "\"" + unit
#                    else:
#                        unit = "\"" + unit + "\""
#
#                units[i] = unit

        summary_units = summ.get_tfidf_summary_units(units, max_units, stem)             

        # for long paragraphs
        if unit_type == 1:
            for i, unit in enumerate(summary_units):
                if unit == "" or unit == " ":
                    continue
                doc = Document(text=unit, max_unit_func=self.max_unit_func)
                doc.recursive = True
                doc.build()
                summary_units[i] = doc.summary

        self.summary = divider.join(summary_units) 

        summary_num_words = len(self.summary.split())

        # for short passages with many short paragraphs (D. Trump)
        if summary_num_words < 200:
            self.summary = re.sub(r"\n\n", " ", self.summary)

        summary_ratio = summary_num_words / self.num_words
        if (not self.recursive and summary_ratio > 0.25):
            self.shorten_summary(summary_ratio*2.5, True)

    # both unit_type and num_units must be given to get a fixed summary
    def get_summary_legacy(self, unit_type=None, max_units=None, stem=True):
        if unit_type is not None and max_units is not None:
            if unit_type == 0:
                units = self.sentences
                divider = " "
            else:
                units = self.paragraphs 
                # for proper printing
                divider = "\n\n"

        else:
            if self.num_words > 250 and self.num_paragraphs > 5:
                units = self.paragraphs
                unit_type = 1
                unit_count = self.num_paragraphs
                divider = "\n\n"
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
                if unit == "" or unit == " ":
                    print("NO UNIT")
                    continue
                doc = Document(text=unit, max_unit_func=self.max_unit_func)
                doc.recursive = True
                doc.build()
                summary_units[i] = doc.summary

        self.summary = divider.join(summary_units) 

        summary_num_words = len(self.summary.split())

        # for short passages with many short paragraphs (D. Trump)
        if summary_num_words < 200:
            self.summary = re.sub(r"\n\n", " ", self.summary)

        summary_ratio = summary_num_words / self.num_words
        if (not self.recursive and summary_ratio > 0.25):
            self.shorten_summary(summary_ratio*2.5, True)

    def shorten_summary(self, degree, recursive):
        doc = Document(text=self.summary, 
                max_unit_func=lambda x: (5-degree)*x**(1/12))
        doc.recursive = recursive
        doc.build()
        self.summary = doc.summary

    def shorten_summary_paragraphs(self):
        pgs = tok.tokenize_to_paragraphs(self.summary)
        for i, unit in enumerate(pgs): 
            doc = Document(text=unit)
            doc.recursive = True
            doc.build()
            pgs[i] = doc.summary

        self.summary = "\n".join(pgs)


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
