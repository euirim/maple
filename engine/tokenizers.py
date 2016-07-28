import re
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer as PST, PunktParameters


def tokenize_to_sentences(doc):
    sent_tokenizer = nltk.data.load("../data/punkt.pk", format="pickle")
    sentences = sent_tokenizer.tokenize(doc)

#    regex = re.compile(
#            "(M[rRsS]{1,2}|Sen|SEN|Rep|REP|Gov|GOV|Pres|PRES|Jdg|JDG|Jus|JUS|u\.s\.a\.)\.$"
#            )
    for i, sentence in enumerate(sentences):
        try:
            if sentences[i+1][0].islower():
                sentences[i:i+2] = [sentence+" "+sentences[i+1]]

#            if bool(regex.search(sentence)):
#                sentences[i:i+2] = [sentence+" "+sentences[i+1]]
        except IndexError:
            continue

    return sentences


def tokenize_to_sentences2(doc):
    punkt_param = PunktParameters()
    abbreviations = [
            "u.s.a", "fig", "gov", "sen", "jus", "jdg", "rep", "pres",
            "mr", "mrs", "ms", "h.r", "s.", "h.b", "s.b", "u.k", "u.n",
            "u.s.s.r",
    ]
    punkt_param.abbrev_types = set(abbreviations)
    tokenizer = PST(punkt_param)
    return tokenizer.tokenize(doc)


def tokenize_to_paragraphs(doc):
    return doc.split("\n")


def tokenize_to_remark(doc):
    raw_remarks = re.split("""(\n((The|THE)[
            ])?((Senator|SENATOR|Governor|GOVERNOR|President|PRESIDENT|Representative|REPRESENTATIVE|Speaker|SPEAKER|Justice|JUSTICE|Judge|JUDGE|SHERIFF|Sheriff)|(REP|Rep|GOV|Gov|PRES|Pres|JUS|Jus|JDG|Jdg|SEN|Sen)\.?)[
            ]?([A-Z\-\'][A-Za-z\-\']*)?[\.;:]|\nM[RSrs]{1,2}\.?
            ([A-Za-z\-\']*[A-Z]){1,2}[A-Za-z\-\']*[\.:;]|\n[A-Z,'\-]+
            ([A-Z,'\-]+
            ?)+[\.:;]|\n(?!M[RSrs]{1,2})(([A-Za-z\-\']*[A-Z]){1,2}[A-Za-z\-\']*)[\.:;])""", "\n"+doc)

    print(raw_remarks)
