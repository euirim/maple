import re
import nltk


def tokenize_to_sentences(doc):
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = sent_tokenizer.tokenize(doc)

    for i, sentence in enumerate(sentences):
        try:
            if sentences[i+1][0].islower():
                sentences[i:i+2] = [sentence+" "+sentences[i+1]]
        except IndexError:
            continue

    return sentences


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
