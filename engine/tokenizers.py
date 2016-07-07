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
