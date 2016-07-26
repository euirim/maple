# FIELD TEST SCRIPT

import os
import pandas
import numpy as np
from engine.documents import Document


def csv_to_df(filename):
    df = pandas.read_csv(filename, encoding="latin1",dtype={"key": np.string_})
    df = df.rename(columns = {"created": "intelligent_summary", "key": "url"})
    df["num_words"] = np.nan
    df["num_sentences"] = np.nan
    df["num_paragraphs"] = np.nan
    return df 


def generate_test_file(srcname, outputdir):
    df = csv_to_df(srcname)

    # find speech_id column number
    for id_num, col in enumerate(df.columns):
        if "speech_id" in col:
            break
            s_id_col = id_num

    for i, row in df.iterrows():
        if not pandas.isnull(row[5]):
            print(row[2])
            doc = Document(text=row[5])
            doc.build()
            df.set_value(i, df.columns[8], doc.summary)
            df.set_value(i, df.columns[9], doc.num_words)
            df.set_value(i, df.columns[10], doc.num_sentences)
            df.set_value(i, df.columns[11], doc.num_paragraphs)

            pk = str(row[id_num])
            df.set_value(i, df.columns[6],
                "http://votesmart.org/public-statement/"+pk)

    filepath = outputdir + "/" + srcname.split("/")[-1]

    df.to_csv(filepath, index=False, encoding="latin1")
    print("Finished {}. Saved to {}".format(srcname.split("/")[-1], filepath))


def generate_test_files(outputdir):
    outputdir = os.path.expanduser(outputdir)

    print(outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print("Hello")

    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    files = []
    for f in os.listdir(cdir+"data/data_field/selected"): 
        if f.endswith(".csv"):
            files.append(f)

    print(files)
    for f in files:
        generate_test_file(cdir+"data/data_field/selected/"+f, outputdir)

    return 0
