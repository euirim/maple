# FIELD TEST SCRIPT

import os
import pandas
import numpy as np
from engine.documents import Document


def csv_to_df(filename):
    df = pandas.read_csv(filename, encoding="latin1",
            dtype={"speech_id":np.string_,})
    df = df.rename(columns = {"speech_id": "url",})
    df["num_sentences"] = np.nan
    df["i_summary_1"] = ""
    df["i_summary_2"] = ""
    return df 


def generate_test_file(srcname, outputdir):
    df = csv_to_df(srcname)

    s_id_col = 0

    for i, row in df.iterrows():
        if not pandas.isnull(row[3]):
            print(row[1])
            doc = Document(text=row[3])
            doc.build()
            df.set_value(i, df.columns[4], doc.num_sentences)
            df.set_value(i, df.columns[5], doc.summary)
            # extra summary restricted to 4 sentences
            doc.get_summary(unit_type=0, max_units=4)
            df.set_value(i, df.columns[6], doc.summary)

            pk = str(row[s_id_col])
            df.set_value(i, df.columns[s_id_col],
                "http://votesmart.org/public-statement/"+pk)

    filepath = outputdir + "/" + srcname.split("/")[-1]

    df.to_csv(filepath, index=False, encoding="latin1")
    print("Finished {}. Saved to {}".format(srcname.split("/")[-1], filepath))


def generate_test_files(outputdir):
    outputdir = os.path.expanduser(outputdir)

    print(outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    cdir = os.path.dirname(os.path.realpath(__file__)) + "/"
    files = []
    for f in os.listdir(cdir+"data/data_alpha"): 
        if f.endswith(".csv"):
            files.append(f)

    print(files)
    for f in files:
        generate_test_file(cdir+"data/data_alpha/"+f, outputdir)

    return 0
