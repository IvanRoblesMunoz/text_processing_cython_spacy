#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:07:58 2020

@author: ivan
"""



# =============================================================================
# import libraries
# =============================================================================
# --- import spacy ---
import spacy
from pathlib import Path
from datetime import datetime
# =============================================================================
# Paths
# =============================================================================
nlp_path = Path("/home/ivan/git/natural-language-processing/")
data_path = nlp_path / "week3/data"

# =============================================================================
# import data
# =============================================================================
# --- read text data ----
def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data

sentences = read_corpus(data_path/ "train.tsv")
sentences = [i[0] for i in sentences][:100000]

# sentences = ['the car is car' for i in range(0,5)]


# --- Load spacy English ---
nlp = spacy.load("en_core_web_sm",
                 disable=["tagger", # tagger takes 4x longer
                          "parser",
                          "ner",
                          "textcat"]
                 )



# --- Custom cython libraries ----
import spacy_tokenize as st


# =============================================================================
# preprocess
# =============================================================================
# --- make doclist ---
from datetime import datetime as dt
start_nlp = dt.now()
sentences = [doc for doc  in nlp.pipe(sentences)]

end_nlp = dt.now()

start_pipeline = dt.now()
res = st.run_pipeline(sentences)
end_pipeline = dt.now()

print('nlp time:', end_nlp-start_nlp)
print('pipeline time:', end_pipeline-start_pipeline)


# =============================================================================
# run the preprocessing
# =============================================================================

# import spacy
# nlp = spacy.load('en_core_web_sm')
# t = (u"India Australia Brazil")
# li = nlp(t)
# for i in li:
#     print(i.text)


# list_of_strings  = [i.text for i in li]

