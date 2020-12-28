#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:07:58 2020

@author: ivan
"""



# =============================================================================
# import libraries
# =============================================================================
from pathlib import Path
from datetime import datetime as dt
import pandas as pd

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

# --- Load spacy English ---
import spacy
import re
from spacy.tokenizer import Tokenizer

# --- Define spacy and remove unwanted steps
nlp = spacy.load("en_core_web_sm",
                 disable=["tagger", # tagger takes 4x longer
                          "parser",
                          "ner",
                          "textcat"]
                 )

# --- we want to modify the tokenizer so that it also splits on "|"
# note: we need to use escape characters
infix_re = re.compile(r'''\|''')

def custom_tokenizer(nlp):
  return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

nlp.tokenizer = custom_tokenizer(nlp)



# =============================================================================
# generate spacy docs and run cython preprocessing steps
# =============================================================================

import spacy_tokenize as st
start_nlp = dt.now()
sentences = [doc for doc  in nlp.pipe(sentences)]

end_nlp = dt.now()

start_pipeline = dt.now()
res = st.run_pipeline(sentences)
end_pipeline = dt.now()

print('nlp time:', end_nlp-start_nlp)
print('pipeline time:', end_pipeline-start_pipeline)

res = pd.DataFrame(res,
                   columns = ['word','frequency'])

