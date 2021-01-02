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
import re

import python_helper_functions as ph

# =============================================================================
# Paths
# =============================================================================
data_path = Path("/home/ivan/Downloads/All_Amazon_Review.json.gz")

# =============================================================================
# import data
# =============================================================================
# --- read text data ----

# adding expand contractions at this step makes reading ~10x slower but this is
# a really fast step, compared to adding it as a spacy nlp step it is
# 10x faster
# Might be worth checking FlashText (claims massive improvements at least for a lot
# of cases but only when number of keywords> 500) only works on keywords not regex
# https://dev.to/vi3k6i5/regex-was-taking-5-days-to-run-so-i-built-a-tool-that-did-it-in-15-minutes-c98

# Test expand_contractions function
# ph.expand_contractions("I can't wait to go! isn't aren't couldn't doesn't doesnt cannot")
# Adding the lower() step here makes nlp() step faster



import json
import gzip

def read_corpus(filename, rows=100000):
    data = []
    scores = []
    counter = 0
    
    for line in gzip.open(filename, 'rb'):
        one_line = json.loads(line)
        try:
            text = one_line['reviewText']
            score = one_line['overall']
            
            text = text.strip().lower()
            text = ph.expand_contractions(text)
            
            data.append(text)
            scores.append(score)
            counter+=1
        except KeyError:
            pass
        
        if counter>=rows:
            break
    return data, scores



# import data
start_read = dt.now()
sentences, scores = read_corpus(filename = data_path )
end_read = dt.now()


# --- Load spacy English ---
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load(
    "en_core_web_sm",
    disable=["tagger", "parser", "ner", "textcat"],  # tagger takes 4x longer
)


# --- Define custom tokenizer ----
# We want to modify the tokenizer so that it also splits on the values
# in regex_list (for  "\'" to make sense, contractions must have been expanded)
# note: we need to use escape characters
# To do: figure out why I need to specify "\(" twice for it to work
# note2: adding all these rules makes it faster?
regex_list = [
    "\|",
    '"',
    "'",
    "\?",
    "!",
    "\\",
    "\(",
    "\)",
    "\(",
    "\{",
    "\}",
    "\[",
    "\]",
    "_",
    "-",
    "`",
    ">",
    "<",
    # comas that arent surrounded by numbers
    ",(?![0-9])(?<![0-9])",
    "(?<=^),",
]

regex_infix, regex_prefix = ph.combine_re(regex_list)

infix_re = re.compile(regex_infix, re.MULTILINE)
prefix_re = re.compile(regex_prefix, re.MULTILINE)


nlp.tokenizer = Tokenizer(
    nlp.vocab, infix_finditer=infix_re.finditer, prefix_search=prefix_re.search
)


# Test that our regex infix and prefixes works
# test_sentence = (
#     "\"double  space is| |a|te \"quoted1\"'let 'quoted2' isn't don't can't isn't"
# )
# test_sentence = "(let) us? see !hhh!oooo!www! it (separ(a)tes) [th[e]se] {or{the}se} end"
# test_sentence = ",100,uio ,100 this,should, be ,separated 200, a, but 200,000 shouldnt 200,"
# test_sentence = "<lets see how <brac<ket?s> are separated>"

# print(test_sentence)
# test_sentence = ph.expand_contractions(test_sentence)
# test_sentence = nlp(test_sentence)
# for i in test_sentence:
#     print(i)


# =============================================================================
# generate spacy docs and run cython preprocessing steps
# =============================================================================

import spacy_tokenize as st
st.call_reset_global_variables()

start_nlp = dt.now()
sentences = [doc for doc in nlp.pipe(sentences)]
end_nlp = dt.now()

start_pipeline = dt.now()
res_overall, res_sentence = st.count_words(sentences)
end_pipeline = dt.now()

res_overall = pd.DataFrame(res_overall, columns=["word", "frequency"])
res_sentence = pd.DataFrame(res_sentence, columns=["word", "in_sentence"])
results = pd.merge(res_overall, res_sentence, on=["word"])

# import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords = [i for i in stopwords if "'" not in i]

start_generate_remove_hash = dt.now()
# x2 faster than using python although this is a very fast step anyways
st.call_word_remove_hashmap(
    min_count=50,
    max_doc=int(len(sentences)*0.70),
    other_words=stopwords,
)
end_generate_remove_hash = dt.now()

start_removed_words = dt.now()
rem_words = st.call_remove_words()
end_removed_words = dt.now()


rem_words_hash = st.get_remove_words()
rem_words_hash = [i.decode("utf-8") for i in rem_words_hash]
rem_words_hash = pd.DataFrame(data = rem_words_hash,
                              columns = ['remove_words'])

results = rem_words_hash.merge(results, left_on=['remove_words'],
                                right_on=['word'], how = 'outer')


final_words = st.get_removed_sentences()


print("read time:", end_read - start_read)
print("nlp time:", end_nlp - start_nlp)
print("pipeline time:", end_pipeline - start_pipeline)
print("remove_hash time:", end_generate_remove_hash - start_generate_remove_hash)
print("removed_words time:", end_removed_words - start_removed_words)


# =============================================================================
# generate word embedings
# =============================================================================

# import multiprocessing
# from gensim.models import Word2Vec


# cores = multiprocessing.cpu_count()
# w2v_model = Word2Vec(min_count = 0,
#                      window=2,
#                      size=300,
#                      sample=6e-5, 
#                      alpha=0.03, 
#                      min_alpha=0.0007, 
#                      negative=20,
#                      workers=cores-1)


# make_embedings = [[word.decode('utf8') for word in sentence] for sentence in final_words]

# w2v_model.build_vocab(final_words, progress_per=10000)
