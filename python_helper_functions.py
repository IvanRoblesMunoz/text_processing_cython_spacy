#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:57:38 2020

@author: ivan
"""
# =============================================================================
# libraries
# =============================================================================
import re

# =============================================================================
# functions
# =============================================================================
def expand_contractions(phrase):
    '''
    Takes a string of sentence and expandes the contractions 
    
    adding expand contractions durning reading makes it~10x slower but this is
    a really fast step, compared to adding it as a spacy nlp step it is
    10x faster
    Might be worth checking FlashText (claims massive improvements at least for a lot
    of cases but only when number of keywords> 500) only works on keywords not regex
    https://dev.to/vi3k6i5/regex-was-taking-5-days-to-run-so-i-built-a-tool-that-did-it-in-15-minutes-c98
    
    Parameters
    ----------
    phrase : str
        string we want to expand contractions for.

    Returns
    -------
    phrase : str
        string with expanded contractions.

    '''
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # appostrophy misspelss (these are quite slow as each one needs to be don
    # one by one)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"cannot", "can not", phrase)
    phrase = re.sub(r"doesnt", "does not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"hasnt", "has not", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"lets", "let us", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def combine_re(regex_list):
    """
    Concatenates all the elements in the input list using and or opperator and
    produces a list for both prefixes and infixess

    Parameters
    ----------
    regex_list : list[str]
        list of strings we want to concatenate

    Returns
    -------
    regex_infix : str
        infix regex expresion concatenating the inputs using an or operator
    regex_prefix : str
        DESCRIPTION.
        prefix regex expresion concatenating the inputs using an or operator

    """
    regex_infix = "|".join(regex_list)
    regex_prefix = "^" + "|^".join(regex_list)
    return regex_infix, regex_prefix
