#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:59:42 2020

@author: ivan
"""

# =============================================================================
# Import libraries
# =============================================================================
# cython: infer_types=True
cimport cython

# Cython memory pool for memory management
from cymem.cymem cimport Pool

# C string management
from libc.string cimport memcpy           # C string copy function
from libc.stdint cimport uint32_t         # C unsigned 32-bit integer type

# Cython Hash table and fast counter
from preshed.maps cimport PreshMap        # Hash table
from preshed.counter cimport count_t      # Count type (equivalent to C int64_t)
from preshed.counter cimport PreshCounter # Fast counter

# spaCy C functions and types
from spacy.strings cimport hash_utf8      # Hash function (using MurmurHash2)
from spacy.typedefs cimport hash_t        # Hash type (equivalent to C uint64_t), fixed width integer 
from spacy.strings cimport Utf8Str        # C char array/pointer
from spacy.strings cimport decode_Utf8Str # C char array/pointer to Python string function

from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport hash_t
from spacy.structs cimport TokenC

# --- 100x faster tutorial ---
# https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced
import numpy as np # Sometime we have a fail to import numpy compilation error if we don't import numpy
cimport numpy as np

from datetime import datetime as dt
from libcpp cimport bool


# =============================================================================
# Define global variables and objects
# =============================================================================


cdef PreshMap hashmap_all_words = PreshMap(initial_size=1024)
cdef PreshMap hashmap_words_to_remove = PreshMap(initial_size=1024)

cdef PreshCounter counter_overall_word = PreshCounter(initial_size=256)
cdef PreshCounter counter_word_in_sentences = PreshCounter(initial_size=256)

cdef list byte_sentences = []
cdef list removed_word_sentences = []

# =============================================================================
# work
# =============================================================================

def call_reset_global_variables():
    reset_global_variables()
    
cdef void reset_global_variables():
    global hashmap_all_words
    global hashmap_words_to_remove
    global counter_overall_word
    global counter_word_in_sentences
    global byte_sentences
    global removed_word_sentences
    
    # del hashmap_all_words
    hashmap_all_words = PreshMap(initial_size=1024)
    hashmap_words_to_remove = PreshMap(initial_size=1024)
    counter_overall_word = PreshCounter(initial_size=256)
    counter_word_in_sentences = PreshCounter(initial_size=256)
    byte_sentences = []
    removed_word_sentences = []

    
# =============================================================================
# Completed functions
# =============================================================================

def call_remove_words():
    remove_words()
    return removed_word_sentences

cdef void remove_words():
    cdef:
        bytes word
        list byte_sentence, utf8_sentence
        int length
        hash_t key
    
    for byte_sentence in byte_sentences:
        utf8_sentence = []
        for word in byte_sentence:
            length = len(word)
            utf8_word = hash_utf8(word, length)
            value = <Utf8Str*>hashmap_words_to_remove.get(key)
        
            if value is NULL:
                removed_word_sentences.append(word)
                
        removed_word_sentences.append(utf8_sentence)
  

def call_word_remove_hashmap(
        min_count = 100,
        max_doc = 90000,
        other_words = []
        ):
    '''
    Calls a function that generates a hashmap with all the words that we want
    to remove.

    Parameters
    ----------
    min_count : int
        minimum times the word must appear in total
    max_doc : int
        maximum number of documents the word can appear in to keep it
    other_words : list[str]
        list of other words we want to remove
    '''
                             
    generate_word_remove_hashmap(
        min_count = min_count,
        max_doc = max_doc,
        other_words = other_words
        )
    
    
cdef void generate_word_remove_hashmap(int min_count,
                                       int max_doc,
                                       list other_words = []):
    ''' 
    generates a hashmap of words we want to remove based on 
    the arguments passed
    '''
    cdef:
        hash_t key_word_remove
        int freq_overall, freq_sentence, length
        str other_word_remove
        Pool mem = hashmap_words_to_remove.mem
        Utf8Str* value
        unicode unicode_word
        bytes byte_word
        bool value_in

    # --- insert other_words that we want to remove ----
    for other_word_remove in other_words:
        length = len(other_word_remove)
        byte_word = bytes(other_word_remove.lower(), "utf-8")
        key_word_remove = hash_utf8(byte_word, length)
        value = <Utf8Str*>hashmap_words_to_remove.get(key_word_remove)
        
        if value is NULL:
            value = _allocate(mem, byte_word, length)
            hashmap_words_to_remove.set(key_word_remove, value)
    
    # --- check if any of word is outside the bounds specified and insert ---
    
    for key_word_remove in hashmap_all_words.keys():
        insert_value = True

        freq_overall = counter_overall_word[key_word_remove]
        if freq_overall <= min_count:
            
            value = <Utf8Str*>hashmap_words_to_remove.get(key_word_remove)
            
            if value is NULL:
                value = <Utf8Str*>hashmap_all_words.get(key_word_remove)
                hashmap_words_to_remove.set(key_word_remove, value)
                insert_value = False
                
                
        elif insert_value:
            freq_sentence = counter_word_in_sentences[key_word_remove]
            if freq_sentence >= max_doc:
                value = <Utf8Str*>hashmap_words_to_remove.get(key_word_remove)
    
                if value is NULL:
                    value = <Utf8Str*>hashmap_all_words.get(key_word_remove)         
                    hashmap_words_to_remove.set(key_word_remove, value)


# --- Python functions ----------
def count_words(list sentences ):
    '''
    This function takes a list of sentences and counts the number of times
    each word appears and the number sentneces in which that word appears

    Parameters
    ----------
    sentences list[Doc] : 
        The sentences that we want to count the words for, containing spacy Doc
        objects

    Returns
    -------
    results_overall : list[list[str, int]]
        list of lists containing a list per word and the total count of times
        the word appears
    results_sentence : list[list[str, int]]
        list of lists containing a list per word and the count of documents
        the word appears in
    '''
    cdef:
        list byte_sentence,  results_overall, results_sentence, utf8_sentence
        # TokenC word
        Doc words
    
    # --- reset global variables ---
    reset_global_variables()
    
    # --- convert python to cython ----
    start_convert = dt.now()
    # byte_sentences = []
    for words in sentences:
        byte_sentence = [bytes(word.text,'utf-8') for word in words]
        utf8_sentence = []
        byte_sentences.append(byte_sentence)
    end_convert = dt.now()   

    # --- generate hashmap and counter ----
    start_insert = dt.now()
    iterate_through_words(byte_sentences) 
    end_insert = dt.now()
    
    # --- read counter (all word occurences)---
    start_read_count = dt.now()
    results_overall = preshcount_to_list(counter_overall_word)
    end_read_count = dt.now()
    
    # --- read counter (occurences in sentences) ---
    start_count_sent_read = dt.now()
    results_sentence = preshcount_to_list(counter_word_in_sentences)
    end_count_sent_read= dt.now()
    
    print('convert time: ',end_convert - start_convert)
    print('insert time: ',end_insert - start_insert)
    print('read overall_word counter time: ',end_read_count - start_read_count)    
    print('read in_sentences counter time: ',end_count_sent_read - start_count_sent_read)    
    
    return results_overall, results_sentence


def get_byte_sentences():
    '''
    Returns the byte_sentence list

    Returns
    -------
    byte_sentences : list[list[bytes]]
        byte_sentence list containing the sentences in byte for
    '''
    return byte_sentences


# --- cython functions -----
cdef list preshcount_to_list(PreshCounter counter):
    '''
    This function takes a counter and the global object hashmap_all_words and 
    returns 
    Parameters
    ----------
    PreshCounter counter : PreshCounter
        DESCRIPTION.
        
    Returns
    -------
    results : list 
        list of lists containing [unicode word, number of times in the text]
    '''
    cdef:
        list results
        int i, freq
        hash_t wordhash

    results = []
    for i in range(counter.c_map.length):
        wordhash = counter.c_map.cells[i].key
        if wordhash != 0:
            freq = <count_t>counter.c_map.cells[i].value
            # returning the acctual words takes ~ 2x longer than returning utf8 keys
            results.append([get_unicode(wordhash, hashmap_all_words),freq])
        
    return results            


cdef void iterate_through_words(list byte_sentences):
    ''' This function iterates through the words and generates the counters 
       for how many times a word appears overall, and in how many sentences
       it appears, as well as a hashmap of all available words.
    '''
    cdef:
        list byte_sentence
        int length
        bytes word
        hash_t key_overall, key_sentence
        PreshMap words_in_sentence 
        # long value_sentence
        
    words_in_sentence = PreshMap(initial_size=1024)
        
    for byte_sentence in byte_sentences:
        words_in_sentence = PreshMap(initial_size=1024)
        for word in byte_sentence:
            # --- insert in overall hashmap ---
            key_overall = insert_in_hashmap( word, hashmap_all_words)
            # --- insert in overall word count --
            counter_overall_word.inc(key_overall,1)
            # --- insert in sentence hashmap ---
            key_overall = insert_in_hashmap( word, words_in_sentence)
        
        for key_sentence in words_in_sentence.keys():
            counter_word_in_sentences.inc(key_sentence,1)
        # --- dealocate memory ---
        # To do: check if this acctually deletes anything (seems to work)
        # at least for python lists
        # import psutil
        # for _ in range(10):
        #     simple_test(10)
        #     print(psutil.virtual_memory().percent)
        del words_in_sentence
      

  
def get_remove_words():        
    res_list = hashmap_to_list(hashmap_words_to_remove)
    return res_list
    
cdef list hashmap_to_list(PreshMap hashmap):
    ''' converts hashmap keys to list as a byte '''
    cdef list res_list = []
    cdef hash_t key
    cdef unicode word
    cdef bytes byte_word
    cdef Utf8Str* value
    
    for key in hashmap.keys():
            value = <Utf8Str*>hashmap.get(key)
            if value is not NULL:
                word = get_unicode(key, hashmap)
                byte_word = word.encode('utf-8')
                res_list.append(byte_word)
            
    return res_list
        
@cython.final
cdef hash_t insert_in_hashmap(bytes word, PreshMap hashmap ):
    '''
    This function takes a string of bytes, hashes it into fixed width
    integer and then inserts it into the hashmap while alocating memory 
    for it

    Parameters
    ----------
    word : bytes
        utf-8 encoded bytes representing a python string

    Returns
    -------
    key : hash_t
        hased word into fixed intiger code
    '''
    cdef:
        int length 
        hash_t key
        Utf8Str* value 
        #  Pool deals with memory management 
        Pool mem = hashmap.mem

    length = len(word)
    # our hash table assumes prehashed key, so we hash it so to get a utf8 code 
    key = hash_utf8(word, length)
    # here we are defining the pointer for our string 
    value = <Utf8Str*>hashmap.get(key)

    if value is not NULL:
        # print('unicode:1',get_unicode(key, hashmap))
        return key
            
    else:
        value = _allocate(mem, word, length)
        hashmap.set(key, value)
        # print('unicode:2',get_unicode(key, hashmap))
        return key
             

# =============================================================================
# external functions
# =============================================================================
# Functions from fast BOW
# https://medium.com/glose-team/%EF%B8%8F-fast-bag-of-words-using-spacy-and-cython-574c308a9ff3

# To do: figure out how memory allocation acctually works
cdef Utf8Str* _allocate(Pool mem, const unsigned char* chars, uint32_t length) except *:
    cdef:
        int n_length_bytes
        int i
        Utf8Str* string = <Utf8Str*>mem.alloc(1, sizeof(Utf8Str))
        uint32_t ulength = length

    if length < sizeof(string.s):
        string.s[0] = <unsigned char>length
        memcpy(&string.s[1], chars, length)
        return string
    elif length < 255:
        string.p = <unsigned char*>mem.alloc(length + 1, sizeof(unsigned char))
        string.p[0] = length
        memcpy(&string.p[1], chars, length)
        return string
    else:
        i = 0
        n_length_bytes = (length // 255) + 1
        string.p = <unsigned char*>mem.alloc(length + n_length_bytes, sizeof(unsigned char))
        for i in range(n_length_bytes-1):
            string.p[i] = 255
        string.p[n_length_bytes-1] = length % 255
        memcpy(&string.p[n_length_bytes], chars, length)
    return string
 

cdef unicode get_unicode(hash_t wordhash,PreshMap hashmap):
    utf8str = <Utf8Str*>hashmap.get(wordhash)
    if utf8str is NULL:
        raise KeyError(f'{wordhash} not in hash table')
    else:
        return decode_Utf8Str(utf8str)
        