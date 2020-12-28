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

from libcpp.string cimport string         # decode unicode string

# Cython Hash table and fast counter
from preshed.maps cimport PreshMap        # Hash table
from preshed.counter cimport count_t      # Count type (equivalent to C int64_t)
from preshed.counter cimport PreshCounter # Fast counter

# spaCy C functions and types
from spacy.strings cimport hash_utf8      # Hash function (using MurmurHash2)
from spacy.typedefs cimport hash_t        # Hash type (equivalent to C uint64_t) 
                                          # fixed width integer
                                          
from spacy.strings cimport Utf8Str        # C char array/pointer
from spacy.strings cimport decode_Utf8Str # C char array/pointer to Python string function


# --- 100x faster tutorial ---
# https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced
import numpy as np # Sometime we have a fail to import numpy compilation error if we don't import numpy
cimport numpy as np
from cymem.cymem cimport Pool
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport hash_t
from spacy.structs cimport TokenC
from tqdm import tqdm
from datetime import datetime as dt


# --- Print function for cython
from libc.stdio cimport printf

cimport numpy as np
# =============================================================================
# Work
# =============================================================================

# define global variables
cdef PreshMap hashmap_words = PreshMap(initial_size=1024) 
cdef PreshCounter overall_word_count = PreshCounter(initial_size=256)

def run_pipeline(list sentences):
    cdef:
        list byte_sentence, byte_sentences, results
        # TokenC word
        Doc words
    
    # --- convert python to cython ----
    start_convert = dt.now()
    byte_sentences = []
    
    for words in sentences:
        byte_sentence = [bytes(word.text,'utf-8') for word in words]
        byte_sentences.append(byte_sentence)   
    end_convert = dt.now()   

    # --- generate hashmap and counter ----
    start_insert = dt.now()
    iterate_through_words(byte_sentences) 
    end_insert = dt.now()
    
    # --- read counter ---
    start_read_count = dt.now()
    results = preshcount_to_list(overall_word_count)
    end_read_count = dt.now()
    
    print('convert time: ',end_convert - start_convert)
    print('insert time: ',end_insert - start_insert)
    print('insert time: ',end_read_count - start_read_count)    
    
    return results
    
    
            
cdef list preshcount_to_list(PreshCounter counter):
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
            results.append([get_unicode(wordhash, hashmap_words),freq])
        
    return results
        
    
# --- Completedish ----       
cdef hash_t insert_in_hashmap(bytes word):
    '''
    This function takes a string of bytes, hashes it into fixed width
    integer and then inserts it into the global hashmap

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
        Pool mem = hashmap_words.mem

    length = len(word)
    # our hash table assumes prehashed key, so we hash it so to get a utf8 code 
    key = hash_utf8(word, length)
    # here we are defining the pointer for our string 
    value = <Utf8Str*>hashmap_words.get(key)

    if value is not NULL:
        # print('unicode:1',get_unicode(key, hashmap_words))
        return key
            
    else:
        value = _allocate(mem, word, length)
        hashmap_words.set(key, value)
        # print('unicode:2',get_unicode(key, hashmap_words))
        return key
             
     
cdef void iterate_through_words(list byte_sentences):
    ''' This function iterates through the words and generates the counters
    '''
    cdef:
        list byte_sentence
        bytes word
        hash_t key
        
    for byte_sentence in byte_sentences:
        for word in byte_sentence:
            key = insert_in_hashmap( word)
            overall_word_count.inc(key,1)  
            

# =============================================================================
# reference
# =============================================================================
# Functions from fast BOW
# https://medium.com/glose-team/%EF%B8%8F-fast-bag-of-words-using-spacy-and-cython-574c308a9ff3

  

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
        
        