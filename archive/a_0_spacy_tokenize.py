#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:59:42 2020

@author: ivan
"""
# =============================================================================
# Import libraries
# =============================================================================
'''
Among all these imports let us comment on a few:
    
    - The cymem package is used to tie memory to a Python object, so that the 
      memory is freed when the object is garbage collected.
    - The preshed package contains both the Hash table where we store the 
      (64-bit hash, C char array/pointer) couples, and a fast counter extension 
      type (PreshCounter) that we will use to perform the BoW counting.
    - We use cimport instead of import to access the extensions types’ C 
    methods and attributes.
'''

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
from spacy.typedefs cimport hash_t        # Hash type (equivalent to C uint64_t)
from spacy.strings cimport Utf8Str        # C char array/pointer
from spacy.strings cimport decode_Utf8Str # C char array/pointer to Python string function

from cymem.cymem cimport Pool
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport hash_t
from spacy.structs cimport TokenC


# =============================================================================
# work
# =============================================================================

cdef struct DocElement:
    TokenC* c
    int length


def populate_doc(doc_list):
    cdef Pool mem = Pool()
    cdef DocElement* docs = <DocElement*>mem.alloc(n_docs, sizeof(DocElement))
    cdef Doc doc
    for i, doc in enumerate(doc_list): # Populate our database structure
        docs[i].c = doc.c
        docs[i].length = (<Doc>doc).length




# =============================================================================
# Define data types?
# =============================================================================
# --- Define Doc Element struct, this is the datatype for tokens


    


# cdef list fast_loop(DocElement* docs, int n_docs, hash_t word, hash_t tag):
#     cdef int n_out = 0
#     for doc in docs[:n_docs]:
#         for c in doc.c[:doc.length]:
#             if c.lex.lower == word and c.tag == tag:
#                 n_out += 1
#     return n_out

# =============================================================================
# Define functions
# =============================================================================

# cpdef list[bytes] make_docs(list sentences, object nlp):
#     '''
#     !!! This function is not worth it, it is just as fast as using python
#     This function generates a list[Doc] of spacy Doc objectsfrom a list[str].
    
#     It does this by iterating through the list of strings and transforming
#     those strings to Doc object
    
#     Parameters
#     ----------
#     sentences : list[str]
#         The list of strings that will be processed
        
#     nlp : object
#         a spacy nlp language object, this function is used to generate the
#         Doc objects.
        
#     Returns
#     -------
#     sentences: list[Doc]
#         a list of Doc objects
            
#     '''
#     cdef str i
#     sentences = [nlp(i) for i in sentences]
#     return sentences
    


