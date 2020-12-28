#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:07:40 2020

@author: ivan
"""


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

print(Pool)





# =============================================================================
# reference
# =============================================================================
# Functions from fast BOW
# https://medium.com/glose-team/%EF%B8%8F-fast-bag-of-words-using-spacy-and-cython-574c308a9ff3

# We first initialize the hash table where we will store all the (64-bit hash, C char array/pointer) couples. 
cdef PreshMap hashmap = PreshMap(initial_size=1024)


cdef PreshCounter fast_count(list words):
    '''Count the number of occurrences of every word'''

    cdef:
        PreshCounter counter = PreshCounter(initial_size=256)
        bytes word
 
    for word in words:
        # Insert the word into the hash table, and increment the counter with
        # the 64-bit hash
        counter.inc(_insert_in_hashmap(word, len(word)), 1)
 
    return counter


cdef list text2bow(list words):
    '''Build the BoW representation of a list of words'''
    cdef:
        hash_t wordhash
        int i, freq
        list bow

    # First count the number of occurrences of every word
    counter = fast_count(words)

    # Convert the PreshCounter object to a more readable Python list `bow`,
    # for further usage
    bow = []
    for i in range(counter.c_map.length):
        wordhash = counter.c_map.cells[i].key
        if wordhash != 0:
            freq = <count_t>counter.c_map.cells[i].value
            # We use the 64-bit hashes instead of integer ids, which works
            # as well
            bow.append((wordhash, freq))
 
    return bow
  

@cython.final
cdef hash_t _insert_in_hashmap(char* utf8_string, int length):
    cdef hash_t key = hash_utf8(utf8_string, length)
    cdef Utf8Str* value = <Utf8Str*>hashmap.get(key)
    if value is not NULL:
        print(key)
        return key
    value = _allocate(hashmap.mem, <unsigned char*>utf8_string, length)
    hashmap.set(key, value)
    print(key)
    return key



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
 
 
cdef unicode get_unicode(hash_t wordhash):
    utf8str = <Utf8Str*>hashmap.get(wordhash)
    if utf8str is NULL:
        raise KeyError(f'{wordhash} not in hash table')
    else:
        return decode_Utf8Str(utf8str)
        
        