#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:59:42 2020

@author: ivan
"""

cpdef spacy_tokenize(list sentences):
    cdef str i
    for i in sentences:
        print(i)
    