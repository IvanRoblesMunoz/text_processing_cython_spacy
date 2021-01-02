#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:43:00 2020

@author: ivan
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "cython_preprocessing",
        sources=["cython_preprocessing.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
        language="c++",
    )
]

setup(
    name="cython_preprocessing",
    
    ext_modules=cythonize(extensions,
                          annotate = True),
)
