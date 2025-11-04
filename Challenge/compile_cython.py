#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compile a given Cython source file in a subprocess

Usage:
    python compile_script.py filename.pyx build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os


# Expect: script.py file.pyx build_ext --inplace  → 4 args
if len(sys.argv) != 4:
    raise ValueError(f"Wrong number of parameters. Expected 4, got {len(sys.argv)}")

fileToCompile = sys.argv[1]

if not fileToCompile.endswith(".pyx"):
    raise ValueError("First argument must be a .pyx file")

# Remove file argument before invoking setup()
del sys.argv[1]

extensionName = os.path.splitext(fileToCompile)[0]

extensions = [
    Extension(
        extensionName,
        [fileToCompile],
        extra_compile_args=["-O3"],  # Slightly better optimization
        include_dirs=[np.get_include()],
    )
]

setup(
    name=extensionName,
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
