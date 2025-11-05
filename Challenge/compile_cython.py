#!/usr/bin/env python3
# -*- coding: utf-8 -*-

usage = """
Compile a given Cython source file in a subprocess

Usage:
    python compile_script.py filename.pyx build_ext --inplace
    python compile_script.py directory
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

if len(sys.argv) == 4:
    fileToCompile = sys.argv[1]

    if not fileToCompile.endswith(".pyx"):
        raise ValueError("First argument must be a .pyx file")

    # Remove file argument before invoking setup()
    del sys.argv[1]

    module_path = os.path.splitext(fileToCompile)[0]
    extensionName = module_path.replace(os.sep, ".")

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

elif len(sys.argv) == 2:
    directoryToCompile = sys.argv[1]

    if not os.path.isdir(directoryToCompile):
        raise ValueError("First argument must be a directory")

    # Remove directory argument before invoking setup()
    del sys.argv[1]

    extensions = []
    for root, _, files in os.walk(directoryToCompile):
        for file in files:
            if file.endswith(".pyx"):
                filePath = os.path.join(root, file)
                module_path = os.path.splitext(filePath)[0]
                extensionName = module_path.replace(os.sep, ".")

                extensions.append(
                    Extension(
                        extensionName,
                        [filePath],
                        extra_compile_args=["-O3"],  # Slightly better optimization
                        include_dirs=[np.get_include()],
                    )
                )

    setup(
        name="cython_modules",
        ext_modules=cythonize(
            extensions,
            compiler_directives={"language_level": "3"},
        ),
    )

else:
    raise ValueError(f"Wrong number of parameters." + usage)
