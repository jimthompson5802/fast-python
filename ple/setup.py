from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("ple_transformer_cython.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)