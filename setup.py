from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

setup(
    ext_modules = cythonize("vm.pyx", compiler_directives={
        'language_level': 3,
        'cdivision': True,
        'boundscheck': False,
        'wraparound': False
    }),
    include_dirs=[numpy.get_include()],

)
