from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

setup(
    ext_modules = cythonize(["cythondir/*.pyx"]),
    include_dirs=[numpy.get_include()],
)
