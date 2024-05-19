from Cython.Build import cythonize
from distutils.core import setup

setup(name='Hello World app',
     ext_modules=cythonize('feedback.pyx'))