import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']

ext_modules = [
        Extension(
                'eddl',
                    ['tensor.cpp', 'wrap_tensor.cpp'],
                    include_dirs=['pybind11/include'],
                language='c++',
                extra_compile_args = cpp_args,
                ),
    ]

setup(
        name='eddl',
        version='0.0.1',
        author='RPP',
        ext_modules=ext_modules,
    )

