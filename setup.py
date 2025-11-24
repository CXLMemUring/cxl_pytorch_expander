from setuptools import setup, Extension
import os

module = Extension(
    'cxl_memory',
    sources=['src/cxl_memory_module.cpp'],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c++11', '-O2'],
    extra_link_args=[]
)

setup(
    name='cxl_pytorch_expander',
    version='0.1.0',
    description='CXL Memory Expander for PyTorch',
    ext_modules=[module],
    packages=['python'],
    package_dir={'python': 'python'},
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    python_requires='>=3.8',
)
