from __future__ import print_function

import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
import numpy

# Because many people neglected to run the pylearn2/utils/setup.py script
# separately, we compile the necessary Cython extensions here but because
# Cython is not a strict dependency, we issue a warning when it is not
# available.
try:
    from Cython.Distutils import build_ext
    cython_available = True
except ImportError:
    warnings.warn("Cython was not found and hence pylearn2.utils._window_flip "
                  "and pylearn2.utils._video and classes that depend on them "
                  "(e.g. pylearn2.train_extensions.window_flip) will not be "
                  "available")
    cython_available = False

if cython_available:
    cmdclass = {'build_ext': build_ext}
    ext_modules = [Extension("pylearn2.utils._window_flip",
                             ["pylearn2/utils/_window_flip.pyx"],
                             include_dirs=[numpy.get_include()]),
                   Extension("pylearn2.utils._video",
                             ["pylearn2/utils/_video.pyx"],
                             include_dirs=[numpy.get_include()])]
else:
    cmdclass = {}
    ext_modules = []


setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    name='pylearn2',
    version='0.1dev',
    packages=find_packages(),
    description='A machine learning library built on top of Theano.',
    license='BSD 3-clause license',
    long_description=open('README.rst', 'rb').read().decode('utf8'),
    install_requires=[
        'numpy', 
        'pyyaml', 
        'argparse', 
        'six', 
        "Theano"],
    scripts=['bin/pylearn2-plot-monitor', 'bin/pylearn2-print-monitor',
             'bin/pylearn2-show-examples', 'bin/pylearn2-show-weights',
             'bin/pylearn2-train'],
    package_data={
        '': ['*.cu', '*.cuh', '*.h'],
    },
)
