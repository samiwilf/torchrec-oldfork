# ## To install:
# ## Set env variable:
# export OMP_NUM_THREADS=8
# ## Then run:
# pip install .

#
# pip install multi_hot_hasher/
#
#
# build cpp main toy:
# g++ main.cpp -I/home/ubuntu/anaconda3/envs/torchrec/include/ -L/usr/lib/x86_64-linux-gnu -l:libcrypto.so.1.1
#

import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig
import glob

cppList = [fn for fn in glob.glob('*dule.cpp')]
print(cppList)

sfc_module = Extension(
    'multi_hot_hasher', sources = cppList,
    include_dirs=["/home/ubuntu/anaconda3/envs/torchrec/include", "/home/ubuntu/anaconda3/envs/torchrec/lib/python3.8/site-packages/torch/include"],
	libraries = [':libcrypto.so.1.1'],
	library_dirs = ['/usr/lib/x86_64-linux-gnu'],
    language='c++',
    extra_compile_args = ['-Ofast', '-fopenmp'],
    )

setup(
    name = 'multi_hot_hasher',
    version = '1.0',
    description = 'Python package with for sha hashing 1-hot vectors to N-hot vectors',
    ext_modules = [sfc_module],
)