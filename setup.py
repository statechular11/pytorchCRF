import os
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(ROOT_DIR, 'README.md')) as file:
    readme = file.read()

setup(
    name='pytorchcrf',
    version='1.2.0',
    description='PyTorch CRF with N-best decoding',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/statech/pytorchCRF',
    author='Feiyang Niu',
    author_email='statech.forums@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'torch'
    ],
    keywords='pytorch crf n-best',
    packages=find_packages(),
    python_requires='>=3.6',
)
