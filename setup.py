from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='POLENE_GPU',
    version='1.0.0',
    description='Library for training neural networks for chemical kinetics simulations with preprocessing on GPU',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),

    author='Baptiste Romain',
    author_email='baptiste.romain@onera.fr',
    url='',
    license='MIT',

    install_requires=[
        'matplotlib~=3.8.4',
        'numpy~=1.26.4',
        'tqdm~=4.66.2',
        'pandas~=2.2.1',
        'cantera~=3.0.1',
        'scikit-learn~=1.5.1',
        'scipy~=1.13.1',
        'joblib~=1.4.2',
        'torch~=2.3.1',
        'pyarrow~=16.1.0',
        'ax-platform~=1.0.0',
        #'json~=2.0.9'
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],

)