from setuptools import setup, find_packages

setup(name='valpy',
      version='1.0.0',
      packages=find_packages(),
      install_requires = ['numpy',
                          'rudolfpy @ git+https://github.gatech.edu/SSOG/python-filter.git@development'],
    )