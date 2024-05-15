from setuptools import setup, find_packages

setup(name='valpy',
      version='1.0.0',
      packages=find_packages(),
      install_requires = ['numpy',
                          'rudolfpy @ git+https://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@github.gatech.edu/SSOG/python-filter.git@dev'],
    )