from setuptools import setup
import sys, os

version = '0.1'

setup(name='linguini',
      version=version,
      description="linguini is an implementation of the vector-space based language identifier",
      long_description= open("README").read(),
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords=['language detection', 'multilingual documents', 'text classification'],
      author='Marco Lui',
      author_email='saffsd@gmail.com',
      url='https://github.com/saffsd/linguini',
      license='BSD',
      packages=['linguini'],
      package_data={'linguini':['models/*']},
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
          'numpy',
      ],
      entry_points= {
        'console_scripts': [
          'linguini = linguini.cli:main',
        ],
      },
      )
