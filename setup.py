"""
Setup
-----

This compiles all the Fortran extensions.
"""

from setuptools import setup, find_packages

setup(name='ndvi',
      version='2.0',
      description='Geoscience Australia - NDVI for AGDC',
      long_description=open('README.md', 'r').read(),
      license='Apache License 2.0',
      url='https://github.com/GeoscienceAustralia/ndvi',
      author='AGDC Collaboration',
      maintainer='AGDC Collaboration',
      maintainer_email='',
      packages=find_packages(),
      install_requires=[
          'datacube',
      ],
      entry_points={
          'console_scripts': [
              'datacube-ndvi = ndvi.ndvi_app:ndvi_app',
          ]
      })
