# write setup.py
from setuptools import setup, find_packages

setup(
    name='m_package',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)