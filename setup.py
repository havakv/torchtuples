#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = [
    'numpy>=1.15.4',
    'pandas>=0.24.2',
    'matplotlib>=3.0.3',
    # 'torch>=1.1.0',
]


setup(
    name='torchtuples',
    version='0.1.1',
    description="Training neural networks in PyTorch",
    author="Haavard Kvamme",
    author_email='haavard.kvamme@gmail.com',
    url='https://github.com/havakv/torchtuples',
    packages=find_packages(include=['torchtuples']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='torchtuples',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.6'
)
