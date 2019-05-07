#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = [
    # TODO: Put package requirements here
]


setup(
    name='torchtuples',
    version='0.0.1',
    description="Model fitting for pytorch",
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
        'Programming Language :: Python :: 3.7',
    ],
)
