#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Robert Martin-Short",
    author_email='martinshortr@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="System to classify cloud images based on data scraped from the Cloud Appreciation Society website",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cloud_classifier',
    name='cloud_classifier',
    packages=find_packages(include=['cloud_classifier', 'cloud_classifier.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/rmartinshort/cloud_classifier',
    version='0.1.0',
    zip_safe=False,
)
