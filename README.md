# Credit Risk Resampling
We will use various techniques to train and evaluate models with imbalanced classes. We will use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Technologies
This project is written in ***Python 3.8*** with the following libraries:

***panda*** -a Python package that provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive.

***numpy*** -is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays. It is the fundamental package for scientific computing with Python.

***sklearn*** -Scikit-Learn is a free machine learning library for Python. It supports both supervised and unsupervised machine learning, providing diverse algorithms for classification, regression, clustering, and dimensionality reduction. The library is built using many libraries you may already be familiar with, such as NumPy and SciPy. It also plays well with other libraries, such as Pandas and Seaborn.

***imbalanced-learn*** -is a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance. It is compatible with scikit-learn and is part of scikit-learn-contrib projects


## Installation Guide

Before running the application first install the following dependencies.

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix

from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')import pandas as pd

## Usage

After cloning the repository,  open the directory Starter Code and run the program by typing python credit_risk_resampling.ipynb

## Contributors
#### James Tagapan

jtagapan@gmail.com

## License
Licensed under the MIT License. Copyright 2020

