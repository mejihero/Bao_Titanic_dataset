#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:24:16 2019

@author: luy1
"""

# https://www.scikit-yb.org/en/latest/api/features/rfecv.html
# Recursive Feature Elimination


from sklearn.svm import SVC
from sklearn.datasets import make_classification
from yellowbrick.features import RFECV

X, y = make_classification(n_samples = 1000,
                           n_features = 25,
                           n_informative = 3,
                           n_redundant = 2,
                           n_repeated = 0,
                           n_classes = 8,
                           n_clusters_per_class = 1,
                           random_state = 0)


viz = RFECV(SVC(kernel = 'linear', C = 1))
viz.fit(X, y)
viz.poof()


#Binary classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

#download datasets
###################

import os

from yellowbrick.download import download_all

## The path to the test data sets
FIXTURES  = os.path.join(os.getcwd(), "data")

## Dataset loading mechanisms
datasets = {
    "bikeshare": os.path.join(FIXTURES, "bikeshare", "bikeshare.csv"),
    "concrete": os.path.join(FIXTURES, "concrete", "concrete.csv"),
    "credit": os.path.join(FIXTURES, "credit", "credit.csv"),
    "energy": os.path.join(FIXTURES, "energy", "energy.csv"),
    "game": os.path.join(FIXTURES, "game", "game.csv"),
    "mushroom": os.path.join(FIXTURES, "mushroom", "mushroom.csv"),
    "occupancy": os.path.join(FIXTURES, "occupancy", "occupancy.csv"),
    "spam": os.path.join(FIXTURES, "spam", "spam.csv"),
}


def load_data(name, download=True):
    """
    Loads and wrangles the passed in dataset by name.
    If download is specified, this method will download any missing files.
    """

    # Get the path from the datasets
    path = datasets[name]

    # Check if the data exists, otherwise download or raise
    if not os.path.exists(path):
        if download:
            download_all()
        else:
            raise ValueError((
                "'{}' dataset has not been downloaded, "
                "use the download.py module to fetch datasets"
            ).format(name))


    # Return the data frame
    return pd.read_csv(path)

########################################


df = load_data('credit')
df.head()

target = 'default'
features = [col for col in df.columns if col != target]


X = df[features]
y = df[target]

cv = StratifiedKFold(5)
oz = RFECV(RandomForestClassifier(),
           cv = cv,
           scoring = 'f1_weighted')

oz.fit(X, y)
oz.poof()

#####################################
#GroupKFold
#sklearn.model_selection

from sklearn.model_selection import GroupKFold
import numpy as np

X = np.arange(24).reshape(12, 2)
y = np.array([1,1,2,3,1,2,3,2,2,3,3,1])
groups = np.array([1,2,3,4,5,6,1,2,3,4,5,6])

kf = list(GroupKFold(n_splits = 6).split(X, y, groups))
kf


groups_2 = np.array([1,2,3,4,5,6,1,2,3,4,5,7])
groups_2

kf_2 = list(GroupKFold(n_splits = 4).split(X, y, groups_2))
kf_2


groups_3 = np.array([1,2,3,4,5,6,1,2,3,4,5,3])
groups_3

kf_3 = list(GroupKFold(n_splits = 5).split(X, y, groups_3))
kf_3





