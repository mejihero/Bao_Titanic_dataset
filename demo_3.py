#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:40:01 2019

@author: luy1
"""
import os

os.chdir('/home/luy1/Desktop/pipe projects/datasets/titanic')
print(os.getcwd())


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_cat.csv')
test = pd.read_csv('test_cat.csv')

pd.set_option('display.max_columns', 12)
train.head(5)

train = train.iloc[:, 1:]
train.head(5)

def encode_f(y, encoding):
    return y.map(encoding)


train.dtypes
test.head(5)

train.drop(['PassengerId'], axis =1, inplace = True)
train.head(5)

test.drop(['Unnamed: 0', 'PassengerId'], axis = 1, inplace = True)
test.head(5)

train.info()
test.info()

#input variables: Pclass, Sex, Age, Ticket, Fare, Cabin, Embarked, Title, FamilySize
#output variable: Survived

train['Survived'].value_counts()
train['Sex'].value_counts()
train['Ticket'].value_counts()

np.unique(train['Ticket'])
train['Cabin'].value_counts()
train['Embarked'].value_counts()
train['Title'].value_counts()
train['FamilySize'].value_counts()


#0: numeric encoding
df = train.copy()
df.head()

X = df.loc[:, ['Pclass', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize']]
y = df.loc[:, 'Survived']

X.head(5)
X.info()


#categorical features: *Pclass, Sex, Ticket, Cabin, Embarked, Title

X['Sex'] = X['Sex'].astype('category')
X['Ticket'] = X['Ticket'].astype('category')
X['Cabin'] = X['Cabin'].astype('category')
X['Embarked'] = X['Embarked'].astype('category')
X['Title'] = X['Title'].astype('category')


#train test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 2019)

print(X_train.shape)
print(X_valid.shape)
print(len(y_train))
print(len(y_valid))


import lightgbm as lgb

train_dmx = lgb.Dataset(data = X_train, label = y_train)
valid_dmx = lgb.Dataset(data = X_valid, label = y_valid)

params_ = {'num_leaves': 31,
           'num_trees': 100,
           'objective': 'binary',
           'metric': 'auc'}


mod = lgb.cv()




