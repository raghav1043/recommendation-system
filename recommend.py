# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 14:33:09 2019

@author: bhutani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

user_col = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('user.csv',names=user_col)

item_col = ['item id','brand', 'item name', 'ram size', 'rom size' ,'price','url'] 
item = pd.read_csv('item.csv',names=item_col)

rating_col = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('rating.csv',names=rating_col)

from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(ratings, test_size = 0.20, random_state=0)

item['model'] = item['brand'].astype(str) + item['item name']

print(users.shape)
users.head()

print(item.shape)
item.head()

print(ratings.shape)
ratings.head()

plt.hist(ratings['rating'])
plt.show()


no_users = ratings.user_id.unique().shape[0]
no_items = ratings.item_id.unique().shape[0]

matrix = np.zeros((no_users, no_items))
for coln in ratings.itertuples():
    r=int(coln[1])
    c=int (coln[2])
    matrix[r-1, c-1] = int (coln[3])
    
from sklearn.metrics.pairwise import pairwise_distances 
user_sim = pairwise_distances(matrix, metric='cosine')
item_sim = pairwise_distances(matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(matrix, user_sim, type='user')
item_prediction = predict(matrix, item_sim, type='item')

def getResultUser(matrix,user_prediction,user_id):
    pred = user_prediction
    for i in range(0,3):
        max=-1;
        idx=-1;
        for x in range(0,len(matrix[0])-1):
            if matrix[user_id-1,x]==0:
                if pred[user_id-1,x]>max:
                    max=pred[user_id-1,x]
                    idx=x
        print(item.iat[idx,7])
        pred[user_id-1,idx]=0
        
def getResultItem(matrix,item_prediction,user_id):
    pred = item_prediction
    for i in range(0,3):
        max=-1;
        idx=-1;
        for x in range(0,len(matrix[0])-1):
            if matrix[user_id-1,x]==0:
                if pred[user_id-1,x]>max:
                    max=pred[user_id-1,x]
                    idx=x
        print(item.iat[idx,7])
        pred[user_id-1,idx]=0

def printResult():
    user_id = int(input("enter your assigned user id"))
    if user_id >len(matrix[0]):
        print("incorrect user id")
    else :
        print("top 3 recommendations according to user-based collaborative filtering method")
        getResultUser(matrix,user_prediction,user_id)
        print()
        print("top 3 recommendations according to item-based collaborative filtering method")
        getResultItem(matrix,item_prediction,user_id)

printResult()

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction,matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction,matrix)))



