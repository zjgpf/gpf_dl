import os
import random
import numpy as np
import pandas as pd
import torch
from torch import tensor
import pdb
import jieba

def sd2ud(labels):
    '''
    skewd_distribution_categories_uniform_distribution
    '''
    categories_set = np.unique(labels)
    category_indices = []
    for category in categories_set:
        category_indices+=[[category, np.where(labels == category)[0]]]
    min_category,min_category_indices = min(category_indices, key = lambda x:len(x[1]))
    min_num = len(min_category_indices)
    print(f"min category: {min_category}, num: {min_num}")
    new_indices = np.array([])
    for item in category_indices:
        category,indices,origin_len = item[0],item[1],len(item[1])
        mask = np.random.binomial(n=1,p=min_num/origin_len,size=origin_len)==1
        indices = indices[mask]
        print(f'category: {category}, origin_len: {origin_len}, new_len: {len(indices)}')
        new_indices = np.hstack([new_indices,indices])
    return new_indices

def transform_x(x):
    '''
    '[1,2,3]' to [1,2,3]
    '''
    x = [int(v) for v in x[1:-1].split(',')]
    return x

def shuffle(*args):
    if len(args) < 2:
        raise Exception('args should be greater than 2')
    item = np.stack(args, axis = 1)
    np.random.shuffle(item)
    return zip(*item)

def padding(X):
    max_len = len(max(X, key = lambda x: len(x)))
    for x in X:
        x+= [0]*(max_len-len(x))
    return X

def index_labels(labels,label_index):
    return np.array([label_index[v] for v in labels])

class DataSet:
    def __init__(self, *args, **kwargs):
        self.trn_ds = None
        self.val_ds = None
        self.test_ds = None
        if 'label_index' in kwargs:
            self.label_index = kwargs[label_index]
            self.index_label = {v:k for k,v in self.label_index.item()}


class TextDataSet(DataSet):
    def make_batches(self, X, y, bs = 64, is_shuffle = True, is_drop_last = True):
        pdb.set_trace()
        if is_shuffle:
            X,y = shuffle(X,y)
        length = len(y)
        num_of_batches = int(length/bs)
        last_batch = not is_drop_last and length%batch_size
        print(f'Making batches... batch size: {bs},num of batchese: {num_of_batches}')
        start,end,= 0,bs

        for i in range(num_of_batches):
            ret_X = X[start:end]
            ret_y = y[start:end]
            padding(ret_X)
            yield tensor(ret_X),tensor(ret_y)
            start+=bs
            end+=bs
        
        if not is_drop_last and length%batch_size:
            yield tensor(X[start:],y[start:])
    
#class ImageDataSet(DataSet):
    #def classfication_from_path(self, path, bs=64, trn_name = 'train', val_name = 'valid', test_name = 'test'):


