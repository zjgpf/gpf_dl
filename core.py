import os
import random
import numpy as np
import pickle
import pandas as pd
import copy
import torch
from torch import tensor
import pdb
from tqdm import tqdm

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
    return new_indices.astype(int)

def str2arr(x):
    '''
    '[1,2,3]' to [1,2,3]
    '''
    return [int(v) for v in x[1:-1].split(',')]

def shuffle(*args):
    if len(args) < 2:
        raise Exception('args should be greater than 2')
    item = np.stack(args, axis = 1)
    np.random.shuffle(item)
    return zip(*item)

def padding(X):
    clone = copy.deepcopy(X)
    max_len = len(max(clone, key = lambda x: len(x)))
    for x in clone:
        x+= [0]*(max_len-len(x))
    return clone

def index_labels(labels,label_index):
    return np.array([label_index[v] for v in labels])

def make_batches(X, y, bs = 512, is_shuffle = True, drop_last = False):
    if is_shuffle:
        X,y = shuffle(X,y)
    last_batch = not drop_last and bool(len(y)%bs)
    num_of_batches = int(len(y)/bs)
    print(f'Making batches... batch size: {bs},num of batchese: {num_of_batches+1 if last_batch else num_of_batches}')
    start,end,= 0,bs

    for i in range(num_of_batches):
        ret_X = X[start:end]
        ret_y = y[start:end]
        ret_X = padding(ret_X)
        yield tensor(ret_X),tensor(ret_y)
        start+=bs
        end+=bs
    
    if last_batch:
        ret_X = X[start:]
        ret_y = y[start:]
        ret_X = padding(ret_X)
        yield tensor(ret_X),tensor(ret_y)

def make_batches_ud(X,y,bs=512,drop_last=False,epochs=3):
    for _ in range(epochs):
        mask = sd2ud(y)
        X_local,y_local = X[mask],y[mask]
        yield make_batches(X_local,y_local,bs=bs,drop_last=drop_last)

def get_word_index(word_dict_path):
    word_dict = {}
    word_dict_file = word_dict_path
    with open(word_dict_file,'rb') as f:
        word_dict = pickle.load(f)
    word_arr = [(k,v) for k,v in word_dict.items()]
    word_arr = sorted(word_arr,key = lambda x:x[1],reverse = True)
    word_occur_greater_5 = [(v[0],v[1]) for v in word_arr if v[1]>=5]

    word_index = {v[0]:i for i,v in enumerate(word_occur_greater_5)}
    index_word = {v:k for k,v in word_index.items()}
    return word_index,index_word


def save_model(m,path = 'tmp_torch_model.torch'):
    print(f'Saving model to {os.path.realpath(path)}')
    torch.save(m,path)

def load_model(path = 'tmp_torch_model.torch'):
    print(f'loading model from {os.path.realpath(path)}')
    return torch.load(path)

def fit(m,X,y, epochs, opt, loss_fn,uds=3):
    m.train()
    device = m.get_device()
    print(f'model is been trained on {device}')
    for epoch in tqdm(range(epochs)):
        data = make_batches_ud(X,y,epochs = uds)
        for i,gen in enumerate(data):
            total_loss = 0
            for batch_X, batch_y in gen:
                opt.zero_grad()
                predict = m(batch_X.transpose(0,1).to(device))
                loss = loss_fn(predict, batch_y.to(device))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch} Generator: {i}, total loss: {total_loss}')
    save_model(m)

def predict_batch(m,X, bs = 512 ):
    X += [X[-1]]*(bs - len(X)%bs)
    num_of_batches = len(X)//bs

    start,end,device,pred = 0,bs,m.get_device(),[]

    for _ in range(num_of_batches):
        batch = padding(X[start:end])
        batch = tensor(batch, requires_grad=False).to(device)
        predict =  m(batch.transpose(0,1))
        predict = torch.argmax(predict, dim=-1)
        pred.extend(predict.tolist())
        start+=bs
        end +=bs

    pred = pred[:-offset]
    return pred
