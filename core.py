import os
import random
import numpy as np
import pickle
import pandas as pd
import copy
import torch
from torch import tensor
import cv2
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
    torch.save(m.state_dict(),path)

def load_model(m,path = 'tmp_torch_model.torch'):
    print(f'loading model from {os.path.realpath(path)}')
    m.load_state_dict(torch.load(path))

def fit_text_classification(m,X,y, epochs, opt, loss_fn,uds=3):
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

def predict_batch_text_classification(m,X, bs = 512 ):
    m.eval()
    offset = bs - len(X)%bs
    X += [X[-1]]*offset
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

    return pred[:-offset]

def evaluation_matrix(predicts, expects):        
    assert(len(predicts)==len(expects))
    expect_set = np.unique(expects)
    correct_all = 0
    for category in expect_set:
        expects_indices = set(np.where(expects == category)[0])
        predicts_indices = set(np.where(predicts == category)[0])
        print(f"---------------------------{category}-------------------------------")
        print(f"total: ", len(expects_indices))
        total_correct = 0
        for i in predicts_indices:
            if i in expects_indices: 
                total_correct+=1
                correct_all+=1
        print("correct: ", total_correct)
        print("accuracy: ", total_correct/len(expects_indices))
    print("---------------------------All-------------------------------")
    print(f'total correct/total:{correct_all}/{len(expects)}')
    print('total accuracy: ', correct_all/len(expects))


def get_img_path_label_from_path(root_path, label_index):
    ret = []
    label_counts = {k:0 for k in label_index}
    for img in os.listdir(root_path):
        if not (img.endswith('.png') or img.endswith('.jpg')): continue
        label = img.split('_')[1][:-4]
        path = os.path.abspath(os.path.join(root_path,img))
        ret += [[path,label_index[label]]]
        label_counts[label]+=1
    for k,v in label_counts.items(): print(k+":"+str(v))
    return ret 

def make_batches_img(img_paths, labels, bs = 32, sz = 32, is_shuffle = True, drop_last = False, stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))):
    if is_shuffle:
        img_paths,labels = shuffle(img_paths, labels)
        labels = np.array(labels).astype(int)
    
    last_batch = not drop_last and bool(len(labels)%bs)
    num_of_batches = int(len(labels)/bs)
    print(f'Making batches... batch size: {bs},num of batchese: {num_of_batches+1 if last_batch else num_of_batches}')
    start,end,= 0,bs

    for i in range(num_of_batches):
        paths = img_paths[start:end]
        imgs = []
        for path in paths:
            imgs += [cv2.resize(cv2.imread(path), (sz,sz))]
        imgs = normalize(np.array(imgs)/255,*stats)
        labels_local = labels[start:end]
        yield tensor(imgs,dtype=torch.float),tensor(labels_local)
        start+=bs
        end+=bs
    
    if last_batch:
        paths = img_paths[start:]
        labels_local = img_paths[start:]
        imgs = []
        for path in paths:
            imgs += [cv2.resize(cv2.imread(path), (sz,sz))]
        imgs = normalize(np.array(imgs)/255,*stats)
        labels_local = labels[start:end]
        yield tensor(imgs,dtype=torch.float),tensor(labels_local)

def normalize(x,m,s):
    return (x-m)/s
    
class dl_img:
   
    def __init__(self, img_paths, labels, bs = 32, sz = 32, is_shuffle = True, drop_last = False,stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))):
        self.img_paths,self.labels,self.bs,self.sz,self.is_shuffle,self.drop_last,self.stats = img_paths,labels,bs,sz,is_shuffle,drop_last,stats
        
    def __iter__(self):
        img_paths,labels,bs,sz,is_shuffle,drop_last,stats = self.img_paths,self.labels,self.bs,self.sz,self.is_shuffle,self.drop_last,self.stats
        if is_shuffle:
            img_paths,labels = shuffle(img_paths, labels)
            labels = np.array(labels).astype(int)
        
        last_batch = not drop_last and bool(len(labels)%bs)
        num_of_batches = int(len(labels)/bs)
        print(f'Making batches... batch size: {bs},num of batchese: {num_of_batches+1 if last_batch else num_of_batches}')
        start,end,= 0,bs

        for i in range(num_of_batches):
            paths = img_paths[start:end]
            imgs = []
            for path in paths:
                imgs += [cv2.resize(cv2.imread(path), (sz,sz))]
            imgs = normalize(np.array(imgs)/255,*stats)
            labels_local = labels[start:end]
            yield tensor(imgs,dtype=torch.float),tensor(labels_local)
            start+=bs
            end+=bs
        
        if last_batch:
            paths = img_paths[start:]
            labels_local = img_paths[start:]
            imgs = []
            for path in paths:
                imgs += [cv2.resize(cv2.imread(path), (sz,sz))]
            imgs = normalize(np.array(imgs)/255,*stats)
            labels_local = labels[start:end]
            yield tensor(imgs,dtype=torch.float),tensor(labels_local)

def fit(m,data, epochs, opt, loss_fn):
    m.train()
    device = m.get_device()
    print(f'model is been trained on {device}')
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_X, batch_y in data:
            opt.zero_grad()
            predict = m(batch_X.to(device))
            loss = loss_fn(predict, batch_y.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch}, total loss: {total_loss}')
    save_model(m)

def predict_batch_img(m,X, bs = 64 ):
    m.eval()
    device,pred = m.get_device(),[]
    for batch in X:
        batch = tensor(batch, requires_grad=False).to(device)
        predict =  m(batch)
        predict = torch.argmax(predict, dim=-1)
        pred.extend(predict.tolist())

    return pred 
