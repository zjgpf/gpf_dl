import sys
import os
sys.path.append(os.path.pardir)
import core
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pdb


df = pd.read_csv('dataset_test.csv')
word_index, index_word = core.get_word_index('word_dict.pickle')

def test_sd2ud():
    print(core.sd2ud(df.y1)[:10])
    print(core.sd2ud([1,2,3,1,1,1,1,1,1,1,1,1])[:10])

def test_shuffle():
    shuffle = core.shuffle
    a1 = np.array([0,1,2])
    a2 = np.array([3,4,5])
    a3 = np.array([6,7,8])
    x,y = shuffle(a1,a2)
    print(x,y)
    x,y,z = shuffle(a1,a2,a3)
    print(x,y,z)

def test_padding():
    padding = core.padding
    x = [[1,2,3],[2,3],[1,2,3,4]]
    print(padding(x)) 

def test_make_batches():
    label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
    x = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)
    dl = core.make_batches(x,y)
    for v in dl:
        print(v[0].shape)

def test_make_batches_ud():
    label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
    x = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)
    dl = core.make_batches_ud(x,y)
    for i,v in enumerate(dl):
        print('generator: ',i)
        for a in v:
            print(a[0].shape)

def test_fit():
    vocal_size = len(word_index)
    embedding_dim = 7
    n_hidden = 128
    n_out = 4

    m = model.SimpleGRU(vocal_size,embedding_dim,n_hidden,n_out).to(torch.device('cuda:0'))
    label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
    X = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)

    opt = optim.SGD(m.parameters(), lr=0.3, momentum=0.5)
    loss_fn = F.nll_loss

    core.fit(m, X,y,3, opt, loss_fn)
    

test_fit()
    
