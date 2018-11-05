import sys
import os
sys.path.append(os.path.pardir)
import dataset as ds
import pandas as pd
import numpy as np
import pdb


df = pd.read_csv('dataset_test.csv')
def test_sd2ud():
    print(ds.sd2ud(df.y1)[:10])
    print(ds.sd2ud([1,2,3,1,1,1,1,1,1,1,1,1])[:10])

def test_shuffle():
    shuffle = ds.shuffle
    a1 = np.array([0,1,2])
    a2 = np.array([3,4,5])
    a3 = np.array([6,7,8])
    x,y = shuffle(a1,a2)
    print(x,y)
    x,y,z = shuffle(a1,a2,a3)
    print(x,y,z)

def test_padding():
    padding = ds.padding
    x = [[1,2,3],[2,3],[1,2,3,4]]
    print(padding(x)) 

def test_make_batches():
    label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
    tds = ds.TextDataSet()
    x = df.x.apply(ds.str2arr)
    y = ds.index_labels(df.y1,label_index)
    dl = tds.make_batches(x,y)
    for v in dl:
        print(v[0].shape)

def test_make_batches_ud():
    label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
    tds = ds.TextDataSet()
    x = df.x.apply(ds.str2arr)
    y = ds.index_labels(df.y1,label_index)
    dl = tds.make_batches_ud(x,y)
    for i,v in enumerate(dl):
        print('generator i:',i)
        for a in v:
            print(a[0].shape)

test_make_batches_ud()
    
