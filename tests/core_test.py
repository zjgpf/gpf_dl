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


df = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')
word_index, index_word = core.get_word_index('word_dict.pickle')
label_index = {'办公费':0,'业务招待费':1,'福利费':2,'差旅费':3}
index_label = {v:k for k,v in label_index.items()}

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
    x = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)
    dl = core.make_batches(x,y)
    for v in dl:
        print(v[0].shape)

def test_make_batches_ud():
    x = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)
    dl = core.make_batches_ud(x,y)
    for i,v in enumerate(dl):
        print('generator: ',i)
        for a in v:
            print(a[0].shape)

def test_fit_text_classification():
    vocal_size = len(word_index)
    embedding_dim = 7
    n_hidden = 128
    n_out = 4

    m = model.SimpleGRU(vocal_size,embedding_dim,n_hidden,n_out).to(torch.device('cuda:0'))
    #m=core.load_model()
    m.to(torch.device('cuda:0'))
    X = df.x.apply(core.str2arr)
    y = core.index_labels(df.y1,label_index)

    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    loss_fn = F.nll_loss

    core.fit_text_classification(m, X,y,3, opt, loss_fn)

def test_predict_batch_text_classification():
    m = model.SimpleGRU(vocal_size,embedding_dim,n_hidden,n_out).to(torch.device('cuda:0'))
    m=core.load_model(m)
    X = list(df_test.x.apply(core.str2arr))
    print(len(X))
    pred = core.predict_batch_text_classification(m,X)
    print(len(pred))
    print(pred[:3])
    return pred

def test_evaluation_matrix():
    pred = test_predict_batch_text_classification()
    pred = np.array([index_label[i] for i in pred])
    expect = df_test.y1.values
    core.evaluation_matrix(pred, expect)
    

def test_get_img_path_label_from_path():
    label_index_img_c = {'frog':0,'truck':1,'deer':2,'automobile':3,'bird':4,'horse':5,'ship':6,'cat':7,'airplane':8,'dog':9} 
    ret = core.get_img_path_label_from_path('/data/gpf/tutorial/dl/cnn/cifar/train',label_index_img_c)
    print(len(ret))
    print(ret[:3])
    return ret

def test_make_batches_img():
    imgPath_label = test_get_img_path_label_from_path()
    img_paths, labels = zip(*imgPath_label)
    dl = core.make_batches_img(np.array(img_paths), np.array(labels).astype(int),bs=64,sz=16)
    for v in dl:
        print(v[0].shape)


test_make_batches_img()
