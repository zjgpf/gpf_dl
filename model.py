import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size,self.embedding_dim,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        embs = self.emb(seq)
        gru_out, self.h = self.gru(embs, self.h)

        outp = self.out(self.h) 
        return torch.squeeze(F.log_softmax(outp, dim=-1),0)
    
    def init_hidden(self, batch_size):
        return torch.zeros((1,batch_size,self.n_hidden)).to(self.get_device())
    
    def get_device(self):
        p = next(self.parameters())
        device_type = str(p.device.type)
        device_index = p.device.index
        ret = device_type + ':' + str(device_index) if device_type == 'cuda' else device_type
        return torch.device(ret)

class SimpleNet(nn.Module):
    
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])
        
    def forward(self, seq):
        bs = seq.size()[0]
        X = seq.view(bs,-1)
        for layer in self.layers:
            X = F.relu(layer(X))
        return F.log_softmax(X,dim=-1)
    
    def get_device(self):
        p = next(self.parameters())
        device_type = str(p.device.type)
        device_index = p.device.index
        ret = device_type + ':' + str(device_index) if device_type == 'cuda' else device_type
        return torch.device(ret)  

class ConvNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.convlayers = nn.ModuleList([nn.Conv2d(layers[i],layers[i+1],kernel_size=3,stride=2) for i in range(len(layers)-1)])
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Linear(layers[-1],c)
    
    def forward(self, x):
        for layer in self.convlayers: x = F.relu(layer(x))
        x = self.pool(x)
        x = x.view(x.size()[0],-1)
        return F.log_softmax(self.output(x), dim=-1)
    
    def get_device(self):
        p = next(self.parameters())
        device_type = str(p.device.type)
        device_index = p.device.index
        ret = device_type + ':' + str(device_index) if device_type == 'cuda' else device_type
        return torch.device(ret)
