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

        outp = self.out(self.h[-1]) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size):
        return torch.zeros((1,batch_size,self.n_hidden)).to(self.get_device())
    
    def get_device(self):
        p = next(self.parameters())
        device_type = str(p.device.type)
        device_index = p.device.index
        ret = device_type + ':' + str(device_index) if device_type == 'cuda' else device_type
        return torch.device(ret)
