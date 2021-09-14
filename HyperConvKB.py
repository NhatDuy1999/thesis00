import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from Model import Model
from numpy.random import RandomState
import sys

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class HyperConvKB(Model):

    def __init__(self, config):
        super(HyperConvKB, self).__init__(config)

        self.hidden_size = self.config.hidden_size
        self.fil_size = self.config.kernel_size
        self.dropout_rate =  self.config.HyperConvKB_drop_prob
        self.out_channels = self.config.out_channels

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size) 
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.bn = nn.BatchNorm2d(self.out_channels)

        self.fc_r = nn.Linear(self.hidden_size, self.out_channels*3*self.fil_size)

        l = self.config.hidden_size*2

        self.fc_1 = nn.Linear((self.hidden_size-self.fil_size + 1)*self.out_channels, l)
        self.fc_r2 = nn.Linear(self.hidden_size, l)

        self.in_drop = nn.Dropout(0.2)
        self.w_drop = nn.Dropout(0.4)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.elu = nn.ELU()

        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            
            nn.init.xavier_normal_(self.ent_embeddings.weight.data)
            nn.init.xavier_normal_(self.rel_embeddings.weight.data)

        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs

        nn.init.xavier_normal_(self.fc_r.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal_(self.fc_r2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc_1.weight.data, gain=nn.init.calculate_gain('relu'))

    def _calc(self, h, r, t):

        bs = h.size(0)

        I = r
        
        h = h.unsqueeze(1)
        t = t.unsqueeze(1)
        r_ = r.unsqueeze(1)

        E = torch.cat([h, r_, t], dim=1)
        E = E.transpose(1, 2)
        E = E.unsqueeze(0)

        R = self.fc_r(I)
        R = self.tanh(R)
        R = R.view(bs*self.out_channels, 1, self.fil_size, 3)

        X = F.conv2d(E, R, groups=bs)
        X = self.relu(X)
        X = X.view(bs, self.out_channels, -1, 1)
        X = self.bn(X)

        X = X.view(bs, -1)
      
        X = self.dropout(X)


        X = self.fc_1(X)
        X = self.elu(X)
       
        X = X.unsqueeze(-1)
        
        W = self.fc_r2(I)
        W = self.elu(W)
        
        W = W.unsqueeze(1)


        score = torch.matmul(W, X).view(-1)
        
        return score 

    def loss(self, score):
        sc = self.criterion(score * self.batch_y) 
        return torch.mean(sc) 

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)


        loss = self.loss(score)

        return loss

    def predict(self):

        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)
        score = score.cpu().data.numpy()

        sc = np.copy(score)
        sc = np.argsort(sc)
        sm = np.arange(0, len(sc), 1)
        score[sc] = score[sc] + sm
        
        return score
