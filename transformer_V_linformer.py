import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tabulate import tabulate
from transformers import DisilBertTokenizer
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics import accuracy_score



def intialize_weights(module):
    if isinstance(module, nn.linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embeddings):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class FeedFoward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FeedFoward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(initialize_weights)
    
    def foward(self, x):
        return self.net(x)
    
class Multi_Head_Attention(nn.Module):
    def __initi__(seld, d, n_heads, dropout=0.3):
        super(Multi_Head_Attention, self).__init__()
        assert d % n_heads == 0 
        self.n_heads = n_heads
        self.d_heads = d // n_heads
        self.d = d
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.apply(initialize_weights)
    
    def foward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads,
            self.d_heads).transpose (1,2)
        
        K = self.W_k(K).view(batch_size, -1, self.n_heads,
            self.d_heads).transpose(1,2)
        
        V = self.W_v(V).view(batch_size, -1, self.n_heads,
            self.d_heads).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_heads)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contigous().view(batch_size, -1, self.d)
        return self.W_o(context), attn

     

