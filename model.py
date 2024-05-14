import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class MyPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        # creating a matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)
        # creating a vector of shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # (seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return x

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))   
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class MultiheadAttnBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # (batch_size, h, seq_length, d_k) x (batch_size, h, d_k, seq_length) = (batch_size, h, seq_length, seq_length)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        mask = mask.unsqueeze(0).unsqueeze(1)
        # print("Shape of mask: ", mask)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf')) # (batch_size, h, seq_length, seq_length)
        attention_score = F.softmax(attention_score, dim=-1) # (batch_size, h, seq_length, seq_length)
        # print("Shape of attention_score: ", attention_score)

        if dropout is not None: 
            attention_score = dropout(attention_score)
        return torch.matmul(attention_score, value), attention_score
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        query = self.w_q(q) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        key = self.w_k(k) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        value = self.w_v(v) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k) 
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k)
        
        x, attention_score = MultiheadAttnBlock.attention(query, key, value, mask, self.dropout) # (batch_size, h, seq_length, d_k)
        x = x.permute(0,2,1,3).contiguous().view(x.size(0), -1, self.h * self.d_k) # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
        x = self.w_o(x) 
        return x # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

     
class feed_forward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class final_layer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, hidden_dim, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
   
class model_v(nn.Module):
    def __init__(self, emb, norm, pos, attn,feed_forward, d_model, fl, seq_length, hidden_dim, h,vocab_size, mask=None):
        super().__init__()
        self.mask = mask  
        self.emb = emb(vocab_size, d_model)
        self.norm = norm()
        self.feed_forward = feed_forward(d_model, hidden_dim)
        self.pos = pos(d_model = d_model, seq_length = seq_length)
        self.attn = attn(d_model, h) 
        self.fl =  fl(d_model = d_model, vocab_size = vocab_size, hidden_dim = hidden_dim)
    def forward(self, x): 
        x = self.emb(x)
        x = self.pos(x)
        x = self.norm(x)
        x = self.attn(x,x,x, self.mask)
        x = self.feed_forward(x)
        x = self.norm(x)
        x = self.attn(x,x,x, self.mask)
        x = self.feed_forward(x)
        x = self.norm(x)       
        x = self.fl(x)
        return x

