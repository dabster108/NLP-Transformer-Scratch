import torch
import torch.nn as nn
import math


#Position Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.pe = pe.unsqueeze(0)  # (1, max_len, embed_size)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


# Scaled dot product attenion 
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert embed_size % heads == 0

        self.query = nn.Linear(embed_size, embed_size)
        self.key   = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # reshape into multiple heads
        Q = Q.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.embed_size)

        return self.fc_out(out)

# FEED FORWARD NETWORK
class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, expansion * embed_size),
            nn.ReLU(),
            nn.Linear(expansion * embed_size, embed_size)
        )

    def forward(self, x):
        return self.net(x)
