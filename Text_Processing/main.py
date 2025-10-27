import re
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'test.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()
print(f"Text length: {len(text)} characters")
print(text)


#Step 2: Read and Clean Text

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove digits
    text = text.strip()
    return text

cleaned_text = clean_text(text)
print("Cleaned Text:\n", cleaned_text[:300], "...\n")

# Tokenization
tokens = cleaned_text.split()
print(f"Total tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")

# Build Vocabulary
vocab = {word: idx+1 for idx, word in enumerate(set(tokens))}  # word -> index
vocab_size = len(vocab) + 1  # +1 for padding index 0
print("Vocabulary size:", vocab_size)

# Convert text to sequences
sequence = [vocab[word] for word in tokens]
print("First 20 indices:", sequence[:20])


#padding sequences
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        seq += [0]*(max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

max_len = 50
padded_seq = pad_sequence(sequence, max_len)
print("Padded sequence:", padded_seq)

#Creating the dataset 
class TextDataset(Dataset):
    def __init__(self, seq):
        self.data = torch.tensor(seq, dtype=torch.long)
    
    def __len__(self):
        return 1 
    
    def __getitem__(self, idx):
        return self.data

dataset = TextDataset(padded_seq)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)



#Creating a simple lstm model 
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # For demonstration
    
    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last output
        return out
    
model = SimpleLSTM(vocab_size=vocab_size, embed_dim=10, hidden_dim=20)
for batch in dataloader:
    output = model(batch)
    print("Model output:", output)

# Example of applying a sigmoid to get probabilities
prob = torch.sigmoid(output)
print("Predicted probability:", prob.item())