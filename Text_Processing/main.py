import re
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os


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