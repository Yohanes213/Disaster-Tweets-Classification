import torch
import numpy as np

def calculate_accuracy(predictions, targets):
    n_correct = (predictions == targets).sum().item()
    accuracy = n_correct / len(targets)

    return np.round((accuracy * 100), 5)

def tokenize_text(text, tokenizer, device):
    encoded_text = tokenizer(text, return_tensors='pt', padding=True)
    return encoded_text.to(device)