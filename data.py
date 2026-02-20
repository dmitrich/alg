"""
Data loading and batch generation utilities.

This module provides utilities for loading training data and generating batches
for training and evaluation. It handles data splitting, tokenization, and batch
sampling for the GPT model training pipeline.
"""

import torch
from typing import Tuple
from tokenizer import create_tokenizer

def load_data(data_path: str = 'data/input.txt') -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Load and split training data.
    
    This function loads text data from a file, tokenizes it, and splits it into
    training and validation sets. The split is 90% training and 10% validation.
    
    Args:
        data_path (str): Path to input.txt file (default: 'data/input.txt')
        
    Returns:
        Tuple containing:
            - train_data (torch.Tensor): Training data tensor (90% of data) containing
              integer token IDs
            - val_data (torch.Tensor): Validation data tensor (10% of data) containing
              integer token IDs
            - text (str): Original text string used to build the tokenizer
        
    Raises:
        FileNotFoundError: If data file doesn't exist at the specified path
        IOError: If data file cannot be read due to permissions or other I/O errors
    
    Example:
        >>> train_data, val_data, text = load_data('data/input.txt')
        >>> print(f"Training samples: {len(train_data)}")
        Training samples: 900000
        >>> print(f"Validation samples: {len(val_data)}")
        Validation samples: 100000
        >>> print(f"Total characters: {len(text)}")
        Total characters: 1000000
    
    Note:
        - The function automatically creates a character-level tokenizer from the text
        - The split ratio is fixed at 90/10 for training/validation
        - The same data file must be used for both training and inference to ensure
          consistent vocabulary
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Training data not found at '{data_path}'. "
            f"Please ensure the data file exists at the specified path. "
            f"For training, place the Shakespeare dataset at 'data/input.txt'."
        )
    except IOError as e:
        raise IOError(
            f"Failed to read training data from '{data_path}': {str(e)}. "
            f"Please check file permissions and ensure the file is readable."
        )
    
    encode, decode, vocab_size = create_tokenizer(text)
    
    data = torch.tensor(encode(text), dtype=torch.long)
    
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, text

def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of training data.
    
    This function samples random sequences from the training or validation data
    to create a batch for training or evaluation. Each sequence has length block_size,
    and the batch contains batch_size sequences.
    
    Args:
        split (str): 'train' or 'val' to specify which dataset to sample from
        train_data (torch.Tensor): Training data tensor containing integer token IDs
        val_data (torch.Tensor): Validation data tensor containing integer token IDs
        block_size (int): Context length (number of tokens per sequence)
        batch_size (int): Number of sequences in the batch
        device (str): 'cuda' or 'cpu' for GPU/CPU placement (default: 'cpu')
        
    Returns:
        Tuple containing:
            - x (torch.Tensor): Input tensor of shape (batch_size, block_size) containing
              token IDs for the input sequences
            - y (torch.Tensor): Target tensor of shape (batch_size, block_size) containing
              token IDs for the target sequences (shifted by 1 position)
    
    Example:
        >>> # Generate a training batch
        >>> xb, yb = get_batch('train', train_data, val_data, block_size=32, batch_size=4, device='cuda')
        >>> print(xb.shape)
        torch.Size([4, 32])
        >>> print(yb.shape)
        torch.Size([4, 32])
        >>> 
        >>> # The target is shifted by 1 position
        >>> # xb[0, :] predicts yb[0, :]
        >>> # where yb[0, i] is the next token after xb[0, i]
    
    Note:
        - Sequences are sampled randomly from the dataset
        - The target sequence is the input sequence shifted by 1 position
        - This implements the standard language modeling objective: predict the next token
        - Tensors are automatically moved to the specified device (GPU/CPU)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
