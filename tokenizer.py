"""
Tokenization utilities for character-level encoding/decoding.

This module provides utilities for creating character-level tokenizers that convert
text to integer sequences and vice versa. The tokenizer is built from a training
corpus and creates a vocabulary of all unique characters.
"""

from typing import Callable, Tuple

def create_tokenizer(text: str) -> Tuple[Callable[[str], list], Callable[[list], str], int]:
    """
    Create character-level tokenizer from text.
    
    This function builds a character-level tokenizer by:
    1. Extracting all unique characters from the input text
    2. Creating bidirectional mappings between characters and integers
    3. Returning encode/decode functions and vocabulary size
    
    Args:
        text (str): Training text to build vocabulary from. The vocabulary will
                   consist of all unique characters found in this text.
        
    Returns:
        Tuple containing:
            - encode (Callable[[str], list]): Function that takes a string and returns
              a list of integer token IDs
            - decode (Callable[[list], str]): Function that takes a list of integer
              token IDs and returns a string
            - vocab_size (int): Size of the vocabulary (number of unique characters)
    
    Example:
        >>> text = "hello world"
        >>> encode, decode, vocab_size = create_tokenizer(text)
        >>> print(vocab_size)
        8  # unique characters: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
        >>> 
        >>> # Encode text to integers
        >>> tokens = encode("hello")
        >>> print(tokens)
        [3, 4, 5, 5, 6]
        >>> 
        >>> # Decode integers back to text
        >>> text = decode(tokens)
        >>> print(text)
        'hello'
    
    Note:
        - The vocabulary is sorted alphabetically for deterministic behavior
        - Each unique character gets a unique integer ID
        - The tokenizer is character-level, not subword or word-level
        - The same text must be used to build the tokenizer for both training and inference
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode, vocab_size
