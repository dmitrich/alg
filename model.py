"""
Character-level GPT (Generative Pre-trained Transformer) model implementation.

This module implements a transformer-based language model with multi-head self-attention
and feed-forward layers. The model uses character-level tokenization and can be trained
on text data to generate new text sequences.

Architecture:
    - Token and position embeddings
    - Multiple transformer blocks with self-attention and feed-forward layers
    - Layer normalization
    - Language modeling head for next-token prediction

Model Persistence:
    The model supports saving and loading weights using the SafeTensors format, which provides:
    - **Security**: No arbitrary code execution during deserialization (unlike pickle-based formats)
    - **Performance**: Zero-copy loading via memory mapping, faster than torch.load()
    - **Interoperability**: Compatible with PyTorch, Hugging Face, and modern inference engines
    - **Determinism**: Reproducible serialization across platforms
    
    Use `save_safetensors()` and `load_safetensors()` methods for model persistence.
    Legacy text-based weight files are deprecated.

Example:
    >>> # Training
    >>> model = BigramLanguageModel()
    >>> # ... train the model ...
    >>> model.save_safetensors('model.safetensors')
    
    >>> # Inference
    >>> model = BigramLanguageModel()
    >>> model.load_safetensors('model.safetensors')
    >>> model.eval()
    >>> # ... generate text ...

Note:
    Model hyperparameters (n_embd, n_head, n_layer, etc.) are currently defined as module-level
    globals. These should be set before instantiating the model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
vocab_size = 65 # size of the vocabulary (will be set from data)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    """
    Character-level transformer language model with multi-head self-attention.
    
    This model implements a decoder-only transformer architecture for character-level
    language modeling. It uses token and position embeddings, multiple transformer blocks
    with self-attention and feed-forward layers, and a language modeling head for
    next-token prediction.
    
    Architecture Components:
        - Token embedding table: Maps character indices to embedding vectors
        - Position embedding table: Adds positional information to token embeddings
        - Transformer blocks: Multiple layers of self-attention and feed-forward networks
        - Layer normalization: Applied after transformer blocks
        - Language modeling head: Projects embeddings to vocabulary logits
    
    Model Configuration:
        The model architecture is determined by module-level hyperparameters:
        - vocab_size: Size of the character vocabulary
        - n_embd: Embedding dimension
        - n_head: Number of attention heads
        - n_layer: Number of transformer blocks
        - block_size: Maximum context length
        - dropout: Dropout probability
    
    Training:
        The model can be trained using the forward() method which computes cross-entropy
        loss when targets are provided. After training, use save_safetensors() to persist
        the model weights.
    
    Inference:
        For text generation, use the generate() method which autoregressively samples
        tokens from the model's output distribution. Load pre-trained weights using
        load_safetensors() before inference.
    
    Example:
        >>> # Initialize model
        >>> model = BigramLanguageModel()
        >>> model.to(device)
        
        >>> # Training
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        >>> for batch in data_loader:
        ...     xb, yb = batch
        ...     logits, loss = model(xb, yb)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
        
        >>> # Save trained model
        >>> model.save_safetensors('model.safetensors')
        
        >>> # Load for inference
        >>> model = BigramLanguageModel()
        >>> model.load_safetensors('model.safetensors')
        >>> model.eval()
        
        >>> # Generate text
        >>> context = torch.zeros((1, 1), dtype=torch.long, device=device)
        >>> generated = model.generate(context, max_new_tokens=100)
    
    Attributes:
        token_embedding_table (nn.Embedding): Character token embeddings
        position_embedding_table (nn.Embedding): Positional embeddings
        blocks (nn.Sequential): Stack of transformer blocks
        ln_f (nn.LayerNorm): Final layer normalization
        lm_head (nn.Linear): Language modeling head for logit prediction
    
    Methods:
        forward(idx, targets=None): Compute logits and optionally loss
        generate(idx, max_new_tokens): Generate new tokens autoregressively
        save_safetensors(path): Save model weights in SafeTensors format
        load_safetensors(path): Load model weights from SafeTensors format
        save_weights(weights_dir): [DEPRECATED] Save weights as text files
        load_weights(weights_dir): [DEPRECATED] Load weights from text files
    
    Note:
        The model name "BigramLanguageModel" is historical and does not reflect the
        actual architecture, which is a full transformer model, not a simple bigram model.
    """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Forward pass through the model.
        
        Computes logits for next-token prediction and optionally calculates cross-entropy
        loss when targets are provided. This method is called during both training and
        inference.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (B, T) where:
                - B is the batch size
                - T is the sequence length (must be <= block_size)
                Values should be integers in range [0, vocab_size)
            
            targets (torch.Tensor, optional): Target token indices of shape (B, T) for
                computing loss during training. If None, only logits are returned without
                loss computation. Values should be integers in range [0, vocab_size).
                Default: None
        
        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Predicted logits of shape (B, T, vocab_size)
                  representing unnormalized log probabilities for each token in vocabulary
                - loss (torch.Tensor or None): Cross-entropy loss if targets provided,
                  otherwise None. Loss is averaged over all tokens in the batch.
        
        Example:
            >>> # Training mode with loss computation
            >>> model = BigramLanguageModel()
            >>> xb = torch.randint(0, vocab_size, (4, 32))  # batch_size=4, block_size=32
            >>> yb = torch.randint(0, vocab_size, (4, 32))
            >>> logits, loss = model(xb, yb)
            >>> print(logits.shape)  # torch.Size([4, 32, 65])
            >>> print(loss.item())   # scalar loss value
            
            >>> # Inference mode without loss
            >>> logits, loss = model(xb)
            >>> print(loss)  # None
        
        Note:
            - The method automatically handles token and position embeddings
            - Position embeddings are generated for sequence length T
            - Logits are reshaped for loss computation when targets are provided
            - Uses cross-entropy loss which combines log_softmax and NLL loss
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively from the model.
        
        This method performs autoregressive text generation by repeatedly predicting the
        next token and appending it to the sequence. It uses multinomial sampling from
        the model's output distribution to generate diverse outputs.
        
        Args:
            idx (torch.Tensor): Initial context tokens of shape (B, T) where:
                - B is the batch size (number of sequences to generate in parallel)
                - T is the initial context length (can be any length)
                Values should be integers in range [0, vocab_size)
            
            max_new_tokens (int): Number of new tokens to generate. The output sequence
                will have length T + max_new_tokens.
        
        Returns:
            torch.Tensor: Generated token indices of shape (B, T + max_new_tokens)
                containing the original context followed by newly generated tokens.
        
        Example:
            >>> # Generate from empty context
            >>> model = BigramLanguageModel()
            >>> model.load_safetensors('model.safetensors')
            >>> model.eval()
            >>> 
            >>> # Start with newline character (index 0)
            >>> context = torch.zeros((1, 1), dtype=torch.long, device=device)
            >>> generated = model.generate(context, max_new_tokens=100)
            >>> print(generated.shape)  # torch.Size([1, 101])
            >>> 
            >>> # Decode to text
            >>> from utils import decode
            >>> text = decode(generated[0].tolist())
            >>> print(text)
            
            >>> # Generate multiple sequences in parallel
            >>> context = torch.zeros((4, 1), dtype=torch.long, device=device)
            >>> generated = model.generate(context, max_new_tokens=50)
            >>> print(generated.shape)  # torch.Size([4, 51])
        
        Note:
            - Generation is performed autoregressively (one token at a time)
            - Uses multinomial sampling for diversity (not greedy decoding)
            - Context is automatically cropped to last block_size tokens if too long
            - Model should be in eval mode (model.eval()) for inference
            - Set torch.manual_seed() for reproducible generation
            - Temperature and top-k/top-p sampling are not currently implemented
        
        Implementation Details:
            1. Crop context to last block_size tokens (if needed)
            2. Get logits from forward pass
            3. Extract logits for last position
            4. Apply softmax to get probabilities
            5. Sample next token from multinomial distribution
            6. Append to sequence and repeat
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    # Added for weight persistence - save model weights to separate files
    def save_weights(self, weights_dir='weights'):
        """
        Save each model parameter to a separate text file.
        
        .. deprecated::
            Use :meth:`save_safetensors` instead. This method is deprecated and will be
            removed in a future version. The text-based format is inefficient and non-standard.
            SafeTensors provides faster loading, smaller file sizes, and better security.
        
        Args:
            weights_dir: Directory to save weight files (default: 'weights')
        
        See Also:
            save_safetensors(): Modern method for saving model weights in SafeTensors format
        """
        import os
        import warnings
        
        warnings.warn(
            "save_weights() is deprecated and will be removed in a future version. "
            "Use save_safetensors() instead for better performance, security, and compatibility. "
            "SafeTensors provides faster loading, smaller file sizes, and is the industry standard format.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create weights directory if it doesn't exist - added for persistence
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save each parameter to a separate file - added for persistence
        for name, param in self.named_parameters():
            # Use parameter name as filename - added for persistence
            filename = os.path.join(weights_dir, f"{name}.txt")
            # Convert tensor to numpy and save as text - added for persistence
            with open(filename, 'w') as f:
                # Save tensor shape first for validation - added for persistence
                f.write(f"# shape: {list(param.shape)}\n")
                # Save flattened tensor values - added for persistence
                param_data = param.detach().cpu().numpy().flatten()
                for val in param_data:
                    f.write(f"{val}\n")
    
    def save_safetensors(self, path: str = 'model.safetensors') -> None:
        """
        Save model weights in SafeTensors format.
        
        SafeTensors is a secure, fast, and deterministic format for storing model weights.
        It provides zero-copy loading, memory mapping support, and prevents arbitrary code
        execution during deserialization.
        
        Args:
            path: Path to save the SafeTensors file (default: 'model.safetensors')
        
        Example:
            >>> model = BigramLanguageModel()
            >>> # After training...
            >>> model.save_safetensors('model.safetensors')
            >>> # Or with custom path
            >>> model.save_safetensors('checkpoints/model_epoch_10.safetensors')
        
        Notes:
            - The saved file contains all trainable parameters (embeddings, attention weights,
              MLP weights, LayerNorm parameters, and output head)
            - Optimizer state is NOT included (handled separately if needed)
            - The format is compatible with PyTorch, Hugging Face, and modern inference engines
            - Serialization is deterministic - same weights produce identical files
        
        See Also:
            load_safetensors(): Load weights from SafeTensors format
        """
        from safetensors.torch import save_file
        state_dict = self.state_dict()
        save_file(state_dict, path)
    
    def load_safetensors(self, path: str = 'model.safetensors') -> None:
        """
        Load model weights from SafeTensors format.
        
        SafeTensors is a secure, fast, and deterministic format for loading model weights.
        It provides zero-copy loading, memory mapping support, and prevents arbitrary code
        execution during deserialization, making it safe to load weights from untrusted sources.
        
        Args:
            path: Path to the SafeTensors file to load (default: 'model.safetensors')
        
        Raises:
            FileNotFoundError: If the specified SafeTensors file does not exist.
                Please ensure the file path is correct or train a model first.
            RuntimeError: If the weights in the file are incompatible with the model architecture.
                This can occur if:
                - The model architecture has changed (different n_embd, n_head, n_layer, etc.)
                - The file contains weights for a different model
                - Parameter shapes don't match expected dimensions
        
        Example:
            >>> model = BigramLanguageModel()
            >>> # Load from default path
            >>> model.load_safetensors('model.safetensors')
            >>> # Or from custom path
            >>> model.load_safetensors('checkpoints/model_epoch_10.safetensors')
            >>> # Now ready for inference
            >>> model.eval()
        
        Notes:
            - All weights are loaded to the same device as the model's current parameters
            - The method uses PyTorch's load_state_dict() with strict=True by default,
              meaning all parameter names and shapes must match exactly
            - Loading is significantly faster than pickle-based formats (torch.load)
            - The format is safe - no arbitrary code execution is possible
            - Memory mapping is used when supported, reducing memory overhead
        
        See Also:
            save_safetensors(): Save weights to SafeTensors format
        """
        import os
        from safetensors.torch import load_file
        
        # Check if file exists before attempting to load
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SafeTensors file not found at '{path}'. "
                f"Please ensure the file path is correct or train a model first using train.py."
            )
        
        try:
            # Load state dict from SafeTensors file
            state_dict = load_file(path)
            
            # Load weights into model
            self.load_state_dict(state_dict)
            
        except RuntimeError as e:
            # Handle incompatible architecture errors
            raise RuntimeError(
                f"Failed to load weights from '{path}'. The weights are incompatible with the current model architecture. "
                f"This may occur if the model configuration (n_embd, n_head, n_layer, vocab_size, etc.) "
                f"has changed since the weights were saved. "
                f"Original error: {str(e)}"
            )
    
    # Added for weight persistence - load model weights from separate files
    def load_weights(self, weights_dir='weights'):
        """
        Load model parameters from separate text files.
        
        .. deprecated::
            Use :meth:`load_safetensors` instead. This method is deprecated and will be
            removed in a future version. The text-based format is inefficient and non-standard.
            SafeTensors provides faster loading, smaller file sizes, and better security.
        
        Args:
            weights_dir: Directory containing weight files (default: 'weights')
            
        Raises:
            FileNotFoundError: If weights directory or weight files are missing
            ValueError: If weight files are corrupted or incompatible
            IOError: If weight files cannot be read
        
        See Also:
            load_safetensors(): Modern method for loading model weights from SafeTensors format
        """
        import os
        import numpy as np
        import warnings
        
        warnings.warn(
            "load_weights() is deprecated and will be removed in a future version. "
            "Use load_safetensors() instead for better performance, security, and compatibility. "
            "SafeTensors provides faster loading, smaller file sizes, and is the industry standard format.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Check if weights directory exists - added for error handling
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(
                f"Weights directory not found at '{weights_dir}'. "
                f"Please train a model first using train.py or verify the weights directory path."
            )
        
        # Load each parameter from its file - added for persistence
        for name, param in self.named_parameters():
            # Construct filename from parameter name - added for persistence
            filename = os.path.join(weights_dir, f"{name}.txt")
            
            # Check if weight file exists - added for error handling
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Weight file not found: '{filename}'. "
                    f"The model expects {len(list(self.named_parameters()))} weight files. "
                    f"Please ensure all weight files are present or retrain the model."
                )
            
            # Try-except block for weight loading failures - added for error handling
            try:
                # Load weight values from file - added for persistence
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    
                    # Validate file is not empty - added for error handling
                    if not lines:
                        raise ValueError(f"Weight file '{filename}' is empty.")
                    
                    # Parse shape from first line - added for persistence
                    shape_line = lines[0].strip()
                    if not shape_line.startswith("# shape:"):
                        raise ValueError(
                            f"Weight file '{filename}' is corrupted or in wrong format: "
                            f"missing shape information. Expected first line to start with '# shape:'."
                        )
                    
                    # Extract shape - added for persistence
                    shape_str = shape_line.replace("# shape:", "").strip()
                    try:
                        expected_shape = eval(shape_str)
                    except Exception as e:
                        raise ValueError(
                            f"Weight file '{filename}' has invalid shape format: {shape_str}. "
                            f"Error: {str(e)}"
                        )
                    
                    # Validate shape matches parameter - added for error handling
                    if list(param.shape) != expected_shape:
                        raise ValueError(
                            f"Weight file '{filename}' has incompatible shape. "
                            f"Model expects shape {list(param.shape)}, but file contains shape {expected_shape}. "
                            f"This may indicate the weight file is from a different model architecture. "
                            f"Please retrain the model or use compatible weight files."
                        )
                    
                    # Load weight values - added for persistence
                    values = []
                    for line_num, line in enumerate(lines[1:], start=2):
                        line = line.strip()
                        if line:  # Skip empty lines - added for persistence
                            try:
                                values.append(float(line))
                            except ValueError as e:
                                raise ValueError(
                                    f"Weight file '{filename}' contains invalid value at line {line_num}: '{line}'. "
                                    f"Expected a numeric value. Error: {str(e)}"
                                )
                    
                    # Validate correct number of values - added for error handling
                    expected_count = np.prod(expected_shape)
                    if len(values) != expected_count:
                        raise ValueError(
                            f"Weight file '{filename}' has incorrect number of values. "
                            f"Expected {expected_count} values for shape {expected_shape}, "
                            f"but found {len(values)} values. The file may be corrupted or incomplete."
                        )
                    
                    # Convert to numpy array and reshape - added for persistence
                    try:
                        weight_array = np.array(values, dtype=np.float32).reshape(expected_shape)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to reshape weight data from '{filename}'. "
                            f"Shape: {expected_shape}, Values: {len(values)}. Error: {str(e)}"
                        )
                    
                    # Load into parameter - added for persistence
                    param.data = torch.from_numpy(weight_array).to(param.device)
                    
            except IOError as e:
                # Handle file I/O errors - added for error handling
                raise IOError(
                    f"Failed to read weight file '{filename}'. "
                    f"Please check file permissions and disk space. Error: {str(e)}"
                )
            except (ValueError, FileNotFoundError) as e:
                # Re-raise validation errors with context - added for error handling
                raise
            except Exception as e:
                # Catch any unexpected errors - added for error handling
                raise RuntimeError(
                    f"Unexpected error while loading weights from '{filename}': {str(e)}. "
                    f"The weight file may be corrupted. Please retrain the model."
                )
