# Coding Constraints

These rules define the default coding behavior for this repository when using Kiro.

## Core Principles

- Optimize for speed and clarity over extensibility
- Assume code is disposable and short-lived
- Prefer fewer lines over abstraction
- Readability beats architecture

## Hard Rules

- Do NOT write unit tests or integration tests unless explicitly requested
- Do NOT generate mocks, fixtures, or test scaffolding
- Prefer single-file solutions
- Avoid helper utilities and shared libraries
- No design patterns unless strictly required by the language
- No future-proofing or speculative abstractions

## What to Avoid

- Dependency injection
- Configuration layers
- Excessive error handling
- Logging and observability code
- Commenting obvious code
- Explaining best practices

## Assumptions

- Trusted, single-user environment
- Non-hostile inputs
- Manual validation is sufficient

## Python Environment

- Always use uv virtual environment
- Environment name matches project folder name (e.g., `alg1` folder → `alg1` environment)
- Environment location: `.venv` in project root
- Install packages with: `uv pip install <package>`

## Default Guidance

When in doubt:
- Make it simpler
- Inline it
- Delete unnecessary code

## Python Naming Conventions

**File Names:**
- Project files: lowercase with hyphens (e.g., `config_loader.py`)
- Test/temporary files: lowercase with underscores (e.g., `test_model.py`, `_temp_data.py`)
- Existing pattern: snake_case for modules (e.g., `model.py`, `train.py`, `data.py`, `utils.py`)

**Code:**
- Variables/functions: snake_case (e.g., `batch_size`, `load_config`, `get_batch`)
- Classes: PascalCase (e.g., `BigramLanguageModel`, `ModelConfig`, `Head`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_ITERS`, `DEVICE`)
- Private: prefix with underscore (e.g., `_internal_method`)

## Documentation Style (Observed Patterns)

**Module Docstrings:**
- Multi-line triple quotes at top of file
- Brief description, then detailed explanation
- Include Architecture/Usage/Example sections
- Use reStructuredText style for complex docs

**Class Docstrings:**
- Comprehensive description of purpose
- List all attributes with types and descriptions
- Include example usage
- Document all methods

**Function Docstrings:**
- One-line summary for simple functions
- Multi-line with Args/Returns/Raises for complex functions
- Use Google/NumPy style format
- Include type hints in signature, not docstring

**Example:**
```python
def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
    """
    Forward pass through the model.
    
    Args:
        idx: Input token indices of shape (B, T)
        targets: Target token indices (optional)
    
    Returns:
        tuple: (logits, loss)
    """
```

## Type Hints (Observed Patterns)

- Use type hints for all function signatures
- Import from `typing` module: `Tuple`, `Optional`, `Dict`, `Any`, `List`, `Callable`
- Use `torch.Tensor` for PyTorch tensors
- Use `-> None` for functions with no return
- Use `-> tuple` or `-> Tuple[Tensor, Optional[Tensor]]` for multiple returns

**Example:**
```python
from typing import Tuple, Optional
import torch

def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor,
              block_size: int, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of training data."""
```

## Code Organization (Observed Patterns)

**Import Order:**
1. Standard library (e.g., `import json`, `import os`)
2. Third-party (e.g., `import torch`, `import numpy`)
3. Local modules (e.g., `from config import load_config`)
4. Group related imports together with comments

**Example:**
```python
# Added for modular imports - PyTorch core modules
import torch
import torch.nn as nn

# Added for CLI support - command-line argument parsing
import argparse

# Added for modular imports - configuration management
from config import load_config, validate_config
```

**Module Structure:**
1. Module docstring
2. Imports
3. Constants/globals (if needed)
4. Helper functions
5. Classes
6. Main execution (if script)

## Comments (Observed Patterns)

- Use inline comments for non-obvious logic: `# (B, T, C)`
- Add "Added for X" comments when introducing new functionality
- Explain tensor shapes in comments: `# (B,T,C) -> (B, T, vocab_size)`
- Document why, not what (code should be self-explanatory)

**Example:**
```python
# crop idx to the last block_size tokens
idx_cond = idx[:, -block_size:]

# compute attention scores ("affinities")
wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
```

## Error Handling (Observed Patterns)

- Use descriptive error messages with context
- Provide actionable guidance in error messages
- Use try-except for file I/O and external operations
- Validate inputs early with clear ValueError messages

**Example:**
```python
if not os.path.exists(path):
    raise FileNotFoundError(
        f"SafeTensors file not found at '{path}'. "
        f"Please ensure the file path is correct or train a model first using train.py."
    )
```

## Configuration (Observed Patterns)

- Use dataclasses for type-safe configuration (`@dataclass`)
- Provide validation methods (`validate()`)
- Support JSON serialization (`from_dict()`, `to_dict()`)
- Use argparse for CLI with sensible defaults
- Allow CLI args to override config file values

## PyTorch Specific (Observed Patterns)

- Use `nn.Module` for all model components
- Call `super().__init__()` in constructors
- Use `@torch.no_grad()` decorator for evaluation
- Move tensors to device explicitly: `.to(device)`
- Use `model.eval()` and `model.train()` modes
- Prefer `F.cross_entropy()` over manual loss computation

## Code Style Preferences

- Line length: ~100 characters (flexible, not strict)
- Use 4 spaces for indentation
- Blank line between functions/methods
- Two blank lines between classes
- Use f-strings for formatting: `f"step {iter}: loss {loss:.4f}"`
- Prefer explicit over implicit: `if x is None:` not `if not x:` 
