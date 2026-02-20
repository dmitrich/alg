# Clean Branch

This branch contains the same codebase as `main` but with all inline and standalone comments removed, keeping only docstrings.

## What Was Removed

- All inline comments (e.g., `# This is a comment`)
- All standalone comment lines
- Excessive blank lines (reduced to max 1 consecutive blank line)

## What Was Preserved

- All docstrings (module, class, function, and method documentation)
- All code functionality
- All imports and structure
- Code formatting and indentation

## Statistics

**Changes from main branch:**
- 20 files modified
- 391 lines removed (comments)
- 47 lines added (formatting adjustments)
- Net reduction: 344 lines

## Purpose

This branch provides a cleaner, more concise version of the codebase for:
- Production deployment
- Code review focused on logic rather than comments
- Minimalist code style preference
- Reduced file sizes

## Branches

- **main**: Full codebase with comprehensive comments and documentation
- **clean**: Same codebase with comments removed, docstrings preserved

## Usage

Switch to this branch:
```bash
git checkout clean
```

Or clone directly:
```bash
git clone -b clean https://github.com/dmitrich/alg.git
```

## Note

All functionality remains identical to the `main` branch. The only difference is the removal of comments. Docstrings are fully preserved, so API documentation is still available via:

```python
help(LanguageModel)
help(LanguageModel.prefill)
help(LanguageModel.decode)
```

## Maintenance

This branch should be updated whenever significant changes are made to `main`:

```bash
git checkout clean
git merge main
python remove_comments.py  # If comments were added
git commit -m "sync with main and remove new comments"
git push origin clean
```
