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

## Default Guidance

When in doubt:
- Make it simpler
- Inline it
- Delete unnecessary code
