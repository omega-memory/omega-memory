# Contributing to OMEGA

Thank you for your interest in contributing to OMEGA!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/core.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b my-feature`
5. Make your changes
6. Run tests: `pytest tests/`
7. Run linting: `ruff check src/`
8. Commit with a descriptive message
9. Push and open a pull request

## Development Setup

```bash
pip install -e ".[dev]"
pytest tests/        # Run tests
ruff check src/      # Lint
ruff format src/     # Format
```

## What to Contribute

- Bug fixes
- Documentation improvements
- Test coverage
- Performance optimizations
- New memory tool ideas (open an issue first to discuss)

## Code Style

- We use `ruff` for linting and formatting
- All code must pass `pytest tests/` before merging
- Follow existing patterns in the codebase

## Developer Certificate of Origin

By contributing, you certify that your contribution is your own work and you have the right to submit it under the Apache-2.0 license. We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO).

Sign your commits with `git commit -s` to add the DCO sign-off.

## Questions?

Open a [GitHub Discussion](https://github.com/omega-memory/core/discussions) or file an [issue](https://github.com/omega-memory/core/issues).
