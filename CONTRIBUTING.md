# Contributing to PromptManifold

Thank you for your interest in contributing to PromptManifold. We welcome high-impact contributions that maintain our standards of deterministic performance, strict zero-drift semantic guarantees, and clean system architecture.

## Code of Conduct

All contributors and maintainers are expected to maintain professional, technical, and respectful collaboration.

## Development Workflow

1. Fork the repository and create your feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Set up the development virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run the complete test suite and linter:
   ```bash
   pytest -v
   ruff check src tests examples
   ```
4. Commit your changes referencing relevant issues:
   ```bash
   git commit -m "feat(module): description of changes (#issue-id)"
   ```
5. Submit a pull request with technical context, benchmark output, and verification proof.

## Documentation Standards

- Zero emojis: Do not use emojis in code, commits, or documentation files.
- Mermaid diagrams: Use Mermaid (`mermaid`) for any system, subsystem, or workflow blueprints.
- Line length: Keep line length within 100 characters where possible.
