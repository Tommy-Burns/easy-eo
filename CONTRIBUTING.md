# Contributing to Easy-EO

Thank you for your interest in contributing to **Easy-EO** 🎉
Contributions of all kinds are welcome and appreciated.

Easy-EO aims to provide **high-level, chainable abstractions** over common
earth-observation workflows while remaining **correct, performant, and
transparent**. This document outlines how you can contribute and what
standards we expect.

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Ways to Contribute

Contributions are not limited to writing code. You can help by:

- 🧩 Implementing new features or improving existing ones
- 🐛 Reporting bugs or edge cases
- 🧪 Adding or improving tests
- 📚 Improving documentation clarity and examples
- 📝 Fixing typos, improving explanations, or restructuring docs
- 💬 Reviewing pull requests or providing design feedback

Documentation improvements are **first-class contributions**.

---

## Code Contributions

### Prerequisites

If you are contributing **code**, you should be comfortable with:

- Python **3.10+**
- NumPy array operations
- Rasterio and GDAL concepts
- Core geospatial ideas:
  - CRS and reprojection
  - Affine transforms
  - Raster alignment and resampling
  - Nodata handling

Easy-EO intentionally abstracts Rasterio, but contributors must understand
what is happening *under the hood*. Code should not be treated as a black box.

> ⚠️ Contributions must not be purely AI-generated without human understanding,
> validation, or optimization.

---

### Design Philosophy

Please ensure that new code aligns with the project’s design principles:

- **Chainable operations** should return `EEORasterDataset`
- **Visualization functions** are terminal operations
- Avoid hidden side effects
- Preserve metadata (CRS, transform, nodata) wherever possible
- Favor explicit behavior over magic
- Follow the existing safety checks and validation patterns

---

### Performance & Safety

All code should:

- Avoid unnecessary copies of large arrays
- Use vectorized NumPy operations where possible
- Handle nodata, NaNs, and division safely
- Fail loudly and clearly when assumptions are violated

Correctness and clarity are prioritized.

---

## Development Setup

Set up a local environment once, then use it for all checks below.

1. **Fork and clone** the repository, then enter it:
   ```bash
   git clone https://github.com/<your-username>/easy-eo.git
   cd easy-eo
   ```

2. **Create and activate an isolated environment** (conda or venv):
   ```bash
   # conda
   conda create -n easy-eo-env python=3.12
   conda activate easy-eo-env

   # ...or a virtualenv
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install Easy-EO with the dev extras.** This pulls in the runtime
   dependencies plus `pytest`, `pytest-cov`, `ruff`, `mypy`, and
   `pre-commit`:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install the pre-commit git hook** (one-time, per clone):
   ```bash
   pre-commit install
   ```

> The `mypy` and `ruff` checks (and the `mypy` pre-commit hook) use the tools
> installed in this environment, so keep it **activated** whenever you run
> them or commit.

---

## Running Tests, Linting & Type Checks

Every pull request must pass linting, type checking, and the test suite; CI
runs all three on the full support matrix. Run them locally before opening a
PR.

### Formatting & linting (ruff)

```bash
ruff format eeo tests      # auto-format
ruff check eeo tests       # lint
ruff check --fix eeo tests # lint and auto-fix what it can
```

### Type checking (mypy) — step by step

1. Make sure your dev environment is **activated** (mypy is installed there).
2. From the repository root, run:
   ```bash
   mypy
   ```
   No path is needed: the `[tool.mypy]` config in `pyproject.toml` already
   targets the `eeo` package. `mypy eeo` is equivalent.
3. A clean run prints `Success: no issues found`. Otherwise mypy lists each
   error as `file:line: error: <message>` — fix them until the run is clean.
4. Errors about missing stubs for the geospatial stack (rasterio, geopandas,
   etc.) are already suppressed in config; you should not see them. If you add
   a **new** third-party dependency without type information, add it to the
   `ignore_missing_imports` override in `pyproject.toml`.

### Tests (pytest with coverage)

```bash
pytest                       # full suite; coverage is measured automatically
pytest tests/test_ops.py -x  # a single file, stop at first failure
```

Coverage is enforced: the run fails if total coverage drops below the
project threshold. Bug fixes must include a regression test, and new features
must include tests (see `CODE_STYLE.md`).

### pre-commit — step by step

`pre-commit` runs the whitespace, ruff, ruff-format, and mypy hooks so issues
are caught before they reach CI.

1. **One-time install** (if you have not already):
   ```bash
   pre-commit install
   ```
2. **Automatic on every commit.** After installing, `git commit` runs the
   hooks against your **staged** files. If a hook reformats a file, the commit
   is aborted and the fixes are left unstaged — review them, `git add` the
   changes, and commit again.
3. **Run manually across the whole tree** (recommended before opening a PR):
   ```bash
   pre-commit run --all-files
   ```
4. **Interpreting the output.** Fixer hooks (ruff, ruff-format, whitespace)
   edit files in place and report `Failed ... files were modified` when they
   change something — that means "fixed", not "broken". Re-stage and run
   again; a second run that reports only `Passed` means the tree is clean.
5. **Keep the environment active** — the `mypy` hook runs from your dev
   environment, so it needs `mypy` (and the installed dependencies) on the
   current `PATH`.

---

## Documentation Contributions

Improving documentation is highly encouraged and valued.

Good documentation should:

- Be readable by users who are **not GIS/RS experts**
- Clearly distinguish between chainable vs terminal operations
- Use consistent terminology
- Include examples where helpful

Documentation lives in:
```
docs/
├── user_guide/
├── modules/
└── getting_started.rst
```

To generate a local copy of the documentation, install the docs dependencies in `docs/requirements.txt`.
```commandline
pip install -r ./docs/requirements.txt
cd docs
make html
```
The local documentation can then be accessed at `docs/build`. This local build folder should not be pushed to GitHub.

---

## Pull Request Process

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes.
4. Run the checks locally until they all pass (see
   [Running Tests, Linting & Type Checks](#running-tests-linting--type-checks)):
   ```bash
   ruff check eeo tests
   ruff format --check eeo tests
   mypy
   pytest
   ```
   Or run everything at once with `pre-commit run --all-files` plus `pytest`.
5. Open a pull request (the PR template will prompt you) with:
   1. A clear description
   2. Rationale for design decisions
   3. Any known limitations
   4. Small, focused pull requests are preferred.

## Backers and Acknowledgements
Contributors who provide meaningful improvements — code, documentation,
design, or reviews — will be acknowledged in release notes or documentation.

If you are interested in supporting the project long-term, please open a
discussion or reach out via GitHub issues.

## Questions and Discussions
If you are unsure whether an idea fits the project:

- Open an issue
- Start a discussion
- Ask for feedback before writing large amounts of code
- We value thoughtful collaboration over volume.


```python
print("Thank you for helping make Easy-EO better")
```
