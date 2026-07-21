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

We use [**uv**](https://docs.astral.sh/uv/) to manage the development
environment. A committed `uv.lock` pins the exact versions used by every
contributor and by CI, so everyone works against the same dependency set.
This is the recommended path; a plain-pip alternative follows.

### Recommended: uv

1. **Install uv** (see the [official instructions](https://docs.astral.sh/uv/getting-started/installation/));
   for example:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Fork and clone** the repository, then enter it:
   ```bash
   git clone https://github.com/<your-username>/easy-eo.git
   cd easy-eo
   ```

3. **Create the locked environment** with the dev dependencies. This builds a
   `.venv/` from `uv.lock` (runtime deps plus `pytest`, `pytest-cov`, `ruff`,
   `mypy`, and `pre-commit`):
   ```bash
   uv sync --extra dev
   ```

4. **Install the pre-commit git hook** (one-time, per clone):
   ```bash
   uv run pre-commit install
   ```

Prefix commands with `uv run` to execute them inside the locked environment
(e.g. `uv run pytest`, `uv run mypy`), or activate it once with
`source .venv/bin/activate` (`\.venv\Scripts\activate` on Windows) and run the
tools directly.

#### Updating dependencies

Edit the dependency ranges in `pyproject.toml`, then refresh the lockfile and
commit it:

```bash
uv lock            # re-resolve and update uv.lock
uv sync --extra dev
```

Run `uv lock --check` to verify the lockfile is in sync with `pyproject.toml`
without changing it (CI installs with `uv sync --frozen`, which fails if they
have drifted). Routine version bumps normally arrive via reviewed
Dependabot/Renovate PRs rather than ad hoc.

### Alternative: pip

If you prefer not to use uv, install into a conda env or virtualenv. Note this
resolves fresh versions rather than the locked set:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

> Whichever path you choose, the `mypy`, `ruff`, and `pre-commit` tools run
> from this environment, so keep it **activated** (or use `uv run`) whenever
> you run the checks or commit.

---

## Running Tests, Linting & Type Checks

Every pull request must pass linting, type checking, and the test suite; CI
runs all three on the full support matrix. Run them locally before opening a
PR.

The commands below assume an **activated** environment. With uv, either
activate `.venv` or prefix each command with `uv run` (e.g. `uv run pytest`).

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
