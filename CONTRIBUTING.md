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

#### The default run is offline

No test may reach the network by default — a suite that depends on someone
else's uptime is a suite that fails for reasons you cannot fix. An autouse
fixture blocks Python-level socket connections and fails any test that tries,
so this is enforced rather than merely encouraged.

Work from local files, fakes, or the recorded API responses in
`tests/data/stac/` (refresh them with `python scripts/record_stac_responses.py`;
see the README there). A test that genuinely needs the network is marked and
opted into explicitly:

```python
@pytest.mark.network
def test_real_download():
    ...
```

```bash
pytest --run-network          # include the marked tests
```

Note that GDAL's HTTP stack does not use Python sockets, so the fixture cannot
catch a remote raster read; keep those out of the default suite by design.

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

## Releasing (section for easy-eo maintainers)

A release lands in two places. **PyPI** is fully automated from a tag.
**conda-forge** is a separate repository that does not read this one, and it
needs a review step you cannot skip safely — see step 3.

### 1. Prepare the release here

1. **Pick the version.** While the project is pre-1.0, anything under a
   `### Breaking` heading in `CHANGELOG.md` means a **minor** bump
   (`0.2.x` → `0.3.0`), not a patch. Patch releases are for changes with no
   Breaking entries.
2. **Bump it everywhere** — these are not linked to each other, and nothing
   fails if they drift:
   - `version` in `pyproject.toml` — the real one. PyPI and conda build from
     it, and it is what `eeo.show_versions()` reports, via the installed
     distribution's metadata.
   - `__version__` in `eeo/__init__.py` — a separate hand-maintained string
     that nothing derives from and no test pins. Users who check
     `eeo.__version__` see this one, so a stale value disagrees with
     `show_versions()` in the same interpreter.
   - `release` in `docs/source/conf.py` (what the rendered docs display)
   - then `uv lock`, which rewrites the project's own version in `uv.lock`.
     Skipping it breaks CI, which installs with `uv sync --frozen`. Check the
     diff is only that one line — if the re-resolve also bumped dependencies,
     that is a separate change and does not belong in a release commit.
3. **Close the changelog section.** Rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD` and open a fresh, empty `## [Unreleased]` above it.
4. **Run the full checks** — `pytest`, `ruff check`, `ruff format --check`,
   `mypy`, and a docs build under `-W`.
5. **Tag and push:**
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

`release.yml` takes it from there: it builds the sdist and wheel, runs
`twine check`, publishes to PyPI via trusted publishing (no API token), and
triggers a Read the Docs build. Nothing conda-specific happens here.

### 2. conda-forge follows on its own

The conda package is built from a recipe in
[`conda-forge/easy-eo-feedstock`](https://github.com/conda-forge/easy-eo-feedstock),
a separate repository. conda users never see this one — they get an artifact
built from that recipe.

`regro-cf-autotick-bot` watches PyPI and opens a PR on the feedstock, usually
within a few hours, updating three lines of `recipe/recipe.yaml`: `version`,
`sha256`, and `build.number` back to `0`. Check its CI is green and merge —
maintainers can merge their own feedstock PRs, and the upload to the
conda-forge channel happens automatically. The package appears on anaconda.org
within roughly 15–60 minutes, plus another 10–20 for CDN sync before
`conda install` resolves it.

If no PR has appeared after a day, ask the bot directly. This one is **not** a
comment: open a new issue on the feedstock whose **title** is exactly

```
@conda-forge-admin, please update version
```

The body can be empty. The bot reads the title, checks PyPI for a newer
version, and opens the PR if it finds one. Posting the phrase as a comment
instead does nothing.

### 3. If the release changed dependencies, edit the bot's PR

**The bot never reads `pyproject.toml`.** It compares sdist hashes. So for a
release that adds, removes, or re-bounds a dependency, its PR builds a package
carrying the *old* dependency list — and `conda install easy-eo` then produces
a broken environment, with nothing warning you. This is the reason to read the
bot's PR rather than rubber-stamp it. Edit `recipe/recipe.yaml` in that PR
before merging:

| Changed in `pyproject.toml`                     | Edit in `recipe/recipe.yaml`      |
| ----------------------------------------------- | --------------------------------- |
| `dependencies` — added, removed, rebounded      | `requirements.run`                |
| `requires-python`                               | `python_min` usage in `host`/`run` |
| `[build-system] requires`                       | `requirements.host`               |
| A new public import worth smoke-testing         | `tests.python.imports`            |
| `[project.optional-dependencies]`               | nothing — conda has no extras     |

Two things to know before editing that list:

- **conda names are not always PyPI names.** The recipe depends on
  `matplotlib-base`, *not* `matplotlib`: the plain package pulls in Qt and a
  GUI stack a library has no use for. Keep it that way.
- **Check a new dependency exists on conda-forge at all**, or the release
  cannot ship there:
  ```bash
  curl -s https://api.anaconda.org/package/conda-forge/<name> | head -c 300
  ```
- **`requires-python` is not pinned locally.** The recipe uses
  `${{ python_min }}`, which comes from conda-forge's global configuration, and
  today that matches our `>=3.10`. If you ever raise the minimum ahead of
  conda-forge's global bump, set it explicitly in the recipe — otherwise
  conda-forge keeps building for a Python the code no longer supports. Note the
  ceiling is unpinned in both directions: with no upper bound in
  `requires-python`, conda will resolve the newest Python available, which can
  be one the CI matrix does not test.

Optional extras need no recipe change — conda has no extras mechanism, so
their packages are documented as ordinary installs instead (see
[Getting Started](https://easy-eo.readthedocs.io/en/latest/getting_started.html)).
What a *new* extra does need is an entry in `_CONDA_PACKAGES` in
`eeo/_optional.py`, so the missing-dependency error can name a command a conda
user can actually run; a test fails if one is missing.

### 4. Fixing a bad package without cutting a release

A conda artifact is identified by name + version + build string, and a
published file cannot be overwritten. So when the code is fine but the
*packaging* is wrong — a runtime dependency missing from `requirements.run`, a
bound that turns out to break — fix the recipe and bump `build.number`
(`0` → `1`) in the same PR. Same version, same `sha256`, new artifact; the
solver prefers the highest build number, so fresh installs get the fix. A
version bump resets it to `0` by itself.

### 5. Feedstock files you must never hand-edit

`conda-smithy` generates these and will silently overwrite your changes:

- `.github/workflows/conda-build.yml`
- `.ci_support/`
- `.scripts/`
- the feedstock's own `README.md`

The only files edited by hand are `recipe/recipe.yaml` and `conda-forge.yml`.
To regenerate the rest, comment on a feedstock **pull request**:

```
@conda-forge-admin, please rerender        # after a smithy or pinning update
@conda-forge-admin, please restart ci      # a stuck build
```

`please rerender` pushes the regenerated files straight to that PR, so tick
*Allow edits from maintainers* on it first. Posted as a comment on an **issue**
instead, it opens a fresh PR containing the rerender. `please restart ci` works
only on a PR — it closes and reopens it to re-trigger the builds. Note the
different mechanism for `please update version` in step 2: that one goes in an
issue *title*, not a comment.

Finally: **work through PRs, not pushes to the feedstock's `main`.** A push
there builds and uploads immediately, so a mistake ships to users. Bots will
also open migration PRs of their own — new Python versions, numpy pin bumps,
ABI rebuilds. Those are not about this codebase: check CI is green and merge.

Fuller detail lives in the
[conda-forge maintainer docs](https://conda-forge.org/docs/maintainer/).

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
