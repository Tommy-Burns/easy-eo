# Easy-EO Code Design & Style Guide

This document defines the design principles, coding style, and architectural
rules used throughout Easy-EO. All code contributions — human-written or
AI-assisted — are expected to follow these guidelines.

The goal is to maintain a **clear, performant, and predictable** geospatial
processing library.

---

## Naming & Branding

Consistency matters for discoverability and trust. Use exactly:

- **Easy-EO** — display name in prose, docs, and headings
- **easy-eo** — PyPI / conda package name, repo name, URLs
- **eeo** — import name (`import eeo`) and code references

Never use "EasyEO" or other variants. If you find one, fix it.

---

## Core Design Philosophy

Easy-EO provides **high-level abstractions** over Rasterio, NumPy, and
Matplotlib without hiding geospatial realities.

Code should be:
- Explicit rather than implicit
- Correct before clever
- Optimized for large raster data
- Easy to reason about and debug
- **Interoperable, not isolating** — users must always be able to drop down
  to NumPy, Rasterio, or xarray without friction (`to_array()`, `.ds`,
  `to_rasterio()`, and `to_xarray()` once available)

Target user: GIS analysts, students, and researchers who know some Python and
want common EO workflows in a few readable lines. When in doubt, optimize the
API for that person, not for library authors.

---

## Supported Environments

- Python **3.10+** (test 3.10 through the latest stable release)
- Linux, macOS, and Windows (CI must cover all three)
- Core dependencies carry lower and upper bounds in `pyproject.toml`

---

## Tooling (Required)

- **Formatting & linting:** `ruff` (format + lint). No style debates in
  review; the tool decides.
- **Type checking:** `mypy` on the `eeo` package; new code must pass.
- **Pre-commit:** hooks for ruff, mypy, and whitespace must pass before any
  commit.
- **Tests:** `pytest` with coverage reporting; target ≥ 80% line coverage on
  `eeo/`.

All of the above run in CI on every PR. A PR that fails any of them is not
mergeable.

---

## Chainability Rules

### Chainable Operations

Functions decorated with `@eeo_raster_op` must:

- Accept an `EEORasterDataset` as the first argument
- Return an `EEORasterDataset` for in-memory results
- Preserve CRS, transform, nodata, and metadata unless the operation's
  documented purpose is to change them
- Never mutate the input dataset — always return a new dataset

### Terminal Operations

Functions decorated with `@eeo_raster_viz` are terminal operations:

- They **must return `None`**
- They **must not modify the dataset**
- They **must not create new raster datasets**
- They may accept multiple datasets for visualization

---

## API Stability & Consistency

These rules exist because signature/documentation mismatches destroy user
trust faster than missing features.

- **Signature defaults, `.pyi` stubs, docstrings, and narrative docs must all
  agree.** If `resample()` defaults to one resampling method, every mention
  of it everywhere says the same thing. A mismatch is a release-blocking bug.
- **Defaults must fit the common case.** e.g. percentile normalization
  defaults to the conventional `(2, 98)`, not `(0, 1)`.
- **Semantic versioning.** Pre-1.0, breaking changes are allowed but must be
  listed in `CHANGELOG.md` under a "Breaking" heading. Post-1.0, breaking
  changes require a deprecation cycle (`DeprecationWarning` for at least one
  minor release before removal).
- **All examples in docs and README must be valid, runnable Python.**
  Multi-line chains must be wrapped in parentheses:

  ```python
  result = (
      ds_nir.clip_raster_with_bbox((0, 0, 1000, 1000))
      .resample(scale_factor=2)
      .normalized_difference(ds_red)
      .multiply(100)
  )
  ```

- New public API requires: type hints, a docstring with a runnable example,
  tests, and an entry in the docs API reference. No exceptions.

---

## Raster Metadata Rules

All raster operations must respect:

- CRS
- Affine transform
- Width, height, and band count
- Nodata values
- Data types

If an operation changes any of the above, it must:

- Be explicit
- Be documented
- Update metadata consistently

---

## Nodata & Dtype Propagation

Every operation must have defined, documented, and tested behavior for:

- **Nodata:** masked before computation, restored (or set) in the output, and
  the output's nodata value documented. Nodata pixels must never leak into
  statistics or arithmetic results.
- **Dtype:** state the output dtype in the docstring. Operations producing
  fractional results (indices, normalization, division) return float32 by
  default unless documented otherwise. Never silently truncate to integer
  dtypes.
- **NaN vs nodata:** be explicit about which the output uses, and why.

If you cannot state a function's nodata/dtype behavior in one sentence, the
function is not finished.

---

## Alignment & Reprojection

- Raster alignment must be **explicit**
- Auto-alignment must be controlled via `auto_align=True`
- Reprojection must:
  - Use Rasterio's reprojection utilities
  - Preserve spatial accuracy
  - Clearly document resampling methods

No silent reprojection or resampling. When `auto_align=True` triggers an
alignment, emit a `logging` info message stating what was aligned and with
which resampling method.

---

## Numerical Safety

Raster arithmetic must:

- Handle division by zero safely
- Avoid uncontrolled NaNs or infinities
- Use NumPy error state contexts (`np.errstate`) when necessary

If a function changes the dtype, document it clearly.

---

## Performance & Memory

Easy-EO must scale beyond toy rasters. Real Sentinel-2 / Landsat scenes do
not always fit comfortably in memory.

- Avoid repeated reads of the same band
- Use vectorized NumPy operations; no Python-level pixel loops
- Minimize memory copies
- Prefer **windowed reads** where an operation can work block-wise; do not
  assume the whole raster fits in memory when a windowed implementation is
  practical
- Backend-specific logic belongs in adapters (`eeo/core/adapters/`), never
  in operation functions. This keeps the door open for future lazy or
  Dask-backed adapters.

---

## Function Design

Functions should:

- Do **one thing**
- Have a clear and predictable API
- Validate inputs early
- Raise meaningful exceptions
- Avoid side effects

Keyword-only arguments (`*`) are encouraged for all optional parameters.

---

## Error Handling

- Fail early and loudly
- Never silently ignore errors
- Avoid broad `except Exception`
- Provide actionable error messages, e.g. "expected bbox as (minx, miny,
  maxx, maxy) in the raster's CRS; got 3 values"
- Use a small hierarchy of custom exceptions (e.g. `EEOError`,
  `CRSMismatchError`, `AlignmentError`, `BackendError`) so users can catch
  precisely what they need

---

## Docstrings

There is exactly **one** docstring style: **NumPy style** (Sphinx napoleon /
numpydoc), enforced mechanically by ruff's pydocstyle rules (numpy
convention) and numpydoc validation in CI. Docstrings live only in
implementation `.py` files — never in `.pyi` stubs.

Every public function/method docstring follows this template:

```python
def normalize_percentile(ds, *, lower_percentile=2, upper_percentile=98):
    """Normalize raster values using percentile thresholds.

    Values outside the percentile range are clipped; remaining values are
    scaled to [0, 1]. Robust to outliers.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    lower_percentile : float, default 2
        Lower percentile threshold (0–100).
    upper_percentile : float, default 98
        Upper percentile threshold (0–100).

    Returns
    -------
    EEORasterDataset
        New dataset with float32 values in [0, 1]. Nodata pixels are
        excluded from percentile computation and preserved in the output.

    Raises
    ------
    ValidationError
        If ``lower_percentile >= upper_percentile``.

    Notes
    -----
    Streams block-wise; requires one statistics pass before writing.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> out = ds.normalize_percentile(lower_percentile=5, upper_percentile=95)
    """
```

Required in every public docstring:

- One-line summary in the imperative mood, then a short extended description
- ``Parameters`` with types, defaults, and units/coordinate conventions
  (CRS units vs pixel indices; 1-based band indexing)
- ``Returns`` stating the type **and** output dtype and nodata behavior
- ``Raises`` listing the custom exceptions that can be raised
- ``Notes`` stating memory behavior when it is anything other than
  "streams block-wise", and any side effects
- ``Examples`` with at least one runnable example

Any docstring not following this template is a bug, regardless of when it
was written.

---

## Type Hints

- All public APIs must use type hints
- Prefer concrete types over `Any`
- Use modern syntax (`int | float`, `X | None`) — the package requires
  Python 3.10+
- Use `| None` only when `None` is meaningful

Type information lives **inline in the implementation files**, distributed
via a `py.typed` marker (PEP 561). Per-module `.pyi` stubs that duplicate
inline annotations are not allowed. The single exception is
`eeo/core/core.pyi`, which declares the operation methods dynamically bound
to `EEORasterDataset` (invisible to type checkers otherwise); it is
generated from the decorator registry by a script and checked for freshness
in CI — never edited by hand. Documentation never goes in stub files.

---

## Testing Expectations

- New features must include tests; bug fixes must include a regression test
- Test files follow pytest's default discovery convention: `tests/test_*.py`
- Edge cases must be considered: single-band vs multi-band, nodata-heavy
  rasters, dtype boundaries (uint8 / uint16 / float32), CRS mismatches,
  empty clip results
- Numerical behavior must be tested against known expected values, not just
  "runs without error"
- Tests use **small synthetic rasters generated in fixtures** (`load_array`
  plus in-memory Rasterio). The default test run must not download data or
  depend on files outside the repo
- Visualization tests use Matplotlib's `Agg` backend and assert on figure
  structure (axes count, titles), not rendered pixels

---

## Commits, Branches & Releases

- Conventional commit prefixes are encouraged: `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `ci:`, `chore:`
- Every user-visible change gets a `CHANGELOG.md` entry under "Unreleased"
- Releases are tagged `vX.Y.Z` and published to PyPI via CI (trusted
  publishing); the same version should follow to conda-forge

---

## Dependency Policy

Easy-EO is a library, not an application. Runtime dependencies must never be
pinned exactly — consistency comes from a lockfile for development, and
tested ranges for users.

- **Runtime ranges (`pyproject.toml`):** lower bound = oldest version
  actually tested in CI; upper bound only where breakage is known or likely
  (e.g. capping below a major version with a known API break). Never cap to
  a single minor version without a documented reason.
- **Lockfile for dev/CI:** a committed `uv.lock` defines the exact
  environment for contributors and CI. Update it via dedicated bump PRs
  (Dependabot/Renovate), not ad hoc.
- **Ranges must be verified:** CI includes a minimum-versions job (installs
  the lower bounds) and a scheduled latest-versions job (ignores the
  lockfile). A claim of `numpy>=1.26` support that is never tested is not a
  supported range.
- **Optional extras** (`[stac]`, `[xarray]`, `[lazy]`, `[dev]`) carry their
  own bounds and are tested both installed and absent (the absent case must
  raise the documented, helpful ImportError).
- Any change to dependency bounds gets a `CHANGELOG.md` entry and a note in
  the PR explaining why.
- `eeo.show_versions()` must stay accurate: when adding a core or optional
  dependency, add it to the report.

---

## AI-Assisted Code

AI tools (including Claude Code) may be used to assist development, but:

- Contributors must fully understand the code
- Generated code must be reviewed, optimized, and validated
- Generated code is held to every rule in this document — especially the
  nodata/dtype, testing, and API-consistency rules
- Submitting unreviewed AI-generated code is not acceptable
- AI tools never create commits, branches, tags, or pushes — they prepare
  changes in the working tree and propose commit messages; the maintainer
  makes every commit personally after review

---

## Summary

This guide exists to protect:

- Code quality
- Performance
- Geospatial correctness
- API trust and long-term maintainability

If in doubt, prioritize clarity and correctness.
