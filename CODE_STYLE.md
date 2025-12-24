# Easy-EO Code Design & Style Guide

This document defines the design principles, coding style, and architectural
rules used throughout Easy-EO. All code contributions are expected to follow
these guidelines.

The goal is to maintain a **clear, performant, and predictable** geospatial
processing library.

---

## Core Design Philosophy

Easy-EO provides **high-level abstractions** over Rasterio, NumPy, and Matplotlib
without hiding geospatial realities.

Code should be:
- Explicit rather than implicit
- Correct before clever
- Optimized for large raster data
- Easy to reason about and debug

---

## Chainability Rules

### Chainable Operations

Functions decorated with `@eeo_raster_op` must:

- Accept an `EEORasterDataset` as the first argument
- Return:
  - `EEORasterDataset` for in-memory results
- Preserve CRS, transform, nodata, and metadata

### Terminal Operations

Functions decorated with `@eeo_raster_viz` are terminal operations:

- They **must return `None`**
- They **must not modify the dataset**
- They **must not create new raster datasets**
- They may accept multiple datasets for visualization

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

## Alignment & Reprojection

- Raster alignment must be **explicit**
- Auto-alignment must be controlled via `auto_align=True`
- Reprojection must:
  - Use Rasterio’s reprojection utilities
  - Preserve spatial accuracy
  - Clearly document resampling methods

No silent reprojection or resampling.

---

## Numerical Safety

Raster arithmetic must:

- Handle division by zero safely
- Avoid uncontrolled NaNs or infinities
- Use NumPy error state contexts when necessary

If a function changes the dtype, document it clearly.

---

## Performance Guidelines

- Avoid repeated reads of the same band
- Use vectorized NumPy operations
- Minimize memory copies

---

## Function Design

Functions should:

- Do **one thing**
- Have a clear and predictable API
- Validate inputs early
- Raise meaningful exceptions
- Avoid side effects

Keyword-only arguments (`*`) is encouraged.

---

## Docstrings

- All public functions must have docstrings
- Use consistent parameter descriptions
- Clearly document:
  - Units
  - Return types
  - Side effects

---

## Type Hints

- All public APIs must use type hints
- Prefer concrete types over `Any`
- Use `Union` only when necessary
- Use `Optional[T]` only when `None` is meaningful

`.pyi` stub files may be used for detailed documentation, but runtime behavior
must remain obvious from the corresponding implementation file.

---

## Error Handling

- Fail early and loudly
- Never silently ignore errors
- Avoid broad `except Exception`
- Provide actionable error messages

---

## Testing Expectations

- New features should include tests
- Edge cases must be considered
- Numerical behavior should be tested explicitly
- Tests should not rely on external data unless unavoidable

---

## AI-Assisted Code

AI tools may be used to assist development, but:

- Contributors must fully understand the code
- Generated code must be reviewed, optimized, and validated
- Submitting unreviewed AI-generated code is not acceptable

---

## Summary

This guide exists to protect:

- Code quality
- Performance
- Geospatial correctness
- Long-term maintainability

If in doubt, prioritize clarity and correctness.
