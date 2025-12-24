# Contributing to Easy-EO

Thank you for your interest in contributing to **Easy-EO** 🎉  
Contributions of all kinds are welcome and appreciated.

Easy-EO aims to provide **high-level, chainable abstractions** over common
earth-observation workflows while remaining **correct, performant, and
transparent**. This document outlines how you can contribute and what
standards we expect.

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

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
3. Make your changes
4. Run tests and lint checks (if applicable)
5. Open a pull request with:
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
