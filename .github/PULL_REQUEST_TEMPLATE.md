<!--
Thanks for contributing to Easy-EO! Keep PRs focused on a single task where possible.
-->

## Summary

<!-- What does this PR change, and why?  (e.g. "feat: add NDMI index"). -->

## Related issues / tasks

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (documented under a "Breaking" heading in CHANGELOG)
- [ ] Docs / tooling / CI only

## Checklist (Definition of Done)

- [ ] `pytest` passes and coverage stays at or above the threshold
- [ ] `ruff check` and `ruff format --check` pass
- [ ] `mypy` passes
- [ ] New/changed public functions have NumPy-style docstrings with a runnable
      example, per `CODE_STYLE.md`
- [ ] Nodata and dtype behavior is stated in the docstring and covered by a test
- [ ] Bound ops changed? Regenerated `eeo/core/core.pyi`
      (`python scripts/generate_core_stub.py`)
- [ ] Docs API reference updated where relevant
- [ ] `CHANGELOG.md` updated under "Unreleased"

## Notes for the reviewer

<!-- Anything the maintainer should pay special attention to, verify manually, or be aware of (e.g. dependency bound changes, memory behavior). -->
