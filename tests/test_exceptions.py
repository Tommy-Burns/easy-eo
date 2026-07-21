"""Tests for the Easy-EO exception hierarchy.

These cover only the class relationships: that every domain error derives from
``EEOError``, keeps its historical built-in base for backward compatibility,
and is exported at the top level. Tests asserting that operations raise the
right types live alongside those operations' tests.
"""

import pytest

import eeo
from eeo.core.exceptions import (
    AlignmentError,
    BackendError,
    CRSMismatchError,
    EEOError,
    ValidationError,
)

DOMAIN_ERRORS = [ValidationError, CRSMismatchError, AlignmentError, BackendError]
VALUE_ERRORS = [ValidationError, CRSMismatchError, AlignmentError]


def test_eeoerror_is_plain_exception():
    assert issubclass(EEOError, Exception)
    assert not issubclass(EEOError, (ValueError, RuntimeError))


@pytest.mark.parametrize("exc", DOMAIN_ERRORS)
def test_all_domain_errors_derive_from_eeoerror(exc):
    assert issubclass(exc, EEOError)


@pytest.mark.parametrize("exc", VALUE_ERRORS)
def test_value_error_backward_compat(exc):
    assert issubclass(exc, ValueError)
    with pytest.raises(ValueError):
        raise exc("boom")


def test_backend_error_runtime_compat():
    assert issubclass(BackendError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise BackendError("boom")


@pytest.mark.parametrize("exc", [*DOMAIN_ERRORS, EEOError])
def test_every_error_catchable_as_eeoerror(exc):
    with pytest.raises(EEOError):
        raise exc("boom")


def test_backend_error_is_not_value_error():
    # BackendError must stay a RuntimeError, never a ValueError, so callers
    # can distinguish "bad input" from "backend can't do this".
    assert not issubclass(BackendError, ValueError)


def test_exceptions_exposed_at_top_level():
    assert eeo.EEOError is EEOError
    assert eeo.ValidationError is ValidationError
    assert eeo.CRSMismatchError is CRSMismatchError
    assert eeo.AlignmentError is AlignmentError
    assert eeo.BackendError is BackendError
