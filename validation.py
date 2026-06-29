"""Small, shared input canonicalizers for the numerical public APIs."""

import numpy as np


__all__ = ["boolean", "complex_array", "integer", "normalized_weights", "real", "real_array"]


def real(
    name,
    value,
    *,
    minimum=None,
    maximum=None,
    strict_minimum=False,
    strict_maximum=False,
):
    """Return a finite real scalar satisfying optional bounds."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc

    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number.")

    value = float(array)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")

    if minimum is not None and (
        value <= minimum if strict_minimum else value < minimum
    ):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}.")
    if maximum is not None and (
        value >= maximum if strict_maximum else value > maximum
    ):
        relation = "less than" if strict_maximum else "at most"
        raise ValueError(f"{name} must be {relation} {maximum}.")
    return value


def integer(name, value, *, minimum=None, maximum=None):
    """Return an integer satisfying optional inclusive bounds."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer.")

    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def boolean(name, value):
    """Return an exact Boolean value."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _shape_matches(actual, expected):
    return len(actual) == len(expected) and all(
        wanted is None or got == wanted for got, wanted in zip(actual, expected)
    )


def _array(name, value, dtype, kinds, shape, ndim, nonempty, copy):
    contents = "real numbers" if kinds == "iuf" else "numeric values"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite {contents}.") from exc

    if array.dtype.kind not in kinds:
        raise ValueError(f"{name} must contain only finite {contents}.")

    array = np.asarray(array, dtype=dtype)
    if copy:
        array = array.copy()
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if shape is not None and not _shape_matches(array.shape, shape):
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite {contents}.")
    return array


def real_array(name, value, *, shape=None, ndim=None, nonempty=True, copy=False):
    """Return a finite floating-point array with optional shape constraints."""
    return _array(
        name, value, float, "iuf", shape, ndim, nonempty, copy
    )


def complex_array(name, value, *, shape=None, ndim=None, nonempty=True, copy=False):
    """Return a finite complex array with optional shape constraints."""
    return _array(
        name, value, np.complex128, "iufc", shape, ndim, nonempty, copy
    )


def normalized_weights(weights, count):
    """Return finite, non-negative quadrature weights normalized stably."""
    count = integer("count", count, minimum=1)
    if weights is None:
        return np.full(count, 1.0 / count)

    weights = real_array("weights", weights, shape=(count,))
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")

    scale = float(np.max(weights))
    if scale == 0.0:
        raise ValueError("weights must have a positive total.")
    scaled = weights / scale
    return scaled / np.sum(scaled)
