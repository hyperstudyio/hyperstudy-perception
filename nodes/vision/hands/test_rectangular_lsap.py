"""The transcribed linear_sum_assignment (scipy_lsap/), against brute force and, where SciPy is installed, SciPy itself.
Run: uv run --no-project --with pytest --with numpy --with-editable ./sdk pytest nodes/vision/hands"""

import itertools

import numpy as np
import pytest

from scipy_lsap.rectangular_lsap import linear_sum_assignment


def brute_force(cost) -> float:
    """The minimum total cost over every assignment of the smaller side."""
    nr, nc = cost.shape
    if nr <= nc:
        return min(sum(cost[i, p[i]] for i in range(nr)) for p in itertools.permutations(range(nc), nr))
    return min(sum(cost[p[j], j] for j in range(nc)) for p in itertools.permutations(range(nr), nc))


def matrices(seed=0, n=300):
    """Random matrices up to 5x5, wide and tall, half of them small integers (many ties) as wrist costs are (0 inside)."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        shape = tuple(rng.integers(1, 6, 2))
        yield rng.integers(0, 3, shape).astype(float) if k % 2 else rng.random(shape) * 2


def test_the_assignment_is_optimal_and_complete():
    for cost in matrices():
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == min(cost.shape) and len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
        assert list(rows) == sorted(rows)
        assert cost[rows, cols].sum() == pytest.approx(brute_force(cost))


def test_ties_break_as_scipy_breaks_them():
    """Outputs of scipy 1.18.1's linear_sum_assignment on matrices with tied optima (computed with it)."""
    assert [list(x) for x in linear_sum_assignment(np.zeros((3, 3)))] == [[0, 1, 2], [0, 1, 2]]   # constant: the identity
    assert [list(x) for x in linear_sum_assignment(np.zeros((2, 3)))] == [[0, 1], [0, 1]]
    assert [list(x) for x in linear_sum_assignment(np.zeros((3, 2)))] == [[0, 1], [0, 1]]
    # two wrists inside one box (cost 0 for both), a second box far from both
    assert [list(x) for x in linear_sum_assignment(np.array([[0.0, 0.0], [0.7, 0.9]]))] == [[0, 1], [1, 0]]
    assert [list(x) for x in linear_sum_assignment(np.array([[0.0, 0.0, 0.0]]))] == [[0], [0]]
    assert [list(x) for x in linear_sum_assignment(np.array([[0.0], [0.0], [0.0]]))] == [[0], [0]]


def test_it_agrees_with_scipy_ties_included():
    scipy = pytest.importorskip("scipy.optimize")
    for cost in matrices(seed=1, n=2000):
        assert [list(x) for x in linear_sum_assignment(cost)] == [list(x) for x in scipy.linear_sum_assignment(cost)]


def test_empty_and_invalid_matrices():
    assert [list(x) for x in linear_sum_assignment(np.zeros((0, 3)))] == [[], []]
    with pytest.raises(ValueError):
        linear_sum_assignment(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        linear_sum_assignment(np.array([[np.inf, np.inf]]))   # infeasible
