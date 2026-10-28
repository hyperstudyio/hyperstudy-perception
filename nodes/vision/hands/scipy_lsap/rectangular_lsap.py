"""scipy.optimize.linear_sum_assignment (minimisation), transcribed line for line from SciPy's
scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (BSD-3-Clause, LICENSE; UPSTREAM.md names the commit).

The shortest augmenting path algorithm for the rectangular assignment problem, after DF Crouse, "On implementing 2D
rectangular assignment algorithms", IEEE Trans. Aerospace and Electronic Systems 52(4):1679-1696, 2016. Author of
the C++: PM Larsen.

Transcribed rather than imported so that the hands node's matching stays numpy-only (tested without SciPy) while
choosing exactly what SciPy chooses, ties included: the bake-off matched detector boxes to wrists with SciPy, and two
wrists inside one box cost the same (0).
"""

from __future__ import annotations

import math

import numpy as np


def _augmenting_path(nc, cost, u, v, path, row4col, shortest, i, SR, SC, remaining):
    min_val = 0.0
    num_remaining = nc
    for it in range(nc):
        # filled in reverse order so that a constant cost matrix gives the identity (scipy #11602)
        remaining[it] = nc - it - 1
    for k in range(len(SR)):
        SR[k] = False
    for k in range(nc):
        SC[k] = False
        shortest[k] = math.inf

    sink = -1
    while sink == -1:
        index = -1
        lowest = math.inf
        SR[i] = True
        for it in range(num_remaining):
            j = remaining[it]
            r = min_val + cost[i][j] - u[i] - v[j]
            if r < shortest[j]:
                path[j] = i
                shortest[j] = r
            # of several nodes with the minimum cost, one that gives a new sink
            if shortest[j] < lowest or (shortest[j] == lowest and row4col[j] == -1):
                lowest = shortest[j]
                index = it
        min_val = lowest
        if min_val == math.inf:   # infeasible cost matrix
            return -1, min_val
        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]
        SC[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]
    return sink, min_val


def linear_sum_assignment(cost_matrix) -> tuple[np.ndarray, np.ndarray]:
    """(row indices, column indices) of a minimum-cost assignment, as scipy.optimize.linear_sum_assignment returns
    them (rows ascending; with more rows than columns, ordered by column)."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"expected a matrix (2-D array), got a {cost.ndim}-D array")
    nr, nc = cost.shape
    if nr == 0 or nc == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    transpose = nc < nr   # a tall matrix is solved transposed
    if transpose:
        cost = cost.T
        nr, nc = nc, nr
    if np.isnan(cost).any() or (cost == -math.inf).any():
        raise ValueError("matrix contains invalid numeric entries")
    c = cost.tolist()   # Python floats: the same IEEE doubles, in the same order of operations as the C++

    u, v = [0.0] * nr, [0.0] * nc
    shortest = [0.0] * nc
    path, col4row, row4col = [-1] * nc, [-1] * nr, [-1] * nc
    SR, SC, remaining = [False] * nr, [False] * nc, [0] * nc

    for cur_row in range(nr):
        sink, min_val = _augmenting_path(nc, c, u, v, path, row4col, shortest, cur_row, SR, SC, remaining)
        if sink < 0:
            raise ValueError("cost matrix is infeasible")
        # update the dual variables
        u[cur_row] += min_val
        for i in range(nr):
            if SR[i] and i != cur_row:
                u[i] += min_val - shortest[col4row[i]]
        for j in range(nc):
            if SC[j]:
                v[j] -= min_val - shortest[j]
        # augment the previous solution
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur_row:
                break

    if transpose:
        order = sorted(range(nr), key=lambda k: col4row[k])   # col4row's values are distinct: no ties to break
        return np.array([col4row[k] for k in order], dtype=np.int64), np.array(order, dtype=np.int64)
    return np.arange(nr, dtype=np.int64), np.array(col4row, dtype=np.int64)
