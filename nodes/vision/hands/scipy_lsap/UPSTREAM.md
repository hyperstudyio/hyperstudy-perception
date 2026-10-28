SciPy's `linear_sum_assignment`, transcribed into Python from
https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
(release v1.18.1, commit e4e854eaa8f18d807cd3496028e257e36caa93cc; BSD-3-Clause, `LICENSE`; the C++ by PM Larsen,
after DF Crouse 2016). SciPy 1.18.1 is the version the bake-off's hands stage ran (tools/perception-bakeoff/hands/uv.lock).

`rectangular_lsap.py` follows `augmenting_path` and `solve` statement for statement, with Python floats (IEEE
doubles) in the same order of operations, so it picks the same assignment as SciPy, ties included. Only minimisation
is transcribed (the bake-off never maximised). Errors are raised as ValueError, as SciPy's wrapper does.

Why not SciPy itself: the hands node's matching lives in a numpy-only module, tested in CI without SciPy
(`test_rectangular_lsap.py` checks it against brute force there, and against SciPy where SciPy is installed).
