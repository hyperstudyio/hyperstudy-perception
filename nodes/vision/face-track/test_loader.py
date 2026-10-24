import pytest

from loader import decode_workers


def test_automatic_means_four_where_workers_fork_and_none_where_they_spawn():
    assert decode_workers(None, "fork", cpus=8) == 4
    assert decode_workers(None, "fork", cpus=2) == 1      # leave a core for the detector's own process
    assert decode_workers(None, "spawn", cpus=8) == 0     # macOS: a spawned worker would run the node again


def test_an_explicit_count_is_kept_but_refused_where_workers_spawn():
    assert decode_workers(0, "spawn", cpus=8) == 0
    assert decode_workers(7, "fork", cpus=8) == 7
    with pytest.raises(ValueError, match="needs forked workers"):
        decode_workers(2, "spawn", cpus=8)
