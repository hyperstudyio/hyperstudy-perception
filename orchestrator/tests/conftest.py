import pytest


@pytest.fixture(autouse=True)
def no_real_keychain(monkeypatch):
    """No test reads or writes the user's own keychain (keys.py): the processes tests start inherit this too.
    test_keys.py puts an in-memory keychain in its place where it needs one."""
    monkeypatch.setenv("PERCEPTION_NO_KEYCHAIN", "1")
