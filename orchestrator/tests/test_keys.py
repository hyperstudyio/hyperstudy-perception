"""The user's keys (keys.py, `perception keys`): the environment over the keychain over private files; a file others
can read is refused; values are never printed or written anywhere but the keychain or a 0600 file. An in-memory
keychain stands in for the system's (conftest keeps the real one out of every test)."""

import json
import os
import stat
import subprocess
import sys

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from perception import cli, keys as keys_mod
from perception.keys import Keys

SECRET = "sk-test-0123456789abcdef"


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise PasswordDeleteError(username)
        del self.store[(service, username)]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in [k for k in os.environ if k.endswith(("_KEY", "_TOKEN"))]:
        monkeypatch.delenv(k)
    conf = tmp_path / ".config" / "hyperstudy"
    conf.mkdir(parents=True)
    return conf


@pytest.fixture
def chain(monkeypatch):
    monkeypatch.delenv("PERCEPTION_NO_KEYCHAIN")
    kr = MemoryKeyring()
    previous = keyring.get_keyring()
    keyring.set_keyring(kr)
    yield kr
    keyring.set_keyring(previous)


def env_file(path, text, mode=0o600):
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_environment_wins_over_the_keychain_over_a_private_file(home, chain, monkeypatch):
    env_file(home / "a.env", "ONE_KEY=from-file\nTWO_KEY=from-file\nTHREE_KEY=from-file\n")
    Keys().set("TWO_KEY", "from-keychain")
    Keys().set("THREE_KEY", "from-keychain")
    monkeypatch.setenv("THREE_KEY", "from-environment")
    found = Keys().load()
    assert found.values == {"ONE_KEY": "from-file", "TWO_KEY": "from-keychain", "THREE_KEY": "from-environment"}
    assert found.sources == {"ONE_KEY": str(home / "a.env"), "TWO_KEY": "keychain", "THREE_KEY": "environment"}


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660])
def test_a_file_others_can_read_or_write_is_refused_and_said_so(home, mode):
    env_file(home / "openai.env", f"OPENAI_API_KEY={SECRET}\n", mode)
    found = Keys().load()
    assert "OPENAI_API_KEY" not in found.values
    assert len(found.problems) == 1 and "chmod 600" in found.problems[0] and "openai.env" in found.problems[0]


def test_a_link_in_place_of_a_key_file_is_refused(home, tmp_path):
    target = env_file(tmp_path / "elsewhere.env", f"X_KEY={SECRET}\n")
    (home / "x.env").symlink_to(target)
    found = Keys().load()
    assert "X_KEY" not in found.values and "not a regular file" in found.problems[0]


def test_set_stores_in_the_keychain_and_keeps_only_names_on_disk(home, chain):
    assert Keys().set("RUNPOD_API_KEY", f"  {SECRET}\n") == "keychain"
    assert chain.store[("hyperstudy-perception", "RUNPOD_API_KEY")] == SECRET   # stripped
    index = home / "keys.json"
    assert json.loads(index.read_text()) == ["RUNPOD_API_KEY"] and SECRET not in index.read_text()
    assert stat.S_IMODE(index.stat().st_mode) == 0o600
    assert Keys().load().values["RUNPOD_API_KEY"] == SECRET
    assert Keys().remove("RUNPOD_API_KEY") == ["keychain"]
    assert not chain.store and json.loads(index.read_text()) == [] and "RUNPOD_API_KEY" not in Keys().load().values


def test_without_a_keychain_set_writes_a_file_only_the_user_can_read(home):
    where = Keys().set("HF_TOKEN", SECRET)
    path = home / "keys.env"
    assert where == str(path) and path.read_text() == f"HF_TOKEN={SECRET}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert Keys().load().values["HF_TOKEN"] == SECRET
    assert Keys().remove("HF_TOKEN") == [str(path)] and path.read_text() == ""


def test_a_new_config_folder_is_made_private(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    Keys().set("HF_TOKEN", SECRET)
    assert stat.S_IMODE((tmp_path / ".config" / "hyperstudy").stat().st_mode) == 0o700


@pytest.mark.parametrize("name, value", [("runpod_key", SECRET), ("RUN POD", SECRET), ("OK_KEY", ""), ("OK_KEY", "a\nb")])
def test_set_refuses_a_bad_name_or_value(home, name, value):
    with pytest.raises(ValueError):
        Keys().set(name, value)


def test_the_cli_reads_a_value_from_stdin_and_never_prints_it(home, chain, capsys, monkeypatch):
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO(SECRET + "\n"))
    assert cli.main(["keys", "set", "RUNPOD_API_KEY"]) == 0
    env_file(home / "loose.env", "Z_KEY=z\n", 0o644)
    assert cli.main(["keys", "list"]) == 0
    assert cli.main(["--json", "keys", "list"]) == 0
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    listed = json.loads(out.out.strip().splitlines()[-1])
    assert listed["keys"] == [{"name": "RUNPOD_API_KEY", "source": "keychain"}]
    assert "loose.env" in listed["problems"][0]
    assert "RUNPOD_API_KEY               keychain" in out.out


def test_the_cli_takes_no_value_on_the_command_line(home):
    with pytest.raises(SystemExit):
        cli.main(["keys", "set", "RUNPOD_API_KEY", SECRET])


def test_refused_files_are_said_once_on_stderr(home, capsys, monkeypatch):
    monkeypatch.setattr(keys_mod, "_warned", False)
    env_file(home / "openai.env", f"OPENAI_API_KEY={SECRET}\n", 0o644)
    assert "OPENAI_API_KEY" not in keys_mod.load_secrets()
    keys_mod.load_secrets()
    err = capsys.readouterr().err
    assert err.count("openai.env can be read by others") == 1 and SECRET not in err


def test_a_node_gets_a_key_from_the_keychain_and_only_the_keys_it_declares(home, chain):
    from perception.backends.base import child_env
    Keys().set("ANTHROPIC_API_KEY", SECRET)
    Keys().set("RUNPOD_API_KEY", "rp-other")
    env = child_env(["ANTHROPIC_API_KEY"], keys_mod.load_secrets())
    assert env["ANTHROPIC_API_KEY"] == SECRET and "RUNPOD_API_KEY" not in env
