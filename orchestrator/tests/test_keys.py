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


@pytest.mark.parametrize("name, value", [("runpod_key", SECRET), ("RUN POD", SECRET), ("OK_KEY", ""), ("OK_KEY", "a\nb"),
                                         ("OK_KEY", "abc\rOTHER_KEY=x"), ("OK_KEY", "abc\x85OTHER_KEY=x"), ("OK_KEY", "abc\u2028x")])
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


def test_on_windows_a_key_file_is_not_judged_by_posix_modes(home, monkeypatch):
    # Windows reports 0o666 (0o444 read-only) for every file: no mode can be "private" there, and its profile folder is
    # the user's own; refusing every file would leave keys unreachable, and chmod could not fix it
    monkeypatch.setattr(keys_mod, "POSIX", False)
    env_file(home / "openai.env", f"OPENAI_API_KEY={SECRET}\n", 0o644)
    found = Keys().load()
    assert found.values["OPENAI_API_KEY"] == SECRET and not found.problems


def test_a_config_folder_others_can_write_to_is_refused_whole(home):
    env_file(home / "a.env", f"A_KEY={SECRET}\n")
    home.chmod(0o777)   # anyone could swap a file in there
    found = Keys().load()
    assert "A_KEY" not in found.values and "others can write" in found.problems[0]
    # writing a key makes the user's own folder private first (0700), and then its files are read again
    Keys().set("B_KEY", SECRET)
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert Keys().load().values["A_KEY"] == SECRET


def test_a_config_folder_that_is_a_link_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    env_file(real / "a.env", f"A_KEY={SECRET}\n")
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "hyperstudy").symlink_to(real)
    found = Keys().load()
    assert "A_KEY" not in found.values and "link" in found.problems[0]


def test_an_existing_config_folder_is_made_private_when_a_key_is_written(home):
    home.chmod(0o755)
    Keys().set("HF_TOKEN", SECRET)
    assert stat.S_IMODE(home.stat().st_mode) == 0o700


def test_a_leftover_temporary_file_does_not_loosen_the_key_file(home):
    for name in ("keys.tmp", "keys.env.tmp", "keys.json.tmp"):
        env_file(home / name, "", 0o644)
    Keys().set("HF_TOKEN", SECRET)
    assert stat.S_IMODE((home / "keys.env").stat().st_mode) == 0o600


def test_keys_set_at_the_same_time_all_stay_findable(home, chain):
    import threading
    names = [f"K{i}_KEY" for i in range(16)]
    threads = [threading.Thread(target=Keys().set, args=(n, SECRET)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(json.loads((home / "keys.json").read_text())) == sorted(names)
    assert set(names) <= set(Keys().load().values)


@pytest.mark.parametrize("action", ["set", "remove"])
def test_a_locked_keychain_is_said_plainly_and_nothing_half_done(home, chain, capsys, monkeypatch, action):
    import io
    from keyring.errors import KeyringLocked
    Keys().set("HF_TOKEN", SECRET)

    def locked(*a):
        raise KeyringLocked("the keychain is locked")
    monkeypatch.setattr(chain, "set_password", locked)
    monkeypatch.setattr(chain, "delete_password", locked)
    monkeypatch.setattr(sys, "stdin", io.StringIO("new-value\n"))
    assert cli.main(["keys", action, "HF_TOKEN"]) == 1
    err = capsys.readouterr().err
    assert "keychain" in err and "locked" in err and "Traceback" not in err
    assert json.loads((home / "keys.json").read_text()) == ["HF_TOKEN"]   # the index still says what the keychain holds


# --- keys check: whether the provider accepts a key ------------------------------------------------------------------

def check_json(capsys, name: str) -> dict:
    assert cli.main(["--json", "keys", "check", name]) == 0
    [out] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert out["type"] == "key_check" and out["name"] == name
    assert SECRET not in json.dumps(out)
    return out


def test_a_key_is_checked_with_its_own_provider_once_and_its_answer_said(home, chain, monkeypatch, capsys):
    from perception import keycheck
    asked = []

    def provider(status, body=b"{}"):
        def get(url, headers):
            asked.append((url, headers))
            return status, body
        return get

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setattr(keycheck, "get", provider(200))
    assert check_json(capsys, "ANTHROPIC_API_KEY") == {"v": 1, "type": "key_check", "name": "ANTHROPIC_API_KEY", "ok": True,
                                                        "detail": "Anthropic accepted the key"}
    assert asked == [(keycheck.ANTHROPIC, {"x-api-key": SECRET, "anthropic-version": "2023-06-01"})]
    monkeypatch.setattr(keycheck, "get", provider(401))
    assert check_json(capsys, "ANTHROPIC_API_KEY")["detail"] == "Anthropic refused the key"
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setattr(keycheck, "get", provider(200, b'{"name": "ana"}'))
    assert check_json(capsys, "HF_TOKEN")["detail"] == "Hugging Face accepted the key (as ana)"
    assert asked[-1] == (keycheck.HUGGING_FACE, {"Authorization": f"Bearer {SECRET}"})


def test_a_key_that_is_not_set_or_a_provider_out_of_reach_is_said_so_and_nothing_is_sent(home, chain, monkeypatch, capsys):
    from perception import keycheck
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(keycheck, "get", lambda url, headers: pytest.fail("nothing to send"))
    assert check_json(capsys, "ANTHROPIC_API_KEY") == {"v": 1, "type": "key_check", "name": "ANTHROPIC_API_KEY", "ok": False,
                                                        "detail": "ANTHROPIC_API_KEY is not set"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)

    def offline(url, headers):
        raise OSError("Network is unreachable")
    monkeypatch.setattr(keycheck, "get", offline)
    out = check_json(capsys, "ANTHROPIC_API_KEY")
    assert out["ok"] is False and "could not reach" in out["detail"]


def test_a_key_perception_has_no_check_for_is_an_error(home, chain, capsys):
    assert cli.main(["--json", "keys", "check", "SOME_OTHER_KEY"]) == 1
    [err] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert err["type"] == "error" and "no check for SOME_OTHER_KEY" in err["problems"][0]
