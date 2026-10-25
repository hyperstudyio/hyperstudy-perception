"""The user's keys (API keys, tokens) for nodes and cloud places, found in this order:

1. the environment (a name already known below, or any *_KEY / *_TOKEN);
2. the system keychain (macOS Keychain, Linux Secret Service, Windows Credential Manager), where `perception keys set`
   stores them, under the service "hyperstudy-perception";
3. NAME=value lines in ~/.config/hyperstudy/*.env, only from files no one else can read or write (0600, the user's own).
   A file others can read is refused, not read, and said so: a key in it may already have been read by someone else.

Values are never printed, logged or written to run files. The keychain cannot list what it holds, so the names stored
there are kept in ~/.config/hyperstudy/keys.json (names only)."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

SERVICE = "hyperstudy-perception"
NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def conf_dir() -> Path:
    return Path.home() / ".config" / "hyperstudy"


@dataclass
class Found:
    values: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)   # name -> "environment", "keychain" or the file's path
    problems: list[str] = field(default_factory=list)


def private_file_problem(path: Path) -> str | None:
    """Why a key file is not used, or None: it must be the user's own, a regular file, and readable by no one else."""
    try:
        st = path.lstat()
    except OSError as e:
        return f"{path} could not be read ({e})"
    if not stat.S_ISREG(st.st_mode):
        return f"{path} is not a regular file (a link?): its keys are not used"
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return f"{path} is not yours: its keys are not used"
    if st.st_mode & 0o077:
        return f"{path} can be read by others (mode {oct(stat.S_IMODE(st.st_mode))}): its keys are not used; chmod 600 it"
    return None


def read_env_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def keyring_module():
    """The keyring library, or None where no keychain can be used (a headless server, or PERCEPTION_NO_KEYCHAIN)."""
    if os.environ.get("PERCEPTION_NO_KEYCHAIN"):
        return None
    import keyring
    from keyring.backends import fail
    return None if isinstance(keyring.get_keyring(), fail.Keyring) else keyring


class Keys:
    def __init__(self, conf: Path | None = None):
        self.conf = conf or conf_dir()
        self.index = self.conf / "keys.json"

    def stored_names(self) -> list[str]:
        """Names `keys set` stored in the keychain (keys.json: names only)."""
        try:
            return [n for n in json.loads(self.index.read_text()) if isinstance(n, str) and NAME.match(n)]
        except (OSError, ValueError, TypeError):
            return []

    def _write(self, path: Path, text: str):
        """Write a file in one step, 0600 from its first byte, in a folder only the user can enter."""
        self.conf.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    def _write_index(self, names: list[str]):
        self._write(self.index, json.dumps(sorted(set(names))))

    def _file_keys(self) -> dict[str, str]:
        """What keys.env holds (the file `set` writes where there is no keychain), refusing one others can read."""
        path = self.conf / "keys.env"
        if not path.exists():
            return {}
        problem = private_file_problem(path)
        if problem:
            raise ValueError(problem)
        return read_env_file(path)

    def _write_file_keys(self, keys: dict[str, str]):
        self._write(self.conf / "keys.env", "".join(f"{k}={v}\n" for k, v in sorted(keys.items())))

    def load(self) -> Found:
        found = Found()
        # 3. private files, lowest: what a keychain or the environment holds wins
        for path in sorted(self.conf.glob("*.env")) if self.conf.is_dir() else []:
            problem = private_file_problem(path)
            if problem:
                found.problems.append(problem)
                continue
            for k, v in read_env_file(path).items():
                found.values[k], found.sources[k] = v, str(path)
        # 2. the keychain
        kr = keyring_module()
        if kr is not None:
            for name in self.stored_names():
                try:
                    value = kr.get_password(SERVICE, name)
                except Exception as e:   # a locked or refused keychain: said, not fatal (other sources may have it)
                    found.problems.append(f"the keychain did not give {name} ({type(e).__name__})")
                    continue
                if value:
                    found.values[name], found.sources[name] = value, "keychain"
        # 1. the environment
        for k, v in os.environ.items():
            if v and (k in found.values or k.endswith(("_KEY", "_TOKEN"))):
                found.values[k], found.sources[k] = v, "environment"
        return found

    def set(self, name: str, value: str) -> str:
        """Store a key: in the keychain, or (no keychain here) in ~/.config/hyperstudy/keys.env, 0600. Where it went."""
        if not NAME.match(name):
            raise ValueError(f"a key's name is upper-case letters, digits and _ (like RUNPOD_API_KEY): {name!r}")
        value = value.strip()
        if not value or "\n" in value:
            raise ValueError("a key is one non-empty line")
        kr = keyring_module()
        if kr is not None:
            kr.set_password(SERVICE, name, value)
            self._write_index([*self.stored_names(), name])
            return "keychain"
        keys = self._file_keys()
        keys[name] = value
        self._write_file_keys(keys)
        return str(self.conf / "keys.env")

    def remove(self, name: str) -> list[str]:
        """Remove a key from the keychain and from keys.env (what `set` wrote); where it was. The user's own .env files
        are theirs to edit."""
        removed = []
        kr = keyring_module()
        if kr is not None and name in self.stored_names():
            from keyring.errors import PasswordDeleteError
            try:
                kr.delete_password(SERVICE, name)
                removed.append("keychain")
            except PasswordDeleteError:
                pass   # already gone from the keychain (removed there by hand)
            self._write_index([n for n in self.stored_names() if n != name])
        keys = self._file_keys()
        if keys.pop(name, None) is not None:
            self._write_file_keys(keys)
            removed.append(str(self.conf / "keys.env"))
        return removed


_warned = False


def load_secrets() -> dict[str, str]:
    """Every key found (Keys.load), saying once on stderr which files were refused and why."""
    global _warned
    found = Keys().load()
    if found.problems and not _warned:
        _warned = True
        for p in found.problems:
            print(f"perception: {p}", file=sys.stderr)
    return found.values
