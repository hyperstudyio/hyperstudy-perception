"""The user's keys (API keys, tokens) for nodes and cloud places, found in this order:

1. the environment (a name already known below, or any *_KEY / *_TOKEN);
2. the system keychain (macOS Keychain, Linux Secret Service, Windows Credential Manager), where `perception keys set`
   stores them, under the service "hyperstudy-perception";
3. NAME=value lines in ~/.config/hyperstudy/*.env, only from files no one else can read or write (0600, the user's own).
   A file others can read is refused, not read, and said so: a key in it may already have been read by someone else.

Values are never printed, logged or written to run files. The keychain cannot list what it holds, so the names stored
there are kept in ~/.config/hyperstudy/keys.json (names only)."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import stat
import sys
import tempfile
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


POSIX = os.name != "nt"   # Windows has no POSIX modes (every file reads 0o666): a user's profile folder is their own there


def folder_problem(conf: Path) -> str | None:
    """Why nothing in the config folder is used, or None: it must be a real folder (not a link), the user's own, and one
    no one else can write to (else a file could be swapped in after it was checked)."""
    try:
        st = conf.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return f"{conf} is not a folder (a link?): its key files are not used"
    if POSIX and st.st_uid != os.getuid():
        return f"{conf} is not yours: its key files are not used"
    if POSIX and st.st_mode & 0o022:
        return f"{conf} is a folder others can write to (mode {oct(stat.S_IMODE(st.st_mode))}): its key files are not used; chmod 700 it"
    return None


def file_problem(path: Path, st: os.stat_result) -> str | None:
    """Why a key file is not used, or None: a regular file, the user's own, readable and writable by no one else."""
    if not stat.S_ISREG(st.st_mode):
        return f"{path} is not a regular file: its keys are not used"
    if POSIX and st.st_uid != os.getuid():
        return f"{path} is not yours: its keys are not used"
    if POSIX and st.st_mode & 0o077:
        return f"{path} can be read by others (mode {oct(stat.S_IMODE(st.st_mode))}): its keys are not used; chmod 600 it"
    return None


def parse_env(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_private(path: Path) -> tuple[dict[str, str], str | None]:
    """A key file's keys, or why it is refused: opened without following a link, and checked on that open file, so what
    is checked is what is read."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as e:
        if e.errno == errno.ELOOP:
            return {}, f"{path} is not a regular file (a link): its keys are not used"
        return {}, f"{path} could not be read ({e.strerror})"
    with os.fdopen(fd) as f:
        problem = file_problem(path, os.fstat(f.fileno()))
        return ({}, problem) if problem else (parse_env(f.read()), None)


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

    def _private_folder(self):
        """The config folder, made if missing and made private if it is not (0700): refused if it is a link or not the
        user's (folder_problem)."""
        self.conf.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = self.conf.lstat()
        if POSIX and not stat.S_ISLNK(st.st_mode) and st.st_uid == os.getuid() and st.st_mode & 0o077:
            os.chmod(self.conf, 0o700)
        problem = folder_problem(self.conf)
        if problem:
            raise ValueError(problem)

    def _write(self, path: Path, text: str):
        """Write a file in one step through a new temporary file of its own (0600 from its first byte, never one left
        over), in the private config folder."""
        self._private_folder()
        fd, tmp = tempfile.mkstemp(dir=self.conf, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _locked(self):
        """One change to keys.json and keys.env at a time (two `keys set` at once would drop a name otherwise)."""
        self._private_folder()
        try:
            import fcntl
        except ImportError:   # Windows: no flock; a single user's own commands
            yield
            return
        fd = os.open(self.conf / ".keys.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _write_index(self, names: list[str]):
        self._write(self.index, json.dumps(sorted(set(names))))

    def _file_keys(self) -> dict[str, str]:
        """What keys.env holds (the file `set` writes where there is no keychain), refusing one others can read."""
        path = self.conf / "keys.env"
        if not path.exists():
            return {}
        values, problem = read_private(path)
        if problem:
            raise ValueError(problem)
        return values

    def _write_file_keys(self, keys: dict[str, str]):
        self._write(self.conf / "keys.env", "".join(f"{k}={v}\n" for k, v in sorted(keys.items())))

    def load(self) -> Found:
        found = Found()
        # 3. private files, lowest: what a keychain or the environment holds wins
        problem = folder_problem(self.conf)
        if problem:
            found.problems.append(problem)
        for path in sorted(self.conf.glob("*.env")) if self.conf.is_dir() and not problem else []:
            values, refused = read_private(path)
            if refused:
                found.problems.append(refused)
            for k, v in values.items():
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
        if not value or value.splitlines() != [value]:   # any line break (\r, \x85, \u2028...) would add a line to keys.env
            raise ValueError("a key is one non-empty line")
        kr = keyring_module()
        with self._locked():
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
        with self._locked():
            return self._remove(name)

    def _remove(self, name: str) -> list[str]:
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
