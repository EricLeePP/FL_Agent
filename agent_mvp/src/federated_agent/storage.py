"""Atomic local JSON checkpoints, strict JSON reads and OS file leases.

This is the only JSON writer in the package. Writes use a unique temporary file and
bounded retries for transient Windows sharing conflicts.
"""

import json
import os
from pathlib import Path
import tempfile
import time


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except PermissionError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 5:
                    raise
                time.sleep(min(0.025 * 2 ** attempt, 0.2))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class LockBusyError(ValueError):
    """Another process holds the lease."""


class FileLease:
    """An OS lock that is released on process exit, including an unclean shutdown."""

    def __init__(self, path):
        self.handle = Path(path).open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.handle.tell() == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise LockBusyError("This session or task is already open in another process.") from None

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def is_locked(path) -> bool:
    """True while another process holds the lease at path."""
    try:
        probe = FileLease(path)
    except LockBusyError:
        return True
    probe.close()
    return False
