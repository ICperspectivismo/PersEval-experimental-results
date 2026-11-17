"""Portable low-level file opening for Linux, macOS and Windows.

``os.open`` exposes platform-specific flags:

* ``O_NOFOLLOW`` and ``O_NONBLOCK`` exist only on POSIX systems. On Windows they
  are absent, so referencing them raises ``AttributeError``.
* ``O_BINARY`` exists only on Windows, where ``os.open`` otherwise opens files in
  *text* mode (CRLF translation and Ctrl-Z treated as end of file). Wrapping such
  a descriptor with ``os.fdopen(fd, 'rb')`` does not undo that, so bundles,
  checksums and Git index bytes would be silently corrupted.
"""
from __future__ import annotations

import errno
import os

O_BINARY = getattr(os, 'O_BINARY', 0)
O_NONBLOCK = getattr(os, 'O_NONBLOCK', 0)  # POSIX: never block on a FIFO; Windows has none.
O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)


def open_fd(path, flags, mode=0o777, *, nofollow=False):
    """``os.open`` in binary mode on every platform, optionally refusing symlinks.

    Where ``O_NOFOLLOW`` is unavailable (Windows), symlinks are rejected with an
    explicit ``lstat`` check before opening. Callers still verify the opened
    descriptor with ``fstat`` where they require a regular file.
    """
    if nofollow:
        if O_NOFOLLOW:
            flags |= O_NOFOLLOW
        elif os.path.islink(path):
            raise OSError(errno.ELOOP, 'refusing to follow a symbolic link', os.fspath(path))
    return os.open(path, flags | O_BINARY, mode)


def open_read(path, *, nofollow=False):
    """Read-only, binary, nonblocking descriptor."""
    return open_fd(path, os.O_RDONLY | O_NONBLOCK, nofollow=nofollow)