"""Offline LFS pointer handling; never fetch objects or install Git configuration."""
from __future__ import annotations

import re

from .protocol import ProtocolError, digest

# Small payloads remain ordinary Git blobs. Never put core metadata in LFS.
LFS_THRESHOLD = 10 * 1024 * 1024
POINTER_PREFIX = b'version https://git-lfs.github.com/spec/v1\n'
POINTER = re.compile(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n')


def eligible(name):
    parts = name.split('/')
    return (len(parts) == 3 and parts[0] == 'runs' and parts[2] == 'predictions.csv') or (
        len(parts) >= 4 and parts[0] == 'runs' and parts[2] == 'artifacts')


def pointer(data):
    if not data.startswith(POINTER_PREFIX):
        return None
    match = POINTER.fullmatch(data)
    if match is None:
        raise ProtocolError('invalid or extended Git LFS pointer; hydrate the payload for full validation')
    return 'sha256:'+match[1].decode(), int(match[2])


def pointer_bytes(data):
    return POINTER_PREFIX + f'oid {digest(data)}\nsize {len(data)}\n'.encode()


def stored_reference(name, data, checksums):
    # An explicitly selected artifact may itself contain literal pointer text.
    # Its ordinary checksum distinguishes that payload from an unhydrated object.
    ordinary = f'{digest(data)[7:]}  {name}'.encode('ascii')
    return None if ordinary in checksums.splitlines() else pointer(data)


def matches(data, blob, name):
    return blob == data or (eligible(name) and blob == pointer_bytes(data))
