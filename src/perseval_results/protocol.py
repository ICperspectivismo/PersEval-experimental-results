"""Version 1.0 byte conventions and offline JSON Schema validation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


class ProtocolError(ValueError):
    pass


def timestamp(value: str):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', value):
        raise ProtocolError('timestamp must include seconds and timezone in the v1 RFC3339 profile')
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ProtocolError(f'invalid timestamp: {value}') from exc


def canonical_json(value) -> bytes:
    """Normative v1 profile: Python JSON shortest numbers, sorted keys, UTF-8, LF."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def digest(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(data: bytes):
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError(f'nonfinite number: {value}')))
        if canonical_json(value) != data:
            raise ProtocolError('JSON must use the canonical v1 byte profile')
        return value
    except (UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError(f'invalid JSON: {exc}') from exc


def canonical_path(name: str) -> str:
    p = PurePosixPath(name)
    if (not name or '\\' in name or p.is_absolute() or str(p) != name
            or any(part in ('.', '..') or not part or not part[0].isascii() or not part[0].isalnum() for part in name.split('/'))
            or any(not (c.isascii() and (c.isalnum() or c in '._-/')) for c in name)):
        raise ProtocolError(f'noncanonical path: {name!r}')
    return name


def checksum_bytes(files: dict[str, bytes]) -> bytes:
    return ''.join(f'{digest(files[name])[7:]}  {name}\n' for name in sorted(files)).encode('ascii')


class Contract:
    def __init__(self, root: Path):
        self.root = root.resolve()
        bundle_files = {p.name: p.read_bytes() for p in (self.root / 'schemas/result-bundle/1.0').glob('*.schema.json')}
        if not bundle_files:
            raise ProtocolError('no bundle schemas; use --repo to select a results checkout')
        inventory = checksum_bytes(bundle_files)
        if inventory != (self.root/'schemas/result-bundle/1.0/SHA256SUMS').read_bytes():
            raise ProtocolError('authoritative bundle schema checksum mismatch (schema drift). Restore the frozen 1.0 schemas and SHA256SUMS together from reviewed Git history; do not accept locally edited schema bytes.')
        self.schema_digest = digest(inventory)

        self.schemas = {}
        resources = []
        for path in sorted((self.root / 'schemas').glob('**/*.schema.json')):
            schema = json.loads(path.read_bytes())
            Draft202012Validator.check_schema(schema)
            self.schemas[schema['$id']] = schema
            resources.append((schema['$id'], Resource.from_contents(schema)))
        # No retrieval callback: an unknown reference fails instead of fetching the network.
        self.registry = Registry().with_resources(resources)

    def validate(self, name: str, value, *, record=False, kind=None):
        kind = kind or ('records' if record else 'result-bundle')
        if isinstance(value,dict) and value.get('schema_version','1.0') != '1.0':
            raise ProtocolError(f'unsupported {kind} schema version {value["schema_version"]!r} in {name}; supported version: 1.0. Use a compatible exporter/importer release with reviewed offline schemas.')
        uri = f'https://perseval.org/schemas/{kind}/1.0/{name}.schema.json'
        if uri not in self.schemas:
            raise ProtocolError(f'unknown schema {uri}')
        validator = Draft202012Validator(self.schemas[uri], registry=self.registry, format_checker=FormatChecker())
        try:
            error = next(validator.iter_errors(value), None)
        except RecursionError as exc:
            raise ProtocolError('schema validation nesting limit exceeded') from exc
        if error is not None:
            raise ProtocolError(f'{name}: {list(error.absolute_path)}: {error.message[:1024]}')


def configuration_id(configuration) -> str:
    return digest(canonical_json(configuration))
