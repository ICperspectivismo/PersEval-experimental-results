from __future__ import annotations

import copy
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from perseval_results.build import build, validate_preview
from perseval_results.bundle import pack_files, read_archive, seal_payload, validate_bundle, validate_files
from perseval_results.cli import main
from perseval_results.protocol import Contract, ProtocolError, canonical_json, checksum_bytes, configuration_id, digest, read_json, timestamp
from perseval_results.publish import publish, validate_git_history
from perseval_results.records import compatible, leaderboard, limitations, validate_sources

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures/conformance'
BUNDLE = FIXTURES / 'valid/minimal.perseval-result.tar.gz'
SID = 'sub_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
RID2 = 'run_22222222-2222-4222-8222-222222222222'
RID3 = 'run_33333333-3333-4333-8333-333333333333'


class Conformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Contract(ROOT)

    def test_every_machine_readable_fixture(self):
        index = json.loads((FIXTURES / 'index.json').read_bytes())
        for filename in index['valid_bundles']:
            with self.subTest(filename=filename):
                validate_bundle(FIXTURES / filename, self.contract)
        for case in index['invalid_bundles']:
            with self.subTest(filename=case['file']):
                with self.assertRaisesRegex(ProtocolError, case['expected_error']):
                    validate_bundle(FIXTURES / case['file'], self.contract)
        for filename in index['valid_reported_records']:
            self.contract.validate('reported', read_json((FIXTURES / filename).read_bytes()), record=True)

    def test_reexport_has_identical_bytes(self):
        for name in ['minimal', 'selected-artifact', 'limited-provenance']:
            path = FIXTURES / f'valid/{name}.perseval-result.tar.gz'
            files = read_archive(path)
            self.assertEqual(pack_files(files), path.read_bytes())
            self.assertEqual(pack_files(files), pack_files(files))
            self.assertNotIn(b'exported_at', files['manifest.json'])

    def test_schema_digest_and_inventory(self):
        directory = ROOT / 'schemas/result-bundle/1.0'
        inventory = checksum_bytes({p.name:p.read_bytes() for p in directory.glob('*.schema.json')})
        self.assertEqual(inventory, (directory / 'SHA256SUMS').read_bytes())
        self.assertEqual(digest(inventory), self.contract.schema_digest)

    def test_identity_patterns_reject_trailing_newlines(self):
        _, docs = validate_bundle(BUNDLE, self.contract)
        for field in ['run_id', 'configuration_id']:
            run = copy.deepcopy(docs['run'])
            run[field] += '\n'
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                self.contract.validate('run', run)

    def test_configuration_identity_excludes_execution_seed(self):
        _, docs = validate_bundle(BUNDLE, self.contract)
        run = docs['run']
        cid = configuration_id(run['configuration'])
        other = copy.deepcopy(run)
        other.update(run_id=RID2, seed=123)
        other['execution']['started_at'] = '2026-01-01T00:00:30Z'
        self.assertEqual(cid, configuration_id(other['configuration']))
        for field, value in [('seed', 99), ('test_fingerprint', digest(b'other test'))]:
            c = copy.deepcopy(run['configuration']); c['split'][field] = value
            self.assertNotEqual(cid, configuration_id(c))

    def test_nonfinite_and_duplicate_json_keys_rejected(self):
        for data in [b'{"x":NaN}\n', b'{"x":Infinity}\n', b'{"x":1e400}\n', b'{"x":1,"x":2}\n']:
            with self.subTest(data=data), self.assertRaises(ProtocolError):
                read_json(data)

    def test_timestamps_require_timezone_without_optional_format_dependencies(self):
        for value in ['2026-01-01T00:00:00', '2026-99-01T00:00:00Z', '2026-01-01', '2026-01-01T00:00:00+99:00']:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                timestamp(value)
        self.assertEqual(timestamp('2026-01-01T00:00:00Z'), timestamp('2025-12-31T21:00:00-03:00'))

    def test_noncanonical_json_rejected(self):
        with self.assertRaises(ProtocolError):
            read_json(b'{ "x": 1 }\n')

    def test_schema_resolution_never_retrieves_network(self):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            validate_bundle(BUNDLE, Contract(ROOT))

    def test_oversized_header_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.tar.gz'
            # A GNU long-name member is rejected without consuming its advertised payload.
            import gzip
            info = tarfile.TarInfo('long'); info.type = tarfile.GNUTYPE_LONGNAME; info.size = 10**9
            path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT), mtime=0))
            with self.assertRaisesRegex(ProtocolError, 'regular'):
                read_archive(path)

    def test_truncated_tar_and_trailing_data_rejected(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.tar.gz'
            path.write_bytes(gzip.compress(b'bad', mtime=0))
            with self.assertRaisesRegex(ProtocolError, 'truncated'):
                read_archive(path)
            path.write_bytes(gzip.compress(gzip.decompress(BUNDLE.read_bytes()) + b'junk', mtime=0))
            with self.assertRaisesRegex(ProtocolError, 'trailing'):
                read_archive(path)

    def test_unknown_schema_version_rejected(self):
        files = read_archive(BUNDLE)
        run = read_json(files['run.json']); run['schema_version']='2.0'
        payload={k:v for k,v in files.items() if k not in ['manifest.json','checksums.sha256']}
        payload['run.json']=canonical_json(run)
        with self.assertRaisesRegex(ProtocolError, 'run'):
            validate_files(seal_payload(payload,self.contract),self.contract)

    def test_reported_rejects_fabricated_run_fields(self):
        record = read_json((FIXTURES / 'valid/reported.json').read_bytes())
        for key, value in [('run_ids', []), ('seed', 42), ('stddev', 0.1), ('environment', {})]:
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                self.contract.validate('reported', {**record,key:value}, record=True)


if __name__ == '__main__':
    unittest.main()
