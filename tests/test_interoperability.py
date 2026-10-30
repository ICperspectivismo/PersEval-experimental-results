"""Portable producer vectors and independent contract checks; no sibling required."""
import copy
import importlib.abc
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from perseval_results.bundle import pack_files, validate_bundle
from perseval_results.protocol import Contract, ProtocolError, canonical_json, checksum_bytes, digest

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'fixtures/producer/perseval-1.0.perseval-result.tar.gz'


def checker():
    spec=importlib.util.spec_from_file_location('offline_integration_checker',ROOT/'tools/check_integration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ProducerConformance(unittest.TestCase):
    def test_real_producer_vector_accepted_without_importing_producer(self):
        class BlockProducer(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,*args):
                if fullname=='perseval' or fullname.startswith('perseval.'):
                    raise AssertionError('consumer imported producer')
        with patch.object(sys,'meta_path',[BlockProducer(),*sys.meta_path]):
            _,docs=validate_bundle(FIXTURE,Contract(ROOT))
        self.assertEqual(docs['run']['configuration']['model']['name'],'roberta-base')
        self.assertEqual(docs['run']['configuration']['hyperparameters']['inference']['model_family'],'encoder')
        self.assertEqual(docs['run']['seed'],42)
        self.assertEqual(docs['metrics']['compact']['global_f1'],1.0)

    def test_producer_vector_integrity_and_schema_digest(self):
        record=json.loads((ROOT/'fixtures/producer/provenance.json').read_bytes())
        self.assertTrue(record['synthetic'])
        self.assertEqual(record['archive_sha256'],digest(FIXTURE.read_bytes()))
        self.assertEqual(record['schema_digest'],Contract(ROOT).schema_digest)
        self.assertIsNone(record['producer_source_revision'])

    def test_unknown_bundle_version_has_compatible_release_diagnostic(self):
        contract=Contract(ROOT);files,docs=validate_bundle(FIXTURE,contract)
        docs['manifest']['schema_version']='999.0'
        files['manifest.json']=canonical_json(docs['manifest'])
        files['checksums.sha256']=checksum_bytes({k:v for k,v in files.items() if k!='checksums.sha256'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unknown.tar.gz';path.write_bytes(pack_files(files))
            with self.assertRaisesRegex(ProtocolError,'unsupported.*schema version.*compatible'):
                validate_bundle(path,contract)

    def test_local_authoritative_schema_drift_has_restoration_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(ROOT/'schemas',root/'schemas')
            path=root/'schemas/result-bundle/1.0/run.schema.json';original=path.read_bytes()
            for changed in (original+b'\n',b'{invalid JSON'):
                path.write_bytes(changed)
                with self.subTest(changed=changed[:20]),self.assertRaisesRegex(ProtocolError,'schema drift.*Restore'):
                    Contract(root)

    def fake_vendor(self,root):
        shutil.copytree(ROOT/'schemas/result-bundle/1.0',root/'perseval/results/schemas/1.0')
        shutil.copytree(ROOT/'fixtures/conformance',root/'tests/fixtures/result-bundle-1.0')
        shutil.copyfile(ROOT/'schemas/vendor-lock.schema.json',root/'perseval/results/schemas/vendor-lock.schema.json')
        files={p.relative_to(ROOT/'fixtures/conformance').as_posix():p.read_bytes() for p in (ROOT/'fixtures/conformance').rglob('*') if p.is_file()}
        record={'schema_digest':Contract(ROOT).schema_digest,'conformance_digest':digest(checksum_bytes(files)),'source_revision':None}
        (root/'perseval/results/schemas/vendor-source.json').write_text(json.dumps(record))

    def test_cross_checkout_comparison_reports_exact_file_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer=Path(tmp);self.fake_vendor(producer)
            self.assertEqual(checker().compare_contract(producer,ROOT)['schema_digest'],Contract(ROOT).schema_digest)
            path=producer/'perseval/results/schemas/1.0/run.schema.json';path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'schema drift: run.schema.json.*Vendor the exact'):
                checker().compare_contract(producer,ROOT)

    def test_cross_checkout_comparison_reports_fixture_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer=Path(tmp);self.fake_vendor(producer)
            path=producer/'tests/fixtures/result-bundle-1.0/index.json';path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'conformance fixture drift'):
                checker().compare_contract(producer,ROOT)

    def test_contract_change_never_downloads_or_imports_benchmark_dependencies(self):
        import ast
        for path in (ROOT/'src/perseval_results').glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):modules=[item.name for item in node.names]
                elif isinstance(node,ast.ImportFrom) and node.level==0:modules=[node.module or '']
                else:continue
                self.assertFalse(any(name.split('.')[0] in ('perseval','torch','transformers','datasets') for name in modules),path)
