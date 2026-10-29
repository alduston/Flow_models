#!/usr/bin/env python3
"""Offline preparation regressions, including a process with a 96 MiB RAM limit."""
import builtins
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import prepare_cache as cache


def fixtures(root):
    remote = root / 'remote'
    remote.mkdir()
    specs = []
    for name, _, _, images in cache.DATA_FILES:
        count = 3 if name.startswith('train') else 2
        header = (2051, count, 28, 28) if images else (2049, count)
        payload = struct.pack('>' + 'I' * len(header), *header)
        payload += bytes(count * (28 * 28 if images else 1))
        compressed = gzip.compress(payload, mtime=0)
        (remote / name).write_bytes(compressed)
        specs.append((name, hashlib.md5(compressed).hexdigest(), count, images))
    return remote, tuple(specs)


def limited_child(root):
    import resource
    limit = 96 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    original_import = builtins.__import__
    forbidden = {'torch', 'torchvision', 'lpips', 'numpy', 'trainer'}

    def checked_import(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            raise AssertionError('Heavy import in preparation: ' + name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = checked_import
    import paper
    metadata = json.loads((root / 'fixtures.json').read_text())
    cache.DATA_FILES = metadata['data_files']
    cache.DATA_MIRRORS = ((root / 'remote').as_uri() + '/',)
    cache.WEIGHTS = metadata['weights']
    os.environ['TORCH_HOME'] = str(root / 'torch')
    original_open = cache.urllib.request.urlopen

    def local_open(request, **kwargs):
        url = request.full_url
        if url.startswith('https://download.pytorch.org/models/'):
            url = (root / 'remote' / url.rsplit('/', 1)[1]).as_uri()
        return original_open(url, **kwargs)

    data_root = root / 'data'
    with mock.patch.object(cache.urllib.request, 'urlopen', side_effect=local_open):
        assert paper.main(['prepare-data', '--weights', '--data-root', str(data_root)]) == 0
    raw = data_root / 'FMNIST' / 'FashionMNIST' / 'raw'
    inodes = {p.name: p.stat().st_ino for p in raw.iterdir()}
    # Every requested asset now exists; a second run must never access the network.
    with mock.patch.object(cache.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected network access')):
        assert paper.main(['prepare-data', '--weights', '--data-root', str(data_root)]) == 0
        assert cache.main(['--weights-only', '--data-root', str(root / 'untouched')]) == 0
    assert inodes == {p.name: p.stat().st_ino for p in raw.iterdir()}
    assert not (root / 'untouched').exists()
    assert not forbidden.intersection(sys.modules)
    print(json.dumps({'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'address_space_limit_mib': 96, 'checkpoint_size_mib': 128,
                      'heavy_imports': False}))


class PreparationTests(unittest.TestCase):
    def test_cli_streams_checkpoint_larger_than_memory_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, specs = fixtures(root)
            # A whole-file read or checkpoint deserialization cannot fit in the
            # child's 96 MiB address space; actual streaming has room to spare.
            path = remote / 'large-test-weight.pth'
            block = bytes(1024 * 1024)
            with path.open('wb') as target:
                for _ in range(128):
                    target.write(block)
            small = remote / 'small-test-weight.pth'
            small.write_bytes(b'fixture checkpoint, never a pickled tensor')
            metadata = {'data_files': specs, 'weights': [
                (p.name, cache.file_hash(p)[:8]) for p in (path, small)]}
            (root / 'fixtures.json').write_text(json.dumps(metadata))
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     '--limited-child', str(root)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(result.stdout.splitlines()[-1])
            self.assertLess(summary['peak_rss_kib'], 96 * 1024)
            print('[PASS] paper.py preparation: 128 MiB checkpoint under 96 MiB address-space limit; '
                  f"peak RSS {summary['peak_rss_kib'] / 1024:.1f} MiB; no heavy imports", flush=True)

    def test_bad_weight_or_interruption_keeps_prior_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoint.pth'
            destination.write_bytes(b'prior cache')
            expected = hashlib.sha256(b'correct replacement').hexdigest()
            response = io.BytesIO(b'corrupt replacement')
            response.headers = {}
            with mock.patch.object(cache.urllib.request, 'urlopen', return_value=response):
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    cache.download(['https://example.invalid/fixture'], destination, expected)
            self.assertEqual(destination.read_bytes(), b'prior cache')
            self.assertFalse(list(Path(directory).glob('*.part')))
            interrupted = mock.MagicMock()
            interrupted.__enter__.return_value = interrupted
            interrupted.headers = {}
            interrupted.read.side_effect = [b'partial', OSError('interrupted stream')]
            with mock.patch.object(cache.urllib.request, 'urlopen', return_value=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupted stream'):
                    cache.download(['https://example.invalid/fixture'], destination, expected)
            self.assertEqual(destination.read_bytes(), b'prior cache')
            self.assertFalse(list(Path(directory).glob('*.part')))

    def test_staged_data_repair_and_raw_only_old_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, specs = fixtures(root)
            data_root = root / 'data'
            old = data_root / 'FMNIST'
            old.mkdir(parents=True)
            (old / 'old-file').write_bytes(b'incomplete old cache')
            with mock.patch.object(cache, 'DATA_FILES', specs), \
                 mock.patch.object(cache, 'DATA_MIRRORS', (remote.as_uri() + '/',)), \
                 contextlib.redirect_stdout(io.StringIO()):
                cache.prepare_data(data_root)
                quarantines = list(data_root.glob('.FMNIST-corrupt-*'))
                self.assertEqual(len(quarantines), 1)
                self.assertEqual((quarantines[0] / 'old-file').read_bytes(), b'incomplete old cache')
                raw = old / 'FashionMNIST' / 'raw'
                # A same-size raw-file corruption is detected against the gzip.
                damaged = raw / specs[0][0][:-3]
                with damaged.open('r+b') as target:
                    target.seek(-1, 2)
                    target.write(b'\x01')
                with self.assertRaisesRegex(ValueError, 'differs from original'):
                    cache.validate_raw(raw)
                cache.prepare_data(data_root)
                raw = old / 'FashionMNIST' / 'raw'
                for name, *_ in specs:
                    (raw / name).unlink()
                with mock.patch.object(cache.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected network')):
                    self.assertEqual(cache.prepare_data(data_root), old)
                self.assertFalse(list(data_root.glob('.FMNIST-stage-*')))

    def test_standard_torch_cache_paths(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(cache, 'WEIGHTS', ()):
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.dict(os.environ, {'TORCH_HOME': str(root / 'explicit')}):
                    self.assertEqual(cache.prepare_weights(), root / 'explicit' / 'hub' / 'checkpoints')
                with mock.patch.dict(os.environ, {'XDG_CACHE_HOME': str(root / 'xdg')}, clear=True):
                    self.assertEqual(cache.prepare_weights(), root / 'xdg' / 'torch' / 'hub' / 'checkpoints')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--limited-child':
        limited_child(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
