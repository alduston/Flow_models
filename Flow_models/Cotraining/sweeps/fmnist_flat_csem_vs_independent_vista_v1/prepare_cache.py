#!/usr/bin/env python3
"""Prepare Fashion-MNIST and Torch weight caches with bounded memory.

Standard library only: never imports Torch, constructs networks, or deserializes
checkpoints. This file can also be copied into an older bundle on its own.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
import urllib.request


CHUNK = 1024 * 1024
# torchvision.datasets.FashionMNIST resources; the cache layout is unchanged.
DATA_FILES = (
    ('train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d', 60000, True),
    ('train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe', 60000, False),
    ('t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79', 10000, True),
    ('t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310', 10000, False),
)
DATA_MIRRORS = (
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
    'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
)
# Exact torchvision checkpoints used by the unchanged training/evaluation code.
WEIGHTS = (
    ('inception_v3_google-0cc3c7bd.pth', '0cc3c7bd'),
    ('vgg16-397923af.pth', '397923af'),
)


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(urls, destination, expected_hash, algorithm='sha256'):
    """Check/reuse an existing file, or stream to a verified atomic replacement."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name('.' + destination.name + '.download.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_file() and file_hash(destination, algorithm).startswith(expected_hash):
            print('[CACHED]', destination, flush=True)
            return destination
        errors = []
        for url in urls:
            fd, name = tempfile.mkstemp(prefix='.' + destination.name + '-', suffix='.part',
                                        dir=destination.parent)
            temporary = Path(name)
            try:
                print('[DOWNLOAD]', url, flush=True)
                digest = hashlib.new(algorithm)
                count = 0
                with os.fdopen(fd, 'wb') as target:
                    request = urllib.request.Request(url, headers={'User-Agent': 'CSEM-cache-preparation/1'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        length = response.headers.get('Content-Length')
                        for chunk in iter(lambda: response.read(CHUNK), b''):
                            target.write(chunk)
                            digest.update(chunk)
                            count += len(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if length is not None and count != int(length):
                    raise ValueError(f'Truncated download: {count} of {length} bytes')
                if not digest.hexdigest().startswith(expected_hash):
                    raise ValueError(f'{algorithm} checksum mismatch')
                os.replace(temporary, destination)
                print('[READY]', destination, flush=True)
                return destination
            except (OSError, ValueError) as exc:
                errors.append(f'{url}: {exc}')
            finally:
                temporary.unlink(missing_ok=True)
        raise RuntimeError('Could not cache ' + str(destination) + ': ' + '; '.join(errors))


def validate_raw(raw):
    """Check both IDX splits without materializing image tensors.

    Where original gzip files are present, also verify the official checksum and
    compare decompressed content. Raw-only torchvision caches remain supported.
    """
    for name, checksum, count, images in DATA_FILES:
        path = raw / name[:-3]
        expected = (2051, count, 28, 28) if images else (2049, count)
        with path.open('rb') as source:
            header = source.read(4 * len(expected))
        if header != struct.pack('>' + 'I' * len(expected), *expected):
            raise ValueError(f'Invalid IDX header: {path}')
        size = len(header) + count * (28 * 28 if images else 1)
        if path.stat().st_size != size:
            raise ValueError(f'Invalid IDX size: {path}')
        archive = raw / name
        if archive.exists():
            if file_hash(archive, 'md5') != checksum:
                raise ValueError(f'Invalid Fashion-MNIST checksum: {archive}')
            with gzip.open(archive, 'rb') as source, path.open('rb') as target:
                while True:
                    chunk = source.read(CHUNK)
                    if chunk != target.read(CHUNK):
                        raise ValueError(f'IDX differs from original gzip: {path}')
                    if not chunk:
                        break


def prepare_data(cache):
    """Use the same exclusive FMNIST lock and staged publication as paper_io."""
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / 'FMNIST'
    with (cache / '.FMNIST.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            validate_raw(destination / 'FashionMNIST' / 'raw')
        except (OSError, ValueError, EOFError) as exc:
            reason = repr(exc)
        else:
            print('[CACHED]', destination, flush=True)
            return destination
        stage = Path(tempfile.mkdtemp(prefix='.FMNIST-stage-', dir=cache))
        try:
            raw = stage / 'FashionMNIST' / 'raw'
            raw.mkdir(parents=True)
            for name, checksum, _, _ in DATA_FILES:
                archive = download([mirror + name for mirror in DATA_MIRRORS], raw / name, checksum, 'md5')
                with gzip.open(archive, 'rb') as source, (raw / name[:-3]).open('wb') as target:
                    shutil.copyfileobj(source, target, length=CHUNK)
            validate_raw(raw)
            (stage / 'prepared.json').write_text(json.dumps({'dataset': 'FMNIST', 'validated': True}) + '\n')
            if destination.exists():
                quarantine = cache / ('.FMNIST-corrupt-' + str(time.time_ns()))
                os.replace(destination, quarantine)
                (quarantine / 'quarantine_reason.txt').write_text(reason + '\n')
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        print('[READY]', destination, flush=True)
        return destination


def prepare_weights(torch_home=None):
    """Populate torch.hub's default checkpoint cache; never torch.load a file."""
    if torch_home is None:
        torch_home = os.environ.get('TORCH_HOME', os.path.join(
            os.environ.get('XDG_CACHE_HOME', '~/.cache'), 'torch'))
    checkpoints = Path(torch_home).expanduser().resolve() / 'hub' / 'checkpoints'
    for name, checksum in WEIGHTS:
        download(['https://download.pytorch.org/models/' + name], checkpoints / name, checksum)
    print('Inception and VGG16 weights cached. LPIPS calibration weights ship with the lpips package.', flush=True)
    return checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=os.environ.get(
        'CSEM_DATA_ROOT', str(Path(__file__).resolve().parent / 'data_cache_fmnist')))
    parser.add_argument('--weights', action='store_true', help='also cache Inception and VGG weights')
    parser.add_argument('--weights-only', action='store_true', help='leave the data cache untouched')
    args = parser.parse_args(argv)
    if not args.weights_only:
        prepare_data(args.data_root)
    if args.weights or args.weights_only:
        prepare_weights()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
