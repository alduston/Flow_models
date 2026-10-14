"""FMNIST trainer entry point: prepared data and strict, atomic resume IO."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import train_hier_csem_fmnist as engine
import importlib
from paper_io import prepare_dataset, load_prepared_pair


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def prepare_data(cache):
    return prepare_dataset(engine.ref.DATASET_INFO['FMNIST']['class'], cache, 'FMNIST')


def loader_factory(cache):
    def load(batch_size, num_workers, dataset_key='FMNIST'):
        if dataset_key != 'FMNIST':
            raise ValueError('This sweep is FashionMNIST only')
        tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])
        tr, te = load_prepared_pair(engine.ref.DATASET_INFO['FMNIST']['class'],
                                   cache, 'FMNIST', transform=tf)
        return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True),
                DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers), 10)
    return load


def recipe(args):
    # Include evaluation and schedule settings as well as the structural model.
    omit = {'output', 'resume', 'device', 'data_root', 'smoke_test', 'self_test'}
    return {k: v for k, v in vars(args).items() if k not in omit}


def check_resume(old, current):
    changed = {k: (old.get(k), v) for k, v in current.items() if old.get(k) != v}
    if changed:
        raise ValueError(f'Resume recipe changed; use a fresh result directory: {changed}')


def trim_logs(directory, epoch):
    """Drop uncommitted rows left by a job killed after its latest checkpoint."""
    for p in Path(directory).glob('*.csv'):
        with p.open(newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        if not fields or 'epoch' not in fields:
            continue
        keep = [r for r in rows if int(float(r['epoch'])) <= epoch]
        if len(keep) != len(rows):
            tmp = p.with_name(p.name + '.tmp')
            with tmp.open('w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader(); wr.writerows(keep)
            os.replace(tmp, p)


def train(args, *, synthetic=False):
    engine = importlib.import_module('train_hier_csem_fmnist_3block' if args.blocks == 3 else 'train_hier_csem_fmnist')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    current = recipe(args)
    guard = out/'resume_recipe.json'
    if args.resume:
        if not guard.is_file():
            raise ValueError('A sweep checkpoint needs its resume_recipe.json; old standalone checkpoints are not imported')
        check_resume(json.loads(guard.read_text()), current)
        # Read CPU RNG bytes on CPU. The trainer moves model tensors separately.
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        trim_logs(out, int(saved['epoch']))
        del saved
    else:
        if (out/'latest.pt').exists() or (out/'loss.csv').exists():
            raise FileExistsError(f'Existing training output needs --resume: {out}')
        atomic_json(guard, current)
    original_loader = engine.ref.make_dataloaders
    if not synthetic:
        cache = Path(args.data_root).resolve() if args.data_root else Path(__file__).resolve().parent/'data_cache_fmnist'
        prepare_data(cache)
        engine.ref.make_dataloaders = loader_factory(cache)
    try:
        engine.train(args)
    finally:
        engine.ref.make_dataloaders = original_loader

