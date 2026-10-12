#!/usr/bin/env python3
"""Run a vendored entrypoint with LSGM's trusted local checkpoint semantics.

No numerical monkey patches. Stage-1 and diffusion implementations stay intact.
"""
import functools
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    import torch
    if len(sys.argv) < 2:
        raise SystemExit('Usage: lsgm_runner.py train_vae.py|train_vada.py|evaluate_vada.py [flags]')
    script, argv = sys.argv[1], sys.argv[2:]
    allowed = {'train_vae.py', 'train_vada.py', 'evaluate_vada.py', 'scripts/precompute_fid_statistics.py'}
    if script not in allowed:
        raise SystemExit('Unknown vendored entrypoint: ' + script)
    # PyTorch 2.6 changed this default; upstream checkpoints contain Namespace.
    original_load = torch.load
    @functools.wraps(original_load)
    def trusted_load(*a, **kw):
        kw.setdefault('weights_only', False)
        return original_load(*a, **kw)
    torch.load = trusted_load
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root/'lsgm'))
    sys.path.insert(0, str(root))
    def flag(k, default=None):
        return argv[argv.index(k)+1] if k in argv else default
    output = Path(flag('--root', str(root/'results')))/flag('--save', 'exp')
    start = time.time()
    if '--help' not in argv and '-h' not in argv:
        output.mkdir(parents=True, exist_ok=True)
        env = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'requested_op_backend': os.environ.get('LSGM_OP_BACKEND', 'auto'),
               'script': script, 'argv': argv}
        try:
            from apex.optimizers import FusedAdam
            env['optimizer_backend'] = 'apex.FusedAdam'
        except ImportError:
            env['optimizer_backend'] = 'torch.optim.Adam'
        (output/'run_environment.json').write_text(json.dumps(env, indent=2)+'\n')
    sys.argv = [str(root/'lsgm'/script)] + argv
    try:
        runpy.run_path(str(root/'lsgm'/script), run_name='__main__')
    finally:
        if '--help' not in argv and '-h' not in argv:
            used = {}
            for name, attr in [('upfirdn2d', 'upfirdn2d_op'), ('fused_act', 'fused')]:
                mod = sys.modules.get('score_sde.op.' + name)
                if mod is not None:
                    used[name] = ('cuda' if getattr(mod, attr, None) is not None else
                                  'native' if getattr(mod, '_native_only', False) or not torch.cuda.is_available() else
                                  'not invoked')
            env['actual_op_backend'] = used
            (output/'run_environment.json').write_text(json.dumps(env, indent=2)+'\n')
    if '--help' not in argv and '-h' not in argv:
        marker = {'script': script, 'requested_epochs': int(flag('--epochs', '0')),
                  'started': start, 'finished': time.time(), 'returncode': 0}
        tmp = output/'execution_complete.json.tmp'
        tmp.write_text(json.dumps(marker, indent=2)+'\n')
        os.replace(tmp, output/'execution_complete.json')


if __name__ == '__main__':
    main()
