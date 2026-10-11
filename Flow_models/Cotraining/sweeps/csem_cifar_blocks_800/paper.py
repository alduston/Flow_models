                print(f'[skip] active cell {row["cell_id"]}: job {st["slurm_job_id"]}')
                continue
        pending.append(row)
    print(f'Selected {len(pending)} training jobs. Dataset preparation runs inside them under a shared lock.')
    (Path(base)/'slurm_logs').mkdir(exist_ok=True)
    for row in pending:
        exports = {'CELL_ID': row['cell_id'], 'BASE_DIR': str(Path(base).resolve()), 'RESTART': str(int(restart))}
        if data_root:
            exports['CSEM_DATA_ROOT'] = str(Path(data_root).resolve())
        if any(',' in v or '\n' in v for v in exports.values()):
            raise ValueError('Slurm export values cannot contain commas or newlines')
        cmd = ['sbatch', '--parsable', f'--job-name=cifar_csem_b{row["blocks"]}',
               '--export=ALL,'+','.join(f'{k}={v}' for k, v in exports.items()), str(Path(base)/'job.slurm')]
        print(shlex.join(cmd))
        if not dry_run:
            q = subprocess.run(cmd, cwd=base, text=True, capture_output=True, check=True)
            jobid = parse_slurm_job_id(q.stdout)
            current = status(base, row)
            if not (str(current.get('slurm_job_id')) == jobid and current.get('started_utc')):
                atomic_json(status_path(base, row), dict(row, returncode=None,
                            slurm_job_id=jobid, config_hash=config_hash(row),
                            queued_utc=datetime.now(timezone.utc).isoformat()))
    return 0


def report(base, level=2.5):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    compiled = Path(base)/'compiled'; compiled.mkdir(exist_ok=True)
    curves, endpoints = [], []
    for row in manifest(base):
        frames = Path(base)/'results'/row['result_name']/'dataframes'
        path = frames/'eval_metrics.csv'
        if not path.is_file():
            path = frames/'eval_metrics_in_progress.csv'
        values = read_evaluation(path)
        for item in values:
            if item.get('tag') != 'LSI_Diff' or item.get('stage') != 'cotrain':
                continue
            try:
                key = fid_column(item, level=level)
            except ValueError:
                continue
            if not math.isfinite(float(item.get(key, 'nan'))):
                continue
            diag = f'fid_encoded_root_generated_children_cfg{str(float(level)).replace(".", "_")}'
            record = dict(blocks=int(row['blocks']), channels=row['block_channels'],
                          score_depths=row['score_depths'], epoch=int(float(item['epoch'])),
                          fid=float(item[key]), fid_posterior_recon=float(item.get('fid_vae_recon', 'nan')),
                          fid_encoded_root_generated_children=float(item.get(diag, 'nan')),
                          cfg=float(level), steps_per_block=20, seed=int(row['seed']),
                          completed=status(base, row).get('returncode') == 0)
            curves.append(record)
        block_rows = [c for c in curves if c['blocks'] == int(row['blocks'])]
        if block_rows:
            endpoints.append(max(block_rows, key=lambda r: r['epoch']))
    df = pd.DataFrame(curves); last = pd.DataFrame(endpoints)
    df.to_csv(compiled/f'curves_cfg{level:g}.csv', index=False)
    last.to_csv(compiled/f'latest_cfg{level:g}.csv', index=False)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for block, group in df.groupby('blocks'):
            group = group.sort_values('epoch')
            ax.plot(group.epoch, group.fid, marker='o', ms=3, label=f'{block} block'+('s' if block != 1 else ' (flat)'))
        ax.set(xlabel='Cotraining epoch', ylabel='Legacy Inception FID',
               title=f'CIFAR-10 CSEM: 8 latent channels, CFG={level:g}, RK4-20 per block')
        ax.legend(); fig.tight_layout()
        fig.savefig(compiled/f'fid_cfg{level:g}.png', dpi=180)
        fig.savefig(compiled/f'fid_cfg{level:g}.pdf'); plt.close(fig)
    print(last.to_string(index=False) if not last.empty else 'No measured results yet; no values inferred.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'submit', 'run', 'prepare-data', 'report', 'smoke'):
        p = commands.add_parser(name)
        p.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parent)
        if name in ('submit', 'run', 'prepare-data'):
            p.add_argument('--data-root', type=Path, default=os.environ.get('CSEM_DATA_ROOT') or None)
        if name == 'submit':
            p.add_argument('--cells', default='all'); p.add_argument('--dry-run', action='store_true')
            p.add_argument('--restart-incomplete', action='store_true')
        if name == 'run':
            p.add_argument('--cell-id', type=int, required=True)
            p.add_argument('--resume', action='store_true'); p.add_argument('--restart', action='store_true')
        if name == 'report':
            p.add_argument('--cfg', type=float, choices=(2.5, 3.0), default=2.5)
        if name == 'smoke':
            p.add_argument('--blocks', type=int, choices=(1, 2, 3), default=None)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    a = parser.parse_args(argv); base = a.base_dir.resolve()
    if a.command == 'validate':
        import tests
        return tests.run(base)
    if a.command == 'smoke':
        import tests
        return tests.torch_smoke(base, a.blocks, a.device)
    if a.command == 'prepare-data':
        prepare_data(base, a.data_root); return 0
    if a.command == 'submit':
        return submit_jobs(base, a.cells, dry_run=a.dry_run, restart=a.restart_incomplete, data_root=a.data_root)
    if a.command == 'run':
        return run_job(base, a.cell_id, resume=a.resume, restart=a.restart, data_root=a.data_root)
    return report(base, a.cfg)


if __name__ == '__main__':
    raise SystemExit(main())
