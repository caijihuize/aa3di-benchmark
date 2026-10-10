# SPDX-License-Identifier: Apache-2.0
"""Independent commands for data, inference, scoring, speed and homology."""
import argparse
import json
import random
import time
from pathlib import Path
from .data import Record, read_fasta, write_fasta
from .runtime import Run, identity, write_json
from .statistics import AA, DI, paired_bootstrap, translation_metrics
from .translation import evaluate
from . import homology


def prepare(run, name):
    spec = run.config['datasets'][name]
    out = run.output / 'inputs' / name
    if spec['format'] == 'hf':
        from datasets import load_from_disk
        source = run.path(spec['source']) / spec.get('split', 'test')
        run.record(f'prepare/{name}', 'running', [source / 'state.json', *sorted(source.glob('*.arrow'))])
        ds = load_from_disk(str(source))
        aa = [Record(f'{i:06d}', ''.join(row['sequence_x'].split()).upper()) for i, row in enumerate(ds)]
        di = [Record(f'{i:06d}', ''.join(row['sequence_y'].split()).upper()) for i, row in enumerate(ds)]
    else:
        sources = [run.path(spec['aa']), run.path(spec['di'])]
        run.record(f'prepare/{name}', 'running', sources)
        aa, di = map(read_fasta, sources)
    if len(aa) != spec['expected_count'] or [(r.identifier,len(r.sequence)) for r in aa] != [(r.identifier,len(r.sequence)) for r in di]:
        raise ValueError('Reference IDs, lengths or expected counts differ')
    for records, alphabet in [(aa, AA), (di, DI)]:
        refs = {r.identifier: r.sequence for r in records}
        translation_metrics(refs, refs, alphabet, reference_unknowns='XBUZO' if alphabet == AA else 'X')
    write_fasta(aa, out / 'aa.fasta')
    write_fasta(di, out / 'di.fasta')
    # Complete the same task with the original fingerprints retained.
    task = json.loads((run.output / 'manifest.json').read_text())['tasks'][f'prepare/{name}']
    run.record(f'prepare/{name}', 'completed', [r['path'] for r in task['inputs']], {'n_sequences': len(aa)})


def prepare_speed(run):
    cfg = run.config['speed']
    source = run.output / 'inputs' / cfg['source_dataset'] / 'aa.fasta'
    records = read_fasta(source)
    rng = random.Random(run.config['seed'])
    run.record('prepare_speed', 'running', [source])
    selected = {}
    for length in cfg['lengths']:
        eligible = sorted([r for r in records if len(r.sequence) >= length], key=lambda r: r.identifier)
        sample = rng.sample(eligible, cfg['per_length'])
        write_fasta([Record(r.header, r.sequence[:length]) for r in sample], run.output / 'inputs/speed' / f'{length}.fasta')
        selected[str(length)] = [{'id': r.identifier, 'original_length': len(r.sequence)} for r in sample]
    run.record('prepare_speed', 'completed', [source], {'policy': 'deterministic prefixes; speed only', 'samples': selected})


def model_inputs(run, method, direction):
    spec = run.config['methods'][method]
    if spec['backend'] == 'prostt5':
        checkpoint = run.path(spec['checkpoint'])
        if not checkpoint.is_dir():
            raise ValueError('ProstT5 local model directory is missing')
        files = sorted(p for p in checkpoint.rglob('*') if p.is_file())
        if not files:
            raise ValueError('Empty model directory')
        return files
    base = run.path(spec['base_checkpoint'])
    selected = run.path(spec['selection_record'])
    record = json.loads(selected.read_text())
    if record.get('stage1_training_records') != 17070828 or record.get('stage2_dataset') != 'pdb_v2_structure':
        raise ValueError('Selection record does not establish full17M/PDB lineage')
    if not record.get('selection_rule') or 'test_used_for_selection' not in record:
        raise ValueError('Checkpoint selection and test-set usage must be disclosed')
    paths = {'base_checkpoint': base}
    if spec['stage'] == 2:
        paths[f'{direction}_adapter'] = run.path(spec[f'{direction}_adapter'])
    for key, path in paths.items():
        if record.get('sha256', {}).get(key) != identity(path)['sha256']:
            raise ValueError(f'Selected checkpoint hash differs: {key}')
    return [selected, *paths.values()]


def build_runner(run, method, direction, device):
    spec = run.config['methods'][method]
    common = {'direction': direction, 'precision': run.config['precision'], 'device': device}
    if spec['backend'] == 'prostt5':
        from .models.prostt5 import ProstT5DecoderRunner
        runner = ProstT5DecoderRunner(model_path=str(run.path(spec['checkpoint'])), **common)
    elif spec['backend'] == 'esm3':
        from .models.esm3 import ESM3Runner
        adapter = str(run.path(spec[f'{direction}_adapter'])) if spec['stage'] == 2 else None
        runner = ESM3Runner(base_checkpoint=str(run.path(spec['base_checkpoint'])), adapter=adapter, stage=spec['stage'], **common)
    else:
        raise ValueError('Unknown model backend')
    runner.method_id = method
    runner.model_name = method
    runner.variant = f'{method}-{direction}'
    return runner


def predict(run, method, dataset, direction, device, speed=False, length=None):
    import torch
    from .speed import run_benchmark
    if speed:
        source = run.output / 'inputs/speed' / f'{length}.fasta'
        reference = None
        out = run.output / 'speed' / method / str(length)
        cfg = run.config['speed']
        repeats, warmup = cfg['repeats'], cfg['warmup_batches']
        if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Paper speed measurements require an allocated CUDA GPU')
    else:
        base = run.output / 'inputs' / dataset
        source = base / ('aa.fasta' if direction == 'aa2di' else 'di.fasta')
        reference = base / ('di.fasta' if direction == 'aa2di' else 'aa.fasta')
        out = run.output / 'inference' / dataset / method / direction
        repeats, warmup = 1, 0
    task = f'speed/{method}/{length}' if speed else f'infer/{dataset}/{method}/{direction}'
    weights = model_inputs(run, method, direction)
    inputs = [source, *([reference] if reference else []), *weights]
    if run.config['methods'][method]['backend'] == 'prostt5':
        bindings = {f'model/{method}': weights}
    else:
        bindings = {'esm3/shared_base': weights[:2], f'model/{method}/{direction}': weights}
    run.record(task, 'running', inputs, bindings=bindings)
    random.seed(run.config['seed'])
    torch.manual_seed(run.config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run.config['seed'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_device(device)
    runner = build_runner(run, method, direction, device)
    started = time.perf_counter()
    runner.load()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    loaded = time.perf_counter()-started
    summary = run_benchmark(runner=runner, records=read_fasta(source), output_dir=out,
        batch_size=run.config['batch_size'], max_residues=0, warmup_batches=warmup,
        repeats=repeats, load_seconds=loaded, valid_characters=runner.valid_characters,
        references=read_fasta(reference) if reference else None)
    if reference:
        evaluate(reference, out / 'predictions.fasta', direction,
                 run.output / 'translation' / dataset / method / direction, method=method, dataset=dataset)
    run.record(task, 'completed', inputs, {'summary': identity(out / 'summary.json'), 'environment': summary['environment']})


def run_homology(run, method):
    cfg = run.config['homology']
    labels = run.path(cfg['labels'])
    if method in ('foldseek', 'mmseqs'):
        engine = method
        db = run.path(cfg['structure_db' if method == 'foldseek' else 'sequence_db'])
        inputs = [labels]
    else:
        engine = 'foldseek'
        aa = run.output / 'inputs/scope208/aa.fasta'
        pred = run.output / 'inference/scope208' / method / 'aa2di/predictions.fasta'
        inputs = [labels, aa, pred]
        db = run.output / 'homology/db' / method / 'DB'
    tool = run.path(cfg[engine])
    version = subprocess_version(tool)
    if version != cfg[f'{engine}_version']:
        raise ValueError(f'{engine} version differs: {version}')
    task = f'homology/{method}'
    if method in ('foldseek', 'mmseqs'):
        inputs += sorted(p for p in db.parent.glob(db.name+'*') if p.is_file())
        if not db.is_file():
            raise ValueError('Baseline database is missing')
    run.record(task, 'running', inputs)
    if method not in ('foldseek', 'mmseqs'):
        homology.build_prediction_db(tool, aa, pred, db)
    hits = run.output / 'homology/aln' / f'{method}.tsv'
    homology.search(tool, engine, db, hits, cfg[f'{engine}_search'], cfg['threads'], structure=method=='foldseek')
    result = homology.score(labels, hits, run.output / 'homology/metrics' / method,
                            cfg['catalog_count'], cfg['eligible_queries'])
    run.record(task, 'completed', inputs, {'tool_version': version, 'alignment': identity(hits), 'metrics': result})


def subprocess_version(tool):
    import subprocess
    return subprocess.run([str(tool), 'version'], capture_output=True, text=True, check=True).stdout.strip()


def smoke(run):
    for name in run.config['datasets']:
        prepare(run, name)
    prepare_speed(run)
    cfg = run.config['smoke']
    prediction, labels, hits = [run.path(cfg[k]) for k in ('prediction','labels','hits')]
    run.record('smoke', 'running', [prediction, labels, hits])
    evaluate(run.output / 'inputs/tiny/di.fasta', prediction, 'aa2di',
             run.output / 'translation/tiny/synthetic/aa2di', method='synthetic', dataset='tiny')
    homology.score(labels, hits, run.output / 'homology/metrics/synthetic', 8, 4)
    run.record('smoke', 'completed', [prediction, labels, hits], {'scientific_result': False})
    from .reporting import report
    report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='runs/current')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    p = sub.add_parser('prepare'); p.add_argument('--dataset', required=True)
    sub.add_parser('prepare-speed')
    p=sub.add_parser('prepare-scope');p.add_argument('--archive',required=True);p.add_argument('--classification',required=True)
    for command in ('infer', 'speed'):
        p = sub.add_parser(command)
        p.add_argument('--method', required=True)
        p.add_argument('--device', default='cuda:0')
        if command == 'infer':
            p.add_argument('--dataset', required=True)
            p.add_argument('--direction', required=True, choices=['aa2di','di2aa'])
    p = sub.add_parser('evaluate')
    p.add_argument('--reference', required=True); p.add_argument('--prediction', required=True)
    p.add_argument('--direction', required=True, choices=['aa2di','di2aa'])
    p.add_argument('--method', required=True); p.add_argument('--dataset', required=True)
    p = sub.add_parser('homology'); p.add_argument('--method', required=True)
    p = sub.add_parser('score-homology')
    p.add_argument('--labels', required=True); p.add_argument('--hits', required=True); p.add_argument('--method', required=True)
    p = sub.add_parser('embedding'); p.add_argument('--method',required=True); p.add_argument('--input',required=True); p.add_argument('--device',default='cuda:0')
    sub.add_parser('report'); sub.add_parser('smoke')
    args = parser.parse_args()
    for key in ('method','dataset'):
        value = getattr(args, key, None)
        if value is not None and (Path(value).name != value or value in ('.','..')):
            parser.error(f'{key} must be a simple identifier')
    run = Run(args.config, args.output)
    if args.command == 'plan':
        print(json.dumps({'protocol': run.config['protocol'], 'methods': list(run.config['methods']),
                          'datasets': list(run.config['datasets']), 'output': str(run.output)}, indent=2))
        return
    try:
        if args.command == 'prepare': prepare(run, args.dataset)
        elif args.command == 'prepare-speed': prepare_speed(run)
        elif args.command == 'prepare-scope':
            from .scope import prepare as prepare_scope
            prepare_scope(run,Path(args.archive).resolve(),Path(args.classification).resolve())
        elif args.command == 'infer': predict(run, args.method, args.dataset, args.direction, args.device)
        elif args.command == 'speed':
            for length in run.config['speed']['lengths']:
                predict(run, args.method, None, run.config['speed']['direction'], args.device, True, length)
        elif args.command == 'evaluate':
            inputs = [Path(args.reference).resolve(), Path(args.prediction).resolve()]
            task = f'evaluate/{args.dataset}/{args.method}/{args.direction}'
            run.record(task, 'running', inputs)
            result = evaluate(*inputs, args.direction, run.output/'translation'/args.dataset/args.method/args.direction, method=args.method, dataset=args.dataset)
            run.record(task, 'completed', inputs, result)
        elif args.command == 'homology': run_homology(run, args.method)
        elif args.command == 'score-homology':
            inputs = [Path(args.labels).resolve(), Path(args.hits).resolve()]
            task = f'score_homology/{args.method}'
            run.record(task, 'running', inputs)
            result = homology.score(*inputs, run.output/'homology/metrics'/args.method)
            run.record(task, 'completed', inputs, result)
        elif args.command == 'embedding':
            from .embedding import benchmark
            benchmark(run,args.method,Path(args.input).resolve(),args.device)
        elif args.command == 'smoke': smoke(run)
        else:
            from .reporting import report
            report(run)
    except Exception as exc:
        # Preserve failures without converting partial scientific work into success.
        path = run.output/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            for value in manifest['tasks'].values():
                if value['status'] == 'running':
                    value.update(status='failed', error=str(exc))
            write_json(path, manifest)
        raise


if __name__ == '__main__':
    main()
