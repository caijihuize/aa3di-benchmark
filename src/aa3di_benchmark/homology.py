# SPDX-License-Identifier: Apache-2.0
"""Explicit Foldseek/MMseqs2 commands and frozen SCOPe catalog scoring."""
import subprocess
from pathlib import Path
import pandas as pd
from .catalog import calc_fp_rates_catalog
from .data import read_fasta
from .statistics import DI
from .runtime import identity, write_json

FORMAT = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits'


def execute(argv):
    print('[CMD]', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True)


def build_prediction_db(tool, aa, prediction, output):
    records = read_fasta(aa)
    preds = {r.identifier: r.sequence.upper() for r in read_fasta(prediction)}
    if set(preds) != {r.identifier for r in records}:
        raise ValueError('SCOPe prediction IDs differ from the full catalog')
    if any(set(s) - set(DI) for s in preds.values()):
        raise ValueError('Invalid predicted 3Di characters')
    output.parent.mkdir(parents=True, exist_ok=True)
    mismatches = [r.identifier for r in records if len(r.sequence) != len(preds[r.identifier])]
    # Decoder length mismatches remain raw, as in the frozen legacy protocol.
    write_json(output.parent / 'length_mismatches.json', mismatches)
    for name, suffix, dbtype in [('aa', '', '0'), ('di', '_ss', '0'), ('header', '_h', '12')]:
        tsv = output.parent / f'{name}.tsv'
        with tsv.open('w') as handle:
            for i, record in enumerate(records, 1):
                value = record.sequence if name == 'aa' else preds[record.identifier] if name == 'di' else record.identifier
                handle.write(f'{i}\t{value}\n')
        execute([tool, 'tsv2db', tsv, str(output)+suffix, '--output-dbtype', dbtype])
    return output


def search(tool, engine, db, output, params, threads, structure=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / (output.stem + '_tmp')
    common = ['--threads', str(threads), '-s', str(params['sensitivity']), '--max-seqs', str(params['max_seqs']), '-e', str(params['evalue'])]
    if engine == 'foldseek':
        execute([tool, 'easy-search', db, db, output, temp, *common,
                 '--sort-by-structure-bits', '1' if structure else '0', '--format-output', FORMAT])
    else:
        temp.mkdir(exist_ok=True)
        result = temp / 'result'
        execute([tool, 'search', db, db, result, temp / 'search', *common, '-a'])
        execute([tool, 'convertalis', db, db, result, output, '--threads', str(threads), '--format-output', FORMAT])
    return output


def score(labels, hits, prefix, expected_catalog=None, expected_queries=None):
    frame = pd.read_csv(labels, sep='\t', header=None, names=['id', 'fa'], dtype=str)
    if frame['id'].duplicated().any() or frame.isna().any().any():
        raise ValueError('Invalid/duplicate SCOPe labels')
    if expected_catalog is not None and len(frame) != expected_catalog:
        raise ValueError('Unexpected catalog count')
    frame['sf'] = frame['fa'].str.replace(r'\.[0-9]+$', '', regex=True)
    frame['fo'] = frame['sf'].str.replace(r'\.[0-9]+$', '', regex=True)
    paths = calc_fp_rates_catalog(Path(hits), frame, Path(prefix))
    result = {}
    for level, path in paths.items():
        values = pd.read_csv(path, sep='\t', header=None, names=['qid', 'tp', 'total', 'sensitivity'])
        if values.empty or (expected_queries is not None and len(values) != expected_queries):
            raise ValueError('Unexpected eligible query count')
        result[level] = {'mean_sensitivity': float(values['sensitivity'].mean()), 'n_queries': len(values)}
    result['query_metric_files'] = [identity(path) for path in paths.values()]
    write_json(Path(str(prefix)+'_summary.json'), result)
    return result
