# SPDX-License-Identifier: Apache-2.0
"""Report available results; keep absent tasks and partial runs explicit."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .runtime import identity, write_json
from .statistics import paired_bootstrap


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def query_scores(path):
    rows = list(csv.reader(path.open(), delimiter='\t'))
    values = {r[0]: float(r[3]) for r in rows}
    if len(values) != len(rows):
        raise ValueError('Duplicate query scores')
    return values


def report(run):
    run.record('report', 'running')
    out = run.output/'reports'
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((run.output/'manifest.json').read_text())['tasks']
    def completed(method, keys):
        return any(tasks.get(k, {}).get('status') == 'completed' for k in keys) or (method == 'synthetic' and tasks.get('smoke', {}).get('status') == 'completed')
    translations = []
    for path in sorted((run.output/'translation').glob('*/*/*/summary.json')):
        row = json.loads(path.read_text())
        suffix = f"{row['dataset']}/{row['method']}/{row['direction']}"
        if not completed(row['method'], ['infer/'+suffix, 'evaluate/'+suffix]):
            continue
        for key in ('reference', 'prediction'):
            if identity(row[key]['path']) != row[key]:
                raise ValueError('Translation inputs changed after evaluation')
        translations.append(row)
    remote, paired, speed, artifacts = [], [], [], []
    prefixes = sorted((run.output/'homology/metrics').glob('*_summary.json'))
    common_queries = None
    remote_scores = {}
    for summary in prefixes:
        method = summary.name.removesuffix('_summary.json')
        if not completed(method, [f'homology/{method}', f'score_homology/{method}']):
            continue
        for artifact in json.loads(summary.read_text())['query_metric_files']:
            if identity(artifact['path']) != artifact:
                raise ValueError('Query metrics changed after scoring')
        remote_scores[method] = {}
        for level in ('fam','sup','fol'):
            path = summary.with_name(f'{method}_{level}.tsv')
            values = query_scores(path)
            if not values or (common_queries is not None and set(values) != common_queries):
                raise ValueError('Remote results have different query sets')
            common_queries = set(values)
            remote_scores[method][level] = values
            remote.append({'method':method,'level':level,'mean_sensitivity':sum(values.values())/len(values),'n_queries':len(values)})
            artifacts.append(identity(path))
    pending = []
    for candidate, baseline in run.config.get('comparisons', []):
        if candidate not in remote_scores or baseline not in remote_scores:
            pending.append([candidate,baseline]); continue
        for level in ('fam','sup','fol'):
            paired.append(dict(candidate=candidate, baseline=baseline, level=level,
                **paired_bootstrap(remote_scores[candidate][level],remote_scores[baseline][level],**run.config['bootstrap'])))
    gpu_ids = set()
    for path in sorted((run.output/'speed').glob('*/*/summary.json')):
        row = json.loads(path.read_text())
        task = tasks.get(f"speed/{row['method_id']}/{path.parent.name}", {})
        if task.get('status') != 'completed':
            continue
        if task['details']['summary'] != identity(path):
            raise ValueError('Speed summary changed after completion')
        cfg = run.config['speed']; env = row['environment']
        uuid = env.get('gpu_uuid') or env.get('cuda_visible_devices')
        if row['batch_size'] != run.config['batch_size'] or row['repeats'] != cfg['repeats'] or row['n_sequences'] != cfg['per_length'] or not uuid or not env.get('gpu_name'):
            raise ValueError('Speed protocol/GPU identity is missing or differs')
        if row['runner_metadata']['precision'] != run.config['precision']:
            raise ValueError('Speed precision differs')
        gpu_ids.add((env['hostname'],env['gpu_name'],uuid))
        speed.append({'method':row['method_id'],'length':int(path.parent.name),
                      'latency_ms':row['milliseconds_per_protein'],'peak_allocated_gib':row['peak_allocated_bytes']/2**30,
                      'load_seconds':row['load_seconds'],'proteins_per_second':row['sequences_per_second']})
        artifacts.append(identity(path))
    if len(gpu_ids)>1:
        raise ValueError('Speed results do not use the same physical GPU; separate the runs')
    if speed:
        fig, axes = plt.subplots(1,2,figsize=(9,3.5))
        for method in sorted({r['method'] for r in speed}):
            rows = sorted([r for r in speed if r['method']==method],key=lambda r:r['length'])
            for ax,metric,label in [(axes[0],'latency_ms','Latency (ms/protein)'),(axes[1],'peak_allocated_gib','Peak allocated memory (GiB)')]:
                ax.plot([r['length'] for r in rows],[r[metric] for r in rows],marker='o',label=method)
                ax.set(xlabel='Input residues',ylabel=label)
        axes[0].legend(fontsize=6);fig.tight_layout()
        for suffix in ('png','svg'):fig.savefig(out/f'length_latency_memory.{suffix}',dpi=180)
        plt.close(fig)
    if remote:
        fig, axes = plt.subplots(1,3,figsize=(11,3))
        for ax,level in zip(axes,('fam','sup','fol')):
            for method,levels in remote_scores.items():
                values=sorted(levels[level].values(),reverse=True)
                ax.plot([(i+1)/len(values) for i in range(len(values))],values,label=method)
            ax.set(title=level,xlabel='Ranked query fraction',ylabel='Sensitivity')
        axes[0].legend(fontsize=6);fig.tight_layout()
        for suffix in ('png','svg'):fig.savefig(out/f'remote_sensitivity.{suffix}',dpi=180)
        plt.close(fig)
    tables={'translation':translations,'remote':remote,'paired_differences':paired,'speed':speed}
    manifest=json.loads((run.output/'manifest.json').read_text())
    result={'status':'partial','paper_ready':False,'protocol':run.config['protocol'],
            'completed_tasks':[k for k,v in manifest['tasks'].items() if v['status']=='completed'],
            'pending_comparisons':pending,'result_sources':artifacts,**tables}
    write_json(out/'results.json',result)
    for name,rows in tables.items():write_csv(out/f'{name}.csv',rows)
    lines=['# AA–3Di benchmark results','','Status: partial. This report contains available tasks; it does not certify that the entire paper is complete.','',
           '| Dataset | Method | Direction | Micro | Macro-F1 | EM | Coverage |','|---|---|---|---:|---:|---:|---:|']
    fmt=lambda v:'NA' if v is None else f'{v:.6f}'
    for r in translations:
        lines.append('|'+ '|'.join(str(r.get(k,'')) for k in ('dataset','method','direction'))+'|'+ '|'.join(fmt(r[k]) for k in ('micro_accuracy','macro_f1','exact_match','coverage'))+'|')
    lines += ['','| Method | Level | Mean sensitivity | Queries |','|---|---|---:|---:|']
    for r in remote:lines.append(f"|{r['method']}|{r['level']}|{r['mean_sensitivity']:.6f}|{r['n_queries']}|")
    lines += ['','| Candidate | Baseline | Level | Mean difference | Paired 95% CI |','|---|---|---|---:|---|']
    for r in paired:lines.append(f"|{r['candidate']}|{r['baseline']}|{r['level']}|{r['mean_difference']:.6f}|{r['ci95']}|")
    lines += ['','| Method | Length | ms/protein | Peak GiB |','|---|---:|---:|---:|']
    for r in speed:lines.append(f"|{r['method']}|{r['length']}|{r['latency_ms']:.3f}|{r['peak_allocated_gib']:.3f}|")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    run.record('report','completed',details={'results':identity(out/'results.json')})
