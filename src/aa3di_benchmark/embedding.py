# SPDX-License-Identifier: Apache-2.0
"""AA -> predicted 3Di -> reverse encoder representation (not AA recovery)."""
import random
import statistics
import time
import torch
from .data import read_fasta
from .runtime import identity, write_json
from .speed import environment_metadata


def benchmark(run, method, source, device):
    from .cli import build_runner, model_inputs
    spec=run.config['methods'][method]
    if spec['backend']!='esm3' or spec['stage']!=2:
        raise ValueError('p3Di representation requires two Stage 2 direction adapters')
    if not torch.cuda.is_available() or torch.device(device).type!='cuda':
        raise ValueError('Representation timing requires an allocated CUDA GPU')
    records=read_fasta(source)
    inputs=[source,*model_inputs(run,method,'aa2di'),*model_inputs(run,method,'di2aa')]
    task=f'embedding/{method}'
    run.record(task,'running',inputs)
    random.seed(run.config['seed']);torch.manual_seed(run.config['seed'])
    torch.cuda.set_device(device)
    started=time.perf_counter()
    forward=build_runner(run,method,'aa2di',device);reverse=build_runner(run,method,'di2aa',device)
    forward.load();reverse.load();torch.cuda.synchronize(device)
    loaded=time.perf_counter()-started

    @torch.inference_mode()
    def encode(record):
        batch=[record]
        predicted=forward.decode(forward.forward(forward.prepare(batch)),batch)
        if len(predicted[0].sequence)!=len(record.sequence):
            raise ValueError('Predicted 3Di length mismatch')
        with torch.autocast('cuda',dtype=reverse.dtype,enabled=run.config['precision']!='fp32'):
            out=reverse.model(**reverse.prepare(predicted),fusion_mode='di2aa',active_heads=())
        pooled=out['embedding'][0,1:len(record.sequence)+1].mean(dim=0)
        if not torch.isfinite(pooled).all().item():raise FloatingPointError('Nonfinite embedding')
        return pooled

    cfg=run.config['speed']
    for record in records[:cfg['warmup_batches']]:encode(record)
    rows=[];canonical=None
    for repeat in range(cfg['repeats']):
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter();features=[encode(r) for r in records]
        torch.cuda.synchronize(device)
        rows.append({'repeat':repeat,'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        if repeat==0:canonical=torch.stack([f.cpu() for f in features])
        del features
    out=run.output/'embedding'/method;out.mkdir(parents=True,exist_ok=True)
    torch.save({'ids':[r.identifier for r in records],'embeddings':canonical},out/'embeddings.pt')
    seconds=statistics.median(r['seconds'] for r in rows)
    summary={'task':'aa_to_p3di_reverse_embedding','method':method,'n_sequences':len(records),
             'load_seconds':loaded,'milliseconds_per_protein':1000*seconds/len(records),
             'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),'repeats':rows,
             'representation':'pre-final-normalization; residue mean, excluding BOS/EOS',
             'memory_policy':'both direction models resident; pooled GPU vectors retained per repeat',
             'timed_file_io':False,'timed_cpu_transfer':False,'environment':environment_metadata(torch.device(device))}
    write_json(out/'summary.json',summary)
    run.record(task,'completed',inputs,{'summary':identity(out/'summary.json')})
