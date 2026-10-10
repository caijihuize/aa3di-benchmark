# SPDX-License-Identifier: Apache-2.0
"""Prepare the frozen SCOPe snapshot from explicitly supplied source files."""
import tarfile
from pathlib import Path
from .data import read_fasta
from .homology import execute
from .runtime import write_json


def prepare(run, archive, classification):
    from .cli import subprocess_version
    cfg=run.config['homology']
    foldseek,mmseqs=run.path(cfg['foldseek']),run.path(cfg['mmseqs'])
    for name,tool in [('foldseek',foldseek),('mmseqs',mmseqs)]:
        if subprocess_version(tool)!=cfg[f'{name}_version']:raise ValueError(f'{name} version differs')
    run.record('prepare_scope','running',[archive,classification])
    classes={}
    for line in Path(classification).read_text().splitlines():
        if line and not line.startswith('#'):
            fields=line.split()
            if len(fields)>=4:classes[fields[0]]=fields[3]
    root=run.output/'scope208'
    single=root/'structures_single';single.mkdir(parents=True,exist_ok=True)
    seen=set();excluded=0
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.ent'):continue
            domain=Path(member.name).stem
            if domain in seen:raise ValueError(f'Duplicate structure name: {domain}')
            seen.add(domain)
            with tar.extractfile(member) as handle:body=handle.read()
            lines=body.decode('ascii',errors='replace').splitlines()
            if domain not in classes or sum(line.startswith('MODEL ') for line in lines)>1 or not any(line.startswith('ATOM') or line.startswith('HETATM') for line in lines):
                excluded+=1;continue
            (single/f'{domain}.ent').write_bytes(body)
    fdb=root/'DB/foldseek_DB/DB';mdb=root/'DB/mmseqs_DB/DB'
    fdb.parent.mkdir(parents=True,exist_ok=True);mdb.parent.mkdir(parents=True,exist_ok=True)
    gt=root/'GT_fasta';gt.mkdir(exist_ok=True)
    execute([foldseek,'createdb',single,fdb,'--threads',str(cfg['threads'])])
    execute([foldseek,'lndb',str(fdb)+'_h',str(fdb)+'_ss_h'])
    execute([foldseek,'convert2fasta',fdb,gt/'DB_aa.fasta'])
    execute([foldseek,'convert2fasta',str(fdb)+'_ss',gt/'DB_di.fasta'])
    aa,di=read_fasta(gt/'DB_aa.fasta'),read_fasta(gt/'DB_di.fasta')
    if len(aa)!=cfg['catalog_count'] or [(r.identifier,len(r.sequence)) for r in aa]!=[(r.identifier,len(r.sequence)) for r in di]:
        raise ValueError('SCOPe input count or paired IDs/lengths differ')
    labels=root/'labels/scop_lookup.tsv';labels.parent.mkdir(exist_ok=True)
    rows=[]
    for record in aa:
        key=record.identifier
        family=classes.get(key) or classes.get(key.rsplit('_',1)[0])
        if not family:raise ValueError(f'Unclassified Foldseek record: {key}')
        rows.append(f'{key}\t{family}\n')
    labels.write_text(''.join(rows))
    execute([mmseqs,'createdb',gt/'DB_aa.fasta',mdb,'--threads',str(cfg['threads'])])
    run.record('prepare_scope','completed',[archive,classification],{'root':str(root),'n_records':len(aa),'excluded_structures':excluded})
    print(f'Set AA3DI_SCOPE_ROOT={root}')
