# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""SCOPe catalog evaluator compatible with Foldseek's bench.noselfhit.awk."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def calc_fp_rates_catalog(aln_tsv_path: Path, cla: pd.DataFrame, out_prefix: Path) -> dict[str, Path]:
    """Evaluate ordered hits until the first wrong-fold hit for each query.

    The remote-level denominators deliberately reproduce the upstream AWK:
    sfamCnt - (famCnt - 1) and foldCnt - (sfamCnt - 1). They therefore include
    the query itself in those two denominators. This is retained for benchmark
    compatibility and recorded explicitly in the report.
    """
    id2fam = dict(zip(cla["id"].astype(str), cla["fa"].astype(str)))
    id2sfam = dict(zip(cla["id"].astype(str), cla["sf"].astype(str)))
    id2fold = dict(zip(cla["id"].astype(str), cla["fo"].astype(str)))
    fam_cnt = Counter(cla["fa"].astype(str))
    sfam_cnt = Counter(cla["sf"].astype(str))
    fold_cnt = Counter(cla["fo"].astype(str))
    found_fam: dict[str, int] = defaultdict(int)
    found_sfam: dict[str, int] = defaultdict(int)
    found_fold: dict[str, int] = defaultdict(int)
    found_fp: dict[str, int] = defaultdict(int)

    with Path(aln_tsv_path).open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            qid, tid = parts[0], parts[1]
            if qid not in id2fam:
                continue
            if tid not in id2fam:
                found_fp[qid] += 1
                continue
            if qid == tid:
                continue
            if found_fp[qid] >= 1:
                continue
            if id2fold[qid] != id2fold[tid]:
                found_fp[qid] += 1
            elif id2fam[qid] == id2fam[tid]:
                found_fam[qid] += 1
            elif id2sfam[qid] == id2sfam[tid]:
                found_sfam[qid] += 1
            else:
                found_fold[qid] += 1

    rows: dict[str, list[tuple[str, int, int, float]]] = {"fam": [], "sup": [], "fol": []}
    for qid, fam in id2fam.items():
        sfam, fold = id2sfam[qid], id2fold[qid]
        n_fam, n_sfam, n_fold = fam_cnt[fam], sfam_cnt[sfam], fold_cnt[fold]
        if not (fam and n_fam > 1 and n_sfam - n_fam > 0 and n_fold - n_sfam > 0):
            continue
        denominators = {
            "fam": n_fam - 1,
            "sup": n_sfam - (n_fam - 1),
            "fol": n_fold - (n_sfam - 1),
        }
        counts = {"fam": found_fam[qid], "sup": found_sfam[qid], "fol": found_fold[qid]}
        for level in ("fam", "sup", "fol"):
            total = denominators[level]
            rows[level].append((qid, counts[level], total, counts[level] / total))

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {level: Path(f"{out_prefix}_{level}.tsv") for level in ("fam", "sup", "fol")}
    for level, path in paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for qid, count, total, fraction in rows[level]:
                handle.write(f"{qid}\t{count}\t{total}\t{fraction:.6f}\n")
    return paths
