# SPDX-License-Identifier: Apache-2.0
"""Evaluate raw predictions without repairing missing or mismatched outputs."""
from .data import read_fasta
from .statistics import AA, DI, translation_metrics
from .runtime import identity, write_json


def evaluate(reference, prediction, direction, output, **labels):
    refs = {r.identifier: r.sequence for r in read_fasta(reference)}
    preds = {r.identifier: r.sequence for r in read_fasta(prediction)}
    metrics, rows = translation_metrics(refs, preds, DI if direction == 'aa2di' else AA,
                                       reference_unknowns='X' if direction == 'aa2di' else 'XBUZO',
                                       case_sensitive=direction == 'di2aa')
    metrics.update(direction=direction, reference=identity(reference), prediction=identity(prediction), **labels)
    write_json(output / 'summary.json', metrics)
    write_json(output / 'per_sequence.json', rows)
    return metrics
