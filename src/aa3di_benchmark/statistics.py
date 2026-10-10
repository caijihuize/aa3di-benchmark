# SPDX-License-Identifier: Apache-2.0
"""Paper metrics: strict coverage and paired, query-level uncertainty."""
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
DI = 'ALGVSREDTIPKFQNYMHWC'


def translation_metrics(reference, prediction, alphabet, reference_unknowns='X', case_sensitive=False):
    if not reference:
        raise ValueError('Empty reference')
    if set(prediction) - set(reference):
        raise ValueError('Unexpected prediction IDs')
    tp = dict.fromkeys(alphabet, 0)
    fp = dict.fromkeys(alphabet, 0)
    fn = dict.fromkeys(alphabet, 0)
    support = dict.fromkeys(alphabet, 0)
    matches = residues = exact = comparable = invalid = missing = mismatch = unknown_reference = predicted_residues = 0
    rows = []
    for qid, raw in reference.items():
        ref = raw if case_sensitive else raw.upper()
        if not ref or set(ref.upper()) - set(alphabet + reference_unknowns):
            raise ValueError(f'Noncanonical/empty reference: {qid}')
        for char in ref:
            if char in support:
                support[char] += 1
            else:
                unknown_reference += 1
        pred = prediction.get(qid)
        if pred is not None:
            invalid += sum(c not in alphabet for c in pred.upper())
            predicted_residues += len(pred)
        status = 'missing' if pred is None else ('length_mismatch' if len(pred) != len(ref) else 'scored')
        missing += status == 'missing'
        mismatch += status == 'length_mismatch'
        row = {'qid': qid, 'length': len(ref), 'status': status}
        if status == 'scored':
            pred = pred if case_sensitive else pred.upper()
            comparable += 1
            count = sum(a == b for a, b in zip(ref, pred))
            matches += count
            residues += len(ref)
            exact += count == len(ref)
            for a, b in zip(ref, pred):
                if a == b and a in tp:
                    tp[a] += 1
                else:
                    if a in fn:
                        fn[a] += 1
                    if b in fp:
                        fp[b] += 1
            row.update(n_match=count, accuracy=count / len(ref), exact=count == len(ref))
        rows.append(row)
    present = [c for c in alphabet if support[c] > 0]
    f1 = {c: 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) if 2 * tp[c] + fp[c] + fn[c] else 0.0 for c in present}
    return {
        'n_reference': len(reference), 'n_scored': comparable,
        'case_sensitive': case_sensitive,
        'coverage': comparable / len(reference), 'missing': missing, 'length_mismatch': mismatch,
        'micro_accuracy': matches / residues if residues else None,
        'macro_f1': sum(f1.values()) / len(present) if comparable else None,
        'macro_f1_policy': 'reference-present classes; zero denominator=0; equal-length outputs only',
        'reference_present_classes': present, 'class_f1': f1,
        'exact_match': exact / comparable if comparable else None,
        'exact_match_all_reference': exact / len(reference),
        'invalid_char_fraction': invalid / predicted_residues if predicted_residues else None,
        'unknown_reference_residues': unknown_reference,
        'unknown_reference_symbols': reference_unknowns,
        'unknown_reference_policy': 'noncanonical reference residues retained in Micro/EM; excluded as Macro-F1 classes; canonical predictions there count as FP',
        'n_scored_residues': residues,
    }, rows


def paired_bootstrap(candidate, baseline, replicates=10000, seed=42):
    if not candidate or set(candidate) != set(baseline):
        raise ValueError('Paired bootstrap requires identical nonempty query sets')
    ids = sorted(candidate)
    delta = np.array([candidate[q] - baseline[q] for q in ids], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite query score')
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 100):
        size = min(100, replicates - start)
        indices = rng.integers(0, len(delta), (size, len(delta)))
        means[start:start + size] = delta[indices].mean(axis=1)
    return {'mean_difference': float(delta.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'n_queries': len(ids), 'replicates': replicates, 'seed': seed,
            'unit': 'query', 'interval': 'paired percentile bootstrap'}
