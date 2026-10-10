# Frozen evaluation protocol

## RQ1: bidirectional translation

Evaluate the Stage 1 source test (474) and PDB structure test (500) separately, in both directions. Do not replace these with SCOPe translation results. The source datasets use paired `sequence_x` (AA) and `sequence_y` (3Di); exported IDs are deterministic row numbers. Full17M means 17,070,828 training records; it does not mean Swiss-Prot. Stage 2 uses `pdb_v2_structure`.

Micro is residue accuracy on equal-length available predictions. Macro-F1 averages only the 20 canonical classes that occur in the entire reference set; a zero denominator gives F1=0. Exact match uses the equal-length scored subset, with an additional all-reference value. Missing and mismatched sequences are excluded from quality denominators and disclosed with coverage. Unexpected prediction IDs are rejected; invalid prediction characters are counted, never repaired.

Reference `X` is retained in Micro/EM and is not a 21st Macro-F1 class. A canonical prediction at reference X contributes a false positive for the predicted class. The PDB test snapshot contains 7 such 3Di residues. Coverage and reference-present classes must accompany the score.

ProstT5 uses one candidate and the existing paper generation preset: forward sampling, 3 beams, top-p=.95, temperature=1.2, top-k=6, repetition penalty=1.2, length penalty=1, early stopping; reverse sampling, top-p=.85, temperature=1, top-k=3, repetition penalty=1.2. Seed=42; source length determines the frozen min/max token lengths. No silent output length repair or selection by downstream scores.

Stage 1 control uses the selected shared base, source-side inputs and target-side masks, the joint `aadi` gate, and no direction adapter. Stage 2 uses the corresponding PDB direction adapter and gate. ESM3 checkpoint loading rejects missing required base tensors; LoRA-only missing keys are expected when loading the base. The bundled architecture uses installed `esm` modules, with explicit standard token IDs and compatible pre/post-normalization transformer output handling.

## RQ2: remote homology

SCOPe 2.08, 40% identity selection, 13,920 catalog entries and 5,046 common eligible queries. Ordered self-hits are ignored. Count correct hits only before the first wrong-fold target. Unknown targets stop the query. Preserve ranking order from the search tool.

With family, superfamily and fold catalog sizes nf/ns/no, denominators are nf−1, ns−nf+1 and no−ns+1. Eligible queries require nf>1, ns−nf>0 and no−ns>0. These sensitivity measures are not ordinary recall or ROC-AUC. Identical query sets are required across all compared methods.

Foldseek 10-941cd33: sensitivity=9.5, max-seqs=2000, E=10. Structure baseline uses structure-bit reranking; predicted AA+3Di databases do not. MMseqs2 18-8cc5c: sensitivity=7.5, max-seqs=2000, E=10000, backtrace enabled. Both search their own full catalog databases.

Prediction IDs must exactly match the catalog and use valid 3Di symbols. Raw decoder length mismatches are retained for compatibility with the existing search protocol and separately recorded. Search engines may still fail on malformed inputs; failures remain explicit.

Paired query bootstrap uses 10,000 replicates, seed42 and percentile 95% intervals on the mean candidate-minus-baseline difference. Do not infer paired significance from overlap of marginal confidence intervals. Query-level intervals do not capture training-seed variation.

## RQ5: inference efficiency

One allocated GPU, FP16 weights and compute, batch=1; lengths 64/128/256/384/510, 10 inputs each, 3 warm-up batches and 5 repeats. Inputs are deterministic SCOPe prefixes, used only for timing. CUDA events measure GPU forward execution; wall timing includes preprocessing, generation and decoding. Model load and artifact IO are separate. Peak allocated/reserved CUDA memory and GPU UUID/allocation identity are recorded.

Different models use separate environments; identical GPU and precision do not imply identical software stacks. Report records dependency versions. Sampled generation and CUDA operations are not promised to be bitwise identical across hardware/library versions.

The separate p3Di task generates 3Di from AA and encodes it with the reverse model. It returns mean residue representations before final layer normalization, excluding BOS/EOS. Both direction models remain resident. This is not reverse AA reconstruction, and neither task is evidence of structural recovery. Parameter and checkpoint storage details are saved in model/run metadata; serialized adapter counts are not automatically training counts.

AA references may additionally contain B/U/Z/O; retain them in Micro/EM, exclude them as Macro-F1 classes, and disclose their count. The legacy SCOPe snapshot has one Z. This preserves the old full-reference recovery denominator without silently normalizing the reference.

Casing is explicit: AA→3Di compares uppercase strings; 3Di→AA keeps the legacy case-sensitive AA comparison. The SCOPe reference has six lowercase AA residues. Under strict AA scoring these are retained as noncanonical reference labels rather than silently converted; Micro includes them and Macro-F1 remains restricted to canonical uppercase reference classes. The 474/500 source tests are already uppercase. This preserves legacy Micro results; a future case-normalized protocol needs a new run/config identity and must not be mixed into the old table.
