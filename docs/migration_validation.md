# Migration validation — 2026-10-10

This clean repository was created alongside the legacy experiment rather than overwriting it. The historical reporting directory with an underscore name was also left intact. No new GPU jobs were submitted during cleanup.

## Completed validation

- Built and installed a wheel into a fresh Python3.10 virtual environment without system packages or GPU/model dependencies.
- Seven core tests cover Micro vs Macro-F1, coverage/noncanonical references, paired bootstrap query alignment, duplicate FASTA IDs, first-wrong-fold catalog scoring, changed-input and cross-task model-identity refusal and the end-to-end synthetic CLI/report workflow.
- Synthetic smoke works from an unrelated working directory after installation; it generates Markdown/CSV/JSON and PNG/SVG artifacts using only this repository's fixtures.
- Re-evaluated the full legacy ProstT5 SCOPe predictions in both directions. Micro scores match exactly: forward0.6697107871825658, reverse0.3713160441152338. Each direction has two length mismatches; those outputs remain raw and coverage is reported.
- New Macro-F1 is computed separately from the legacy per-sequence average accuracy. The SCOPe reference AA has one Z; this is retained and counted rather than removed.
- The migrated evaluator is the frozen catalog implementation previously checked against the upstream AWK fixture. Full ProstT5 query-level files are also compared with the preserved legacy results during migration.
- The bundled standalone ESM3 architecture was instantiated on a meta device. All required base tensor keys/shapes match the existing Stage 1 checkpoint; only the expected384 LoRA A/B keys are absent from the base. No GPU or full tensor allocation was used for this check.

## Remaining scientific validation

Full17M/PDB final model choices are not yet frozen. Meta shape compatibility does not establish numerical or GPU performance equivalence, and no new GPU generation/timing has been completed in this repository. Test the selected checkpoints and shared FP16 stability before publishing final scientific claims.

The optional ESM dependency replaces the old private source checkout. Its transformer return layout is handled explicitly; real output parity needs a fixed-weight inference comparison. Exact processed PDB/source splits and final model access instructions remain separate scientific artifacts, not pretend download links.

The final report remains `partial` and `paper_ready=false`. RQ3/RQ4/RQ6 remain outside this repository. Old30% results are developmental evidence, not complete17M paper results.

A parity check also exposed six lowercase residues in the SCOPe AA reference. Retaining the legacy case-sensitive reverse comparison restores exact Micro parity; silently uppercasing them would add three matching residues. The explicit casing rule is recorded in each translation summary and documented in the protocol.
