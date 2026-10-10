# Reproduction and data preparation

The core package is installable without GPU/model libraries. Optional model environments are independent. Core versions are pinned in `pyproject.toml`; ProstT5 versions match the previous model environment. ESM3 uses `esm==3.2.1.post1`, torch2.5.1 and transformers4.48.1; the bundled loader removes dependence on a private local ESM checkout. That backend still requires validation against selected final weights. Install the CUDA build suitable for your driver and disclose it in the run manifest; CPU install is sufficient for scoring.

## Inputs

- Full Stage 1 source: paired Rostlab/ProstT5Dataset export, 474-record test with fingerprint `63341c1e7b16e3e5`. The historic small/full test exports were checked to contain identical rows. Files and dataset fingerprints are separate concepts; preserve both when publishing data.
- PDB: `pdb_v2_structure_510`, test500, paired `sequence_x`/`sequence_y`. The exact processed split is project-specific. Full reproduction requires distributing a licensed snapshot or its preparation recipe and verified hash manifest; a random PDB sample is not interchangeable.
- SCOPe: `pdbstyle-sel-gs-bib-40-2.08.tgz` and `dir.cla.scope.2.08-stable.txt`. Obtain sources from SCOPe or an authorized mirror, preserve their SHA-256, and pass them explicitly. Nothing is silently downloaded by this package.

You may reuse a validated SCOPe directory containing:

```text
GT_fasta/DB_aa.fasta
GT_fasta/DB_di.fasta
labels/scop_lookup.tsv
DB/foldseek_DB/DB*   # includes coordinate/header/3Di sidecars
DB/mmseqs_DB/DB*
```

Or rebuild it without the old project:

```bash
export FOLDSEEK_BIN=/path/to/foldseek
export MMSEQS_BIN=/path/to/mmseqs
aa3di-benchmark --config configs/paper.yaml --output runs/build \
  prepare-scope --archive /path/to/pdbstyle-sel-gs-bib-40-2.08.tgz \
  --classification /path/to/dir.cla.scope.2.08-stable.txt
export AA3DI_SCOPE_ROOT=/path/to/runs/build/scope208
```

Preparation selects classified, atom-containing single-model `.ent` structures, creates the structure database, extracts AA/3Di FASTA and maps Foldseek IDs to SCOPe classification. Expected catalog counts are checked; scoring checks expected eligible queries. A different count is an error requiring investigation, not a reason to silently change the paper protocol.

## Final model selection

Set checkpoint paths in environment variables. The selection JSON must contain:

```json
{
  "stage1_training_records": 17070828,
  "stage2_dataset": "pdb_v2_structure",
  "selection_rule": "Describe the actual validation-based choice",
  "test_used_for_selection": false,
  "sha256": {
    "base_checkpoint": "ACTUAL_SHA256",
    "aa2di_adapter": "ACTUAL_SHA256",
    "di2aa_adapter": "ACTUAL_SHA256"
  }
}
```

This example is a schema, not a declaration that test sets were unused. Supply the real selection history, epochs, learning rates and split identities. Old30% checkpoints must not be relabeled as full17M. PyTorch checkpoint loading assumes trusted files; this repository does not distribute checkpoints or embed credentials.

## Provenance and publication boundary

Each task saves configuration/code hashes, input/weight hashes, dependency versions and completion/failure status. A changed input or configuration requires a new run directory. Publish the prepared data manifests, raw predictions and query-level scores alongside final tables when licensing permits. Model licenses and dataset redistribution permissions apply separately from this repository's Apache license.

The repository can reproduce scoring from supplied artifacts today. Full paper reproduction additionally needs the exact processed test snapshots, frozen final weights and actual GPU validation. Those artifacts are not fabricated or replaced by the synthetic smoke test.

## Environment lock and execution order

`requirements/core-py310.lock` records the complete tested core environment. Install it before installing this package with `--no-deps` to reproduce that CPU environment exactly. Optional model environments have separately pinned direct dependencies and record their complete installed package versions in each task environment. Their full transitive environments have not been rebuilt in this cleanup; do not describe them as fully validated lockfiles.

Run commands sequentially when sharing a run directory. Use separate output directories for independent concurrent jobs; the manifest is not a multi-writer database. The source repository and its working tree must not be edited while a scientific run is active.
