# Legacy result index

Historical artifacts remain outside this clean repository. They are not copied into Git or relabeled as final-model results.

| Existing artifact | Scope | Role |
|---|---|---|
| Bidirectional SCOPe run, completed2026-10-06 | Five models,13,920 records per direction | Developmental predictions, quality, speed and search |
| ProstT5 decoder / Foldseek / MMseqs2 search baselines | SCOPe2.08,5,046 queries | Candidates for provenance-checked reuse |
| PDB Stage2 screen,2026-10-06 | Two forward adapters on30% base | Developmental candidate selection |
| Formal474/500 and five-length inputs | Prepared but not inferred in the final profile | Input preparation only |

Supply legacy predictions to `evaluate` and ordered alignments to `score-homology` for independent rescoring. Public raw-artifact download URLs are not yet assigned; preserve the original artifacts and attach verified, licensed snapshots when publishing scientific results.
