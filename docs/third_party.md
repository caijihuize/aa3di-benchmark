# Attribution and licensing boundary

The evaluation and AA3Di adaptation code was migrated from the author's experimental workspace. This repository is licensed under Apache-2.0 at the user's explicit choice. The clean package does not vendor EvolutionaryScale/Biohub ESM, Hugging Face Transformers, Foldseek, MMseqs2 or their weights.

Dependencies and model/data artifacts retain their own licenses. Review each upstream repository and model card before downloading, using or redistributing their artifacts:

- ProstT5: https://github.com/mheinzinger/ProstT5 and https://huggingface.co/Rostlab/ProstT5
- ESM: https://github.com/evolutionaryscale/esm
- Foldseek: https://github.com/steineggerlab/foldseek
- MMseqs2: https://github.com/soedinglab/MMseqs2
- SCOPe: https://scop.berkeley.edu/

The SCOPe evaluator preserves the first-wrong-fold logic and unusual catalog denominators of Foldseek's `bench.noselfhit.awk`. The copied upstream AWK fixture is not included in this release tree; tests use author-generated synthetic fixtures and independently specified expected scores. The legacy fixture comparison remains part of migration validation.

There are no pretrained weights, complete datasets, tokens, personal cluster paths or old job logs in the tracked release files. Licensed scientific snapshots and final checkpoints should be published separately with identities and access instructions.
