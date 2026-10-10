# AA–3Di Benchmark

独立评测AA↔3Di翻译、推理效率和SCOPe 2.08远程同源检索。正式范围为ProstT5主decoder、ESM3-AA3Di同基座Stage 1/Stage 2，以及结构Foldseek和MMseqs2检索基线。训练、CATH/迁移学习和逆折叠回折不属于本仓库。

## 无GPU快速验收

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
aa3di-benchmark --config configs/smoke.yaml --output runs/smoke smoke
```

验收使用仓库自带合成FASTA与检索命中，生成翻译评分、检索评分、Markdown/CSV/JSON报告和PNG/SVG图。它验证代码流程，不是论文实验结果，不需要旧项目、模型权重、GPU或Foldseek二进制。

## 正式对照实验

ProstT5推理建议单独使用Python 3.12环境。安装本包的ProstT5和数据附加依赖；ESM3另用环境安装`.[esm3]`，不要把两套相互冲突的依赖合并安装。

```bash
python -m pip install '.[prostt5,data]'
# 根据.env.example设置本机数据、工具和权重路径；不要将实际.env提交Git。
export PROSTT5_MODEL=/path/to/ProstT5
export AA3DI_STAGE1_DATA=/path/to/full17m_dataset
export AA3DI_PDB_DATA=/path/to/pdb_v2_structure_510

aa3di-benchmark --config configs/paper.yaml --output runs/paper plan
aa3di-benchmark --config configs/paper.yaml --output runs/paper prepare --dataset stage1_test474
aa3di-benchmark --config configs/paper.yaml --output runs/paper prepare --dataset pdb_test500
# 两个测试集分别执行两个方向；这里只展示其中一个命令。
aa3di-benchmark --config configs/paper.yaml --output runs/paper infer \
  --method prostt5_decode_fp16_bs --dataset stage1_test474 --direction aa2di
```

不需要配置ESM3就可以执行上述ProstT5任务。各任务可由不同模型环境依次写入同一运行目录，配置与本包源代码须保持相同。

## 分长度测速与远程同源

```bash
export AA3DI_SCOPE_ROOT=/path/to/prepared_scope208
aa3di-benchmark --config configs/paper.yaml --output runs/paper prepare --dataset scope208
aa3di-benchmark --config configs/paper.yaml --output runs/paper prepare-speed
aa3di-benchmark --config configs/paper.yaml --output runs/paper speed --method prostt5_decode_fp16_bs
# 全量AA→3Di预测进入Foldseek预测库。
aa3di-benchmark --config configs/paper.yaml --output runs/paper infer \
  --method prostt5_decode_fp16_bs --dataset scope208 --direction aa2di
export FOLDSEEK_BIN=/path/to/foldseek
export MMSEQS_BIN=/path/to/mmseqs
aa3di-benchmark --config configs/paper.yaml --output runs/paper homology --method prostt5_decode_fp16_bs
aa3di-benchmark --config configs/paper.yaml --output runs/paper homology --method foldseek
aa3di-benchmark --config configs/paper.yaml --output runs/paper homology --method mmseqs
aa3di-benchmark --config configs/paper.yaml --output runs/paper report
```

基线数据库、输入FASTA和标签的准备方法见[数据与复现说明](docs/reproducibility.md)。GPU任务可使用可选`scripts/slurm/run.sbatch`；分区、账户和环境由使用者提供，无固定HPC路径。

## 重新评分已有结果

```bash
aa3di-benchmark --config configs/paper.yaml --output runs/rescore evaluate \
  --reference /path/to/di.fasta --prediction /path/to/predictions.fasta \
  --direction aa2di --method prostt5_decode_fp16_bs --dataset scope208
aa3di-benchmark --config configs/paper.yaml --output runs/rescore score-homology \
  --labels /path/to/scop_lookup.tsv --hits /path/to/ordered_hits.tsv --method prostt5_decode_fp16_bs
```

评分不加载模型。保留缺失、非法字符、长度不匹配及覆盖率，不修改预测；Macro-F1不是逐序列平均准确率。详细分母和X处理见[实验协议](docs/protocol.md)。

ESM3最终权重须由选择记录及SHA-256明确冻结。p3Di生成再编码计时使用`embedding --method esm3_aa3di_full17m_pdb_stage2 --input /path/to/aa.fasta`；它不等于3Di→AA序列恢复。

每次运行保存配置/代码快照哈希、输入/权重哈希、任务完成或失败状态及依赖版本。改变输入、代码或配置时使用新输出目录。报告汇总现有结果，明确保持部分完成，不自动宣布整篇论文就绪。

## 许可证与结果发布

本仓库代码采用Apache-2.0。模型权重、上游库和数据各自遵循其许可，详见[第三方说明](docs/third_party.md)。仓库不包含权重、完整数据集、旧实验日志或敏感配置。

完整GPU实验与最终ESM3权重尚未在此整理版上验收；核心验收、旧预测重新评分与迁移边界见[迁移验证记录](docs/migration_validation.md)。
