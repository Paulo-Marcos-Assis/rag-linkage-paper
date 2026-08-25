# Hybrid RAG Pipeline for Linking Fraud News to Public Procurement Records

**Paper**: *A Hybrid RAG Pipeline for Linking Fraud News to Public Procurement Records*  
**Authors**: Paulo Marcos de Assis, Márcio Castro, Jônata Tyska Carvalho  
**Venue**: ENIAC 2026 — 23º Encontro Nacional de Inteligência Artificial e Computacional  

## Overview

This repository contains the complete code, prompts, and configurations to reproduce the experiments from our paper. We present a hybrid Retrieval-Augmented Generation (RAG) pipeline that links Brazilian fraud-related news articles to official procurement records using:

1. **LLM-based attribute extraction** (municipality, modality, object description)
2. **SQL pre-filtering** (exact match on municipality + modality)
3. **Classical TF-IDF retrieval** (outperforms neural encoders by 57+ pp in top-50)
4. **LLM reranking** (single-call gpt-oss:20b beats tournament qwen2.5:7b)

## Key Results

| Component | Dev (500) | Test (214) |
|---|---|---|
| TF-IDF R@10 | 91.0% | 94.4% |
| Single-call 20B Accuracy | **76.8%** | **71.5%** |
| Tournament 7B Accuracy | 70.2% | — |
| TF-IDF vs Neural (R@50) | +57.3 pp | — |

## Repository Structure

```
rag-linkage-paper/
├── scripts/                    # Pipeline scripts (01-09, feature_extractor, utils)
├── prompts/                    # Exact prompts used in paper (4 files)
├── configs/                    # YAML configs for all stages (5 files)
├── benchmarks/                 # Regeneration wrapper + ethics doc
├── data/                       # Input data (git-ignored large files)
├── models/                     # TF-IDF/FAISS indexes (git-ignored)
├── results/                    # Execution outputs (git-ignored)
├── noticias_simuladas/         # Generation scripts + outputs (git-ignored)
├── paper/                      # LaTeX paper, reviewer notes, tracking
├── documentacao/               # Internal documentation (Portuguese)
├── Makefile                    # Orchestration targets
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama with models: `gpt-oss:20b`, `qwen2.5:7b`
- 48 GB VRAM GPU recommended (RTX A6000 used in paper)

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull gpt-oss:20b
ollama pull qwen2.5:7b
ollama serve
```

### Data Setup
See `data/README.md` for detailed instructions. You need:
1. **Public procurement database** (315k processes from PNCP/e-Sfinge SC)
2. **Real fraud news articles** (500 dev + 214 test) — collect from NSC Total, ND Mais
3. **Sampled processes** paired with news for synthetic generation

### Reproduce Experiments

```bash
# Full development pipeline (generates benchmark, builds indexes, runs all experiments)
make dev

# Full test pipeline
make test

# Individual stages
make indexes          # Build TF-IDF + FAISS
make eval-retrieval   # 13-method retrieval comparison
make rerank           # Tournament + single-call reranking
make bench-both       # Regenerate 714 synthetic pairs
```

## Reproducibility Without Dataset Release

**The 714 synthetic news articles are NOT in this repository** for ethical reasons (fictional fraud narratives linked to real municipalities). 

Full reproducibility is achieved via:
- **Exact prompts** in `prompts/` (generation, extraction, reranking)
- **Fixed configs** in `configs/` (models, seeds, temperatures, hyperparameters)
- **Generation scripts** in `noticias_simuladas/` and `benchmarks/regenerate_benchmark.py`
- **Fixed seed (42)** guarantees bitwise-identical regeneration

See `benchmarks/README.md` for details.

## Citation

```bibtex
@inproceedings{assis2026hybridrag,
  title={A Hybrid RAG Pipeline for Linking Fraud News to Public Procurement Records},
  author={Assis, Paulo Marcos de and Castro, M\'{a}rcio and Carvalho, J\^{o}nata Tyska},
  booktitle={Anais do 23º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)},
  year={2026}
}
```

## Contact

Paulo Marcos de Assis — paulo.marcos@grad.ufsc.br  
Federal University of Santa Catarina (UFSC)

> **Note**: For reproduction questions, open a GitHub issue.