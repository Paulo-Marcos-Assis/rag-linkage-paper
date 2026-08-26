# Benchmarks Directory

This directory contains the **regeneration logic** for the synthetic benchmark.
**The 714 generated news-process pairs are NOT stored in this repository** (see Ethics section below).

## Why the dataset is not public

As stated in the paper (Section 3):

> *"Since these fictional articles attribute fraudulent conduct to real procurement records from real municipalities, they are not publicly released; the generation prompts, model configuration and sampling criteria documented here allow the benchmark to be regenerated from the public database."*

Each synthetic article:
- Is fictional (LLM-generated)
- Attributes potentially fraudulent conduct to a **real** procurement process
- Links to a **real** Brazilian municipality
- Publishing these could spread fabricated corruption narratives about identifiable public entities

## Reproducibility without dataset release

Full reproducibility is achieved by providing:

| Component | Location | Purpose |
|---|---|---|
| Generation prompts | `prompts/generation_prompt.txt` | Exact prompt used (few-shot, rules, format) |
| Generation config | `configs/generation_config.yaml` | Model, temperature, seed, host, retries |
| Extraction prompts | `prompts/extraction_prompt.txt` | Attribute extraction prompt |
| Rerank prompts | `prompts/rerank_*.txt` | Single-call & tournament prompts |
| Retrieval config | `configs/retrieval_config.yaml` | All 13 methods parameters |
| Generation scripts | `noticias_simuladas/gerar_noticias_sinteticas*.py` | Complete generation pipeline |
| Regeneration wrapper | `benchmarks/regenerate_benchmark.py` | One-command regeneration |

## Regenerating the benchmark

### Prerequisites
1. **Ollama running** with `gpt-oss:20b` model:
   ```bash
   ollama pull gpt-oss:20b
   ollama serve
   ```
2. **Input data** in `data/` (see `data/README.md`):
   - 500 real fraud news + 500 sampled processes (dev)
   - 214 real fraud news + 214 sampled processes (test)
   - Public procurement database (315k processes)

### Commands

```bash
# Regenerate development set (500 pairs)
python benchmarks/regenerate_benchmark.py --split dev

# Regenerate test set (214 pairs)
python benchmarks/regenerate_benchmark.py --split test

# Regenerate both (714 pairs)
python benchmarks/regenerate_benchmark.py --split both

# Quick test with 5 pairs
python benchmarks/regenerate_benchmark.py --split dev --limit 5
```

### Outputs (git-ignored)
Generated in `noticias_simuladas/` and `data/`:
- `500_noticias_sinteticas.csv` / `214_noticias_sinteticas_test.csv`
- `mapeamento_pares.json` / `mapeamento_pares_test.json`
- `dataset_500_noticias.json` / `dataset_214_noticias_test.json` (after extraction)

### Determinism guarantee
With `seed=42`, `temperature=0.3`, and fixed model `gpt-oss:20b`, regeneration produces **bitwise-identical** synthetic articles to those used in the paper.

## Expected results (from paper)

| Metric | Dev (500) | Test (214) |
|---|---|---|
| TF-IDF R@10 | 91.0% | 94.4% |
| Single-call 20B Accuracy | 76.8% | 71.5% |
| Tournament 7B Accuracy | 70.2% | — |
| Latency (single-call 20B) | 20.8 s/item | — |

## Citation

If you use this benchmark or regeneration code, please cite:

```bibtex
@inproceedings{assis2026hybridrag,
  title={A Hybrid RAG Pipeline for Linking Fraud News to Public Procurement Records},
  author={Assis, Paulo Marcos de and Castro, M{\'a}rcio and Carvalho, J{\^o}nata Tyska},
  booktitle={Anais do 23º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)},
  year={2026}
}
```