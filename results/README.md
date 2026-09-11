# Results Directory

This directory stores the canonical evaluation of the linkage pipeline (identical-aware,
denominator = total de notícias). Os artefatos brutos (JSONs de execução, caches,
checkpoints, logs) **não são versionados** — são regenerados pelos pipelines e, por
questão ética, o benchmark sintético (714 pares) não é publicado.

## Avaliação canônica (dissertação)

| `arquivo` resultado | Experimento | Acurácia (sobre o total) |
|---|---|---|
| `single_call_top10_n500_gpt-oss_20b.json` | Dev single-call 20B | **76.8%** (384/500) |
| `single_call_top10_n500_qwen2.5_7b.json` | Dev single-call 7B | 65.2% |
| `single_call_top10_n500_qwen3.5_9b.json` | Dev single-call 9B | 76.6% |
| `single_call_top10_n500_qwen3.8_27b.json` | Dev single-call 27B | **77.2%** |
| `single_call_top10_n500_gemma4_31b.json` | Dev single-call 31B | 76.0% |
| `tournament_top10_n500_qwen2.5_7b.json` | Dev tournament 7B | 66.2% |
| `tournament_top10_n500_gpt-oss_20b.json` | Dev tournament 20B | 65.0% |
| `tournament_top10_n500_qwen3.8_27b.json` | Dev tournament 27B | 75.8% |
| `tournament_top10_n500_gemma4_31b.json` | Dev tournament 31B | 60.4% |
| `tournament_n500_gpt-oss_20b.json` | Dev tournament 20B (n=50) | colapsou (falha 41,4%) |
| `single_call_top10_n214_test_gpt-oss_20b.json` | Test single-call 20B | 76.2% (163/214) |
| `single_call_top10_n214_test_qwen3.5_9b.json` | Test single-call 9B | 75.7% |
| `single_call_top10_n214_test_qwen3.8_27b.json` | Test single-call 27B | 78.0% |
| `single_call_top10_n214_test_gemma4_31b.json` | Test single-call 31B | **78.5%** (melhor no teste) |

Recuperação TF-IDF (baseline, strict): dev R@50 96.0% (480/500), R@10 91.0%,
R@1 60.2% (69.4% identical-aware); test R@50 97.2% (208/214), R@10 94.4%,
R@1 65.0% (70.6% identical-aware). Em 12 das 214 notícias o ouro não está no
top-10 (limite estrutural da etapa de reclassificação).

Tabelas completas (dev/test, strict e identical-aware) em `avaliacao_canonica.csv`
e `avaliacao_canonica.json`, geradas por `scripts/99_avaliacao_canonica.py`.

## Regenerando resultados

```bash
# Full pipeline (dev)
make dev

# Full pipeline (test)
make test

# Reranking only
make rerank

# Avaliação canônica dos resultados gerados
python scripts/99_avaliacao_canonica.py
```
(requer os artefatos gerados: caches TF-IDF, mapas de duplicatas/objetos idênticos e
os JSONs de resultado — ver `Makefile` e `benchmarks/README.md`).

## Latency note

The paper reports latencies measured on specific hardware:
- **AMD EPYC 9654** (1.5 TB RAM)
- **NVIDIA RTX A6000** (48 GB VRAM, CUDA 12.8)

Latencies on different hardware will vary. The 20B model fits entirely in VRAM on A6000; CPU offload will significantly increase latency.