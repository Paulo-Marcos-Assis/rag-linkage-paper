# Prompts Directory

This directory contains the **exact prompts** used in the paper's experiments, extracted from the pipeline scripts. Each prompt is versioned and documented for full reproducibility.

## Files

| File | Source Script | Purpose | Model | Temperature |
|---|---|---|---|---|
| `generation_prompt.txt` | `noticias_simuladas/gerar_noticias_sinteticas.py` (lines 67–178) | Generate synthetic news from real news + process attributes | `gpt-oss:20b` | 0.3 |
| `extraction_prompt.txt` | `scripts/feature_extractor.py` (lines 94–260+) | Extract municipality, modality, edital, objeto from news | `gpt-oss:20b` | 0.0 |
| `rerank_single_call_prompt.txt` | `scripts/06_single_call_pipeline.py` (lines 161–212) | Single-call reranking of top-10 candidates | `gpt-oss:20b` | 0.0 |
| `rerank_tournament_prompt.txt` | `scripts/05_tournament_pipeline.py` (lines 135–228) | Tournament reranking (groups of 5, top-50) | `qwen2.5:7b` | 0.0 |

## Versioning

All prompts correspond to the **exact versions used in the paper experiments** (seed=42, fixed model versions). Any modification to prompts should be versioned (e.g., `generation_prompt_v2.txt`) and documented.

## Usage in Pipeline

The pipeline scripts read these prompt files at runtime (to be implemented in `scripts/00_setup_env.py` or via config). Currently, prompts are embedded as string constants in the Python scripts for execution speed. For reproducibility, these files serve as the **canonical reference**.

## Placeholders

Prompts use `{placeholder}` syntax for runtime interpolation:

| Placeholder | Description |
|---|---|
| `{municipio}` | Municipality name (injected from process) |
| `{modalidade}` | Procurement modality (injected from process) |
| `{descricao_objeto}` | Full object description from process |
| `{titulo_original}` | Original real news title |
| `{texto_noticia_original}` | Original real news full text |
| `{objeto_noticia}` | Extracted object from news (query) |
| `{num_candidates}` | Number of candidates (10 or 50) |
| `{candidatos_text}` | Formatted candidate list |

## Model Versions (Critical for Reproducibility)

| Model | Version/Source | Paper Role |
|---|---|---|
| `gpt-oss:20b` | OpenAI gpt-oss-20b (Aug 2025) | News generation, attribute extraction, single-call reranking |
| `qwen2.5:7b` | Qwen2.5-7B-Instruct | Tournament reranking |

> **Note**: The paper uses `gpt-oss:20b` via Ollama. Ensure the exact model tag is pulled: `ollama pull gpt-oss:20b` and `ollama pull qwen2.5:7b`.