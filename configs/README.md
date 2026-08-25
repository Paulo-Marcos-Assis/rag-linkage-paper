# Configs Directory

Centralized YAML configurations for all pipeline stages. Each config corresponds to a specific experiment in the paper.

## Files

| File | Stage | Key Parameters |
|---|---|---|
| `generation_config.yaml` | Synthetic news generation | `gpt-oss:20b`, temp=0.3, seed=42 |
| `extraction_config.yaml` | Attribute extraction | `gpt-oss:20b`, temp=0.0, seed=42 |
| `rerank_20b_config.yaml` | Single-call reranking | `gpt-oss:20b`, top-10, temp=0.0 |
| `rerank_7b_config.yaml` | Tournament reranking | `qwen2.5:7b`, top-50 (10×5), temp=0.0 |
| `retrieval_config.yaml` | 13-method retrieval eval | TF-IDF, n-grams, distances, neural (zero-shot) |

## Usage

Pipeline scripts should load configs via:
```python
import yaml
with open("configs/generation_config.yaml") as f:
    cfg = yaml.safe_load(f)
```

## Critical Reproducibility Notes

1. **Seed = 42** fixed across all stages for deterministic results
2. **Model versions** must match exactly:
   - `gpt-oss:20b` (OpenAI gpt-oss-20b, Aug 2025)
   - `qwen2.5:7b` (Qwen2.5-7B-Instruct)
3. **Ollama host**: Configure via `OLLAMA_HOST` environment variable (default: `http://localhost:11434`). The paper used an internal UFSC host for generation; set your own in the configs or via env var.
4. **Temperature = 0.0** for all deterministic stages (extraction, reranking). Only generation uses `0.3` for diversity.
5. **TF-IDF IDF weights** computed on full 315,052 process corpus (no leakage from news).

## Hardware Reference

Paper experiments ran on:
- CPU: AMD EPYC 9654 (1.5 TB RAM)
- GPU: NVIDIA RTX A6000 (48 GB VRAM, CUDA 12.8)

The 20B model fits entirely in VRAM. For smaller GPUs, use quantization or CPU offload (will affect latency).