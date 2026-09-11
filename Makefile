# Makefile for RAG Linkage Pipeline
# Paper: A Hybrid RAG Pipeline for Linking Fraud News to Public Procurement Records
#
# Usage:
#   make dev          # Full dev pipeline (generation -> extraction -> retrieval -> rerank)
#   make test         # Full test pipeline
#   make eval-retrieval  # 13-method retrieval evaluation only
#   make rerank       # Reranking experiments only
#   make indexes      # Build TF-IDF + FAISS indexes
#   make clean        # Remove generated files (keeps source code)

.PHONY: help dev test eval-retrieval rerank indexes extract clean

# Default target
help:
	@echo "RAG Linkage Pipeline - Make Targets"
	@echo "=================================="
	@echo "  make dev            - Run full development pipeline (500 pairs)"
	@echo "  make test           - Run full test pipeline (214 pairs)"
	@echo "  make eval-retrieval - Canonical evaluation of generated results (99_avaliacao_canonica.py)"
	@echo "  make rerank         - Run reranking experiments (tournament + single-call)"
	@echo "  make indexes        - Build TF-IDF model and FAISS vector stores"
	@echo "  make extract        - Run attribute extraction on generated news"
	@echo "  make bench-dev      - Regenerate synthetic benchmark (dev)"
	@echo "  make bench-test     - Regenerate synthetic benchmark (test)"
	@echo "  make bench-both     - Regenerate both splits (714 pairs)"
	@echo "  make clean          - Remove generated outputs (results/, models/, benchmarks/)"
	@echo "  make clean-all      - Remove everything including downloaded data"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Ollama running with gpt-oss:20b and qwen2.5:7b"
	@echo "  - data/public_procurement_315k.csv in place"
	@echo "  - Real news + sampled processes in data/"

# ============================================================
# BENCHMARK REGENERATION
# ============================================================

bench-dev:
	python benchmarks/regenerate_benchmark.py --split dev

bench-test:
	python benchmarks/regenerate_benchmark.py --split test

bench-both:
	python benchmarks/regenerate_benchmark.py --split both

bench-quick:
	python benchmarks/regenerate_benchmark.py --split dev --limit 5

# ============================================================
# INDEX BUILDING
# ============================================================

indexes: tfidf-model faiss-index tfidf-cache-dev tfidf-cache-test

tfidf-model:
	python scripts/02_build_tfidf_model.py

faiss-index:
	python scripts/01_build_vector_store.py

tfidf-cache-dev:
	python scripts/03_generate_cache.py

tfidf-cache-test:
	python scripts/03_generate_cache_test.py

# ============================================================
# ATTRIBUTE EXTRACTION
# ============================================================

extract:
	python scripts/extract_objects_from_test_news.py
	# Note: Dev extraction is embedded in generation script output
	# For full re-extraction, run feature_extractor.py on generated news

# ============================================================
# RETRIEVAL EVALUATION (13 methods)
# ============================================================

eval-retrieval:
	python scripts/99_avaliacao_canonica.py
	# Nota: avalia os artefatos já gerados (caches TF-IDF, mapas de equivalência e
	# JSONs de resultado). A comparação de 13 métodos de recuperação roda pelos
	# scripts 01-04 + utils a partir dos artefatos gerados por `make indexes`;
	# ver results/README.md para as tabelas finais.

# ============================================================
# RERANKING EXPERIMENTS
# ============================================================

rerank: rerank-tournament-7b rerank-single-7b rerank-single-20b

rerank-tournament-7b:
	python scripts/05_tournament_pipeline.py --config configs/rerank_7b_config.yaml

rerank-single-7b:
	python scripts/06_single_call_pipeline.py --config configs/rerank_7b_config.yaml --top-k 10

rerank-single-20b:
	python scripts/06_single_call_pipeline.py --config configs/rerank_20b_config.yaml --top-k 10

rerank-tournament-20b:
	python scripts/08_tournament_top10_pipeline.py --config configs/rerank_20b_config.yaml --top-k 10

# ============================================================
# FULL PIPELINES
# ============================================================

dev: bench-dev indexes extract rerank
	@echo "Development pipeline complete. Results in results/ (avaliacao canônica: python scripts/99_avaliacao_canonica.py)"

test: bench-test indexes extract rerank
	@echo "Test pipeline complete. Results in results/ (avaliacao canônica: python scripts/99_avaliacao_canonica.py)"

# ============================================================
# CLEANUP
# ============================================================

clean:
	rm -rf models/ benchmarks/*.json benchmarks/*.csv
	find results -type f ! -name 'README.md' ! -name 'avaliacao_canonica.csv' ! -name 'avaliacao_canonica.json' -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned generated outputs."

clean-all: clean
	rm -rf data/*.csv data/*.json data/test/
	rm -rf noticias_simuladas/*.csv noticias_simuladas/*.json noticias_simuladas/*.log
	@echo "Cleaned ALL generated data (including benchmark inputs)."