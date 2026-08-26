#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_benchmark.py
=======================
Wrapper script to regenerate the synthetic benchmark (dev + test sets)
exactly as used in the paper.

This script orchestrates the generation of 714 synthetic news-process pairs:
- 500 for development set (70%)
- 214 for held-out test set (30%)

Usage:
    python scripts/regenerate_benchmark.py --split dev
    python scripts/regenerate_benchmark.py --split test
    python scripts/regenerate_benchmark.py --split both

Requirements:
- Ollama running with gpt-oss:20b model
- Input CSVs in data/ (real news + sampled processes)
- Configs in configs/generation_config.yaml

Outputs (git-ignored, written to noticias_simuladas/ and data/):
- noticias_simuladas/500_noticias_sinteticas.csv
- noticias_simuladas/214_noticias_sinteticas_test.csv
- noticias_simuladas/mapeamento_pares.json
- noticias_simuladas/mapeamento_pares_test.json
- data/dataset_500_noticias.json
- data/dataset_214_noticias_test.json
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_cmd(cmd, cwd=None, env=None):
    """Run command and stream output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env or os.environ)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result

def check_prerequisites(split):
    """Verify required inputs exist."""
    required = {
        'dev': [
            'data/noticias_reais_500.csv',
            'data/processos_500_dev.csv',
        ],
        'test': [
            'data/noticias_reais_214_test.csv',
            'data/processos_214_test.csv',
        ],
        'both': [
            'data/noticias_reais_500.csv',
            'data/processos_500_dev.csv',
            'data/noticias_reais_214_test.csv',
            'data/processos_214_test.csv',
        ]
    }
    
    missing = [f for f in required[split] if not (PROJECT_ROOT / f).exists()]
    if missing:
        print(f"ERROR: Missing required input files for {split}:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease place the required CSV files in data/")
        print("See data/README.md for format specifications.")
        sys.exit(1)
    
    # Check Ollama model
    print("Checking Ollama availability...")
    try:
        subprocess.run(['ollama', 'list'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Ollama not found or not running.")
        print("Please start Ollama and pull required models:")
        print("  ollama pull gpt-oss:20b")

def generate_dev():
    """Generate development set (500 pairs)."""
    print("\n" + "="*60)
    print("GENERATING DEVELOPMENT SET (500 pairs)")
    print("="*60)
    
    script = PROJECT_ROOT / 'noticias_simuladas' / 'gerar_noticias_sinteticas.py'
    if not script.exists():
        print(f"ERROR: Generation script not found: {script}")
        sys.exit(1)
    
    run_cmd([sys.executable, str(script)])

def generate_test():
    """Generate test set (214 pairs)."""
    print("\n" + "="*60)
    print("GENERATING TEST SET (214 pairs)")
    print("="*60)
    
    script = PROJECT_ROOT / 'noticias_simuladas' / 'gerar_noticias_sinteticas_test.py'
    if not script.exists():
        print(f"ERROR: Generation script not found: {script}")
        sys.exit(1)
    
    run_cmd([sys.executable, str(script)])

def build_datasets():
    """Build final dataset JSONs from generated CSVs (placeholder for now)."""
    print("\n" + "="*60)
    print("BUILDING FINAL DATASET JSONS")
    print("="*60)
    print("Note: dataset_500_noticias.json and dataset_214_noticias_test.json")
    print("are built by the extraction pipeline (scripts/feature_extractor.py)")
    print("after attribute extraction. Run the full pipeline via Makefile.")
    # TODO: Add actual dataset building logic if separate from extraction

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate synthetic benchmark for RAG linkage paper"
    )
    parser.add_argument(
        '--split', 
        choices=['dev', 'test', 'both'], 
        default='both',
        help='Which split to generate (default: both)'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=None,
        help='Limit to N pairs (for quick testing)'
    )
    parser.add_argument(
        '--start-from', 
        type=int, 
        default=0,
        help='Resume from index (0-indexed)'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC BENCHMARK REGENERATION")
    print("Paper: A Hybrid RAG Pipeline for Linking Fraud News")
    print("       to Public Procurement Records")
    print("="*60)
    print(f"Split: {args.split}")
    print(f"Seed: 42 (fixed in configs/generation_config.yaml)")
    print(f"Model: gpt-oss:20b (temperature 0.3)")
    
    check_prerequisites(args.split)
    
    if args.split in ['dev', 'both']:
        if args.limit:
            # Modify script call for limit
            script = PROJECT_ROOT / 'noticias_simuladas' / 'gerar_noticias_sinteticas.py'
            run_cmd([sys.executable, str(script), '--limit', str(args.limit), '--start-from', str(args.start_from)])
        else:
            generate_dev()
    
    if args.split in ['test', 'both']:
        if args.limit:
            script = PROJECT_ROOT / 'noticias_simuladas' / 'gerar_noticias_sinteticas_test.py'
            run_cmd([sys.executable, str(script), '--limit', str(args.limit), '--start-from', str(args.start_from)])
        else:
            generate_test()
    
    build_datasets()
    
    print("\n" + "="*60)
    print("BENCHMARK REGENERATION COMPLETE")
    print("="*60)
    print("Generated files (git-ignored):")
    if args.split in ['dev', 'both']:
        print("  - noticias_simuladas/500_noticias_sinteticas.csv")
        print("  - noticias_simuladas/mapeamento_pares.json")
    if args.split in ['test', 'both']:
        print("  - noticias_simuladas/214_noticias_sinteticas_test.csv")
        print("  - noticias_simuladas/mapeamento_pares_test.json")
    print("\nNext steps:")
    print("  1. Run attribute extraction: make extract")
    print("  2. Build retrieval indexes: make indexes")
    print("  3. Run retrieval evaluation: make eval-retrieval")
    print("  4. Run reranking: make rerank")

if __name__ == '__main__':
    main()