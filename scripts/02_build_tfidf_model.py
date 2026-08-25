#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_tfidf_model.py
=======================
Calcula e salva IDF para 315k documentos para uso em TF-IDF.

Saída:
    - models/tfidf_model/idf_dict.pkl: Dicionário {term: idf_score}
    - models/tfidf_model/corpus_stats.json: Estatísticas do corpus

Uso:
    python scripts/02_build_tfidf_model.py
"""

import pickle
import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).parent.parent
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
OUTPUT_DIR = BASE_DIR / "models/tfidf_model"


def tokenize(text: str) -> list:
    """Tokenização simples."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_idf_for_corpus(metadata: list) -> tuple[dict, dict]:
    """Calcula IDF para corpus completo."""
    print("\n" + "=" * 80)
    print(" CALCULANDO IDF PARA CORPUS COMPLETO")
    print("=" * 80)
    
    n_docs = len(metadata)
    print(f"\nTotal de documentos: {n_docs:,}")
    
    # Passo 1: Tokenizar
    print("\n[1/3] Tokenizando documentos...")
    corpus_tokens = []
    
    for meta in tqdm(metadata, desc="  Tokenizando", unit="doc"):
        descricao = meta.get("descricao_objeto", "")
        tokens = tokenize(descricao)
        corpus_tokens.append(tokens)
    
    print(f"  {len(corpus_tokens):,} documentos tokenizados")
    
    # Passo 2: Calcular frequência de documentos
    print("\n[2/3] Calculando frequência de documentos...")
    df = {}
    
    for doc_tokens in tqdm(corpus_tokens, desc="  Processando", unit="doc"):
        unique_tokens = set(doc_tokens)
        for term in unique_tokens:
            df[term] = df.get(term, 0) + 1
    
    n_unique_terms = len(df)
    print(f"  {n_unique_terms:,} termos únicos encontrados")
    
    # Passo 3: Calcular IDF
    print("\n[3/3] Calculando IDF...")
    idf = {}
    
    for term, doc_freq in tqdm(df.items(), desc="  Calculando IDF", unit="termo"):
        idf[term] = np.log((n_docs + 1) / (doc_freq + 1)) + 1
    
    print(f"  IDF calculado para {len(idf):,} termos")
    
    # Estatísticas
    idf_values = list(idf.values())
    stats = {
        "n_documentos": n_docs,
        "n_termos_unicos": n_unique_terms,
        "idf_min": float(min(idf_values)),
        "idf_max": float(max(idf_values)),
        "idf_medio": float(np.mean(idf_values)),
        "idf_mediano": float(np.median(idf_values)),
    }
    
    print("\n" + "=" * 80)
    print(" ESTATÍSTICAS DO CORPUS")
    print("=" * 80)
    print(f"  Documentos:      {stats['n_documentos']:,}")
    print(f"  Termos únicos:   {stats['n_termos_unicos']:,}")
    print(f"  IDF mín/máx:     {stats['idf_min']:.3f} / {stats['idf_max']:.3f}")
    print(f"  IDF médio:       {stats['idf_medio']:.3f}")
    print("=" * 80)
    
    return idf, stats


def main():
    print("=" * 80)
    print(" BUILD TF-IDF MODEL")
    print("=" * 80)
    
    # Carregar metadata
    print(f"\nCarregando metadata: {META_PATH.name}")
    
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"   {len(metadata):,} registros carregados")
    
    # Calcular IDF
    idf_dict, stats = compute_idf_for_corpus(metadata)
    
    # Criar diretório
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Salvar IDF
    idf_path = OUTPUT_DIR / "idf_dict.pkl"
    print(f"\nSalvando IDF: {idf_path.name}")
    with open(idf_path, "wb") as f:
        pickle.dump(idf_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = idf_path.stat().st_size / (1024 * 1024)
    print(f"   Salvo ({file_size_mb:.2f} MB)")
    
    # Salvar estatísticas
    stats_path = OUTPUT_DIR / "corpus_stats.json"
    print(f"\nSalvando estatísticas: {stats_path.name}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   Salvo")
    
    print("\n" + "=" * 80)
    print(" CONCLUÍDO!")
    print("=" * 80)
    print(f"\nModelo TF-IDF salvo em: {OUTPUT_DIR}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
