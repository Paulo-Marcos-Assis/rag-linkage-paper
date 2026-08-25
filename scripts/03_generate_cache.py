#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_generate_cache.py
====================
Gera cache de candidatos TF-IDF top-50 para reutilização em testes de Tournament.

Executa UMA VEZ a recuperação TF-IDF e salva os candidatos top-50 de cada notícia.
Depois, scripts de Tournament podem carregar esse cache ao invés de recalcular.

Economia de tempo: ~90% (só precisa rodar Tournament, não TF-IDF)

Uso:
    python scripts/03_generate_cache.py
"""

import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))
from utils import tokenize, tfidf_cosine_similarity, filtrar_por_municipio_e_modalidade

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data/dataset_500_noticias.json"
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
IDF_PATH = BASE_DIR / "models/tfidf_model/idf_dict.pkl"
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"


def recuperacao_tfidf(objeto_query: str, indices_candidatos: list[int], 
                      metadata: list[dict], idf_dict: dict, k: int = 50) -> list[dict]:
    """Recuperação TF-IDF top-k."""
    scores = []
    
    for idx in indices_candidatos:
        descricao = metadata[idx].get("descricao_objeto", "")
        id_processo = metadata[idx].get("id_processo_licitatorio", "")
        
        if not descricao:
            continue
        
        score = tfidf_cosine_similarity(objeto_query, descricao, idf_dict)
        
        scores.append({
            "idx": idx,
            "score": score,
            "id_processo": id_processo,
            "objeto": descricao,
            "municipio": metadata[idx].get("municipio", ""),
            "modalidade": metadata[idx].get("modalidade", ""),
        })
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]


def main():
    print("=" * 80)
    print(" GERANDO CACHE TF-IDF TOP-50 (500 NOTÍCIAS)")
    print("=" * 80)
    print()
    
    # Carregar dados
    print("Carregando dados...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    noticias = data["noticias"]
    
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    with open(IDF_PATH, "rb") as f:
        idf_dict = pickle.load(f)
    
    print(f"   {len(noticias)} notícias")
    print(f"   {len(metadata):,} processos")
    print(f"   IDF carregado ({len(idf_dict):,} termos)")
    print()
    
    # Processar cada notícia
    print("Processando TF-IDF para cada notícia...")
    cache = []
    
    for noticia in tqdm(noticias, desc="TF-IDF", unit="notícia"):
        indice_par = noticia.get("indice_par")
        id_gold = str(noticia.get("id_processo_gold", ""))
        municipio = noticia.get("municipio_extraido", "")
        modalidade = noticia.get("modalidade_extraida", "")
        objeto = noticia.get("objeto_extraido", "")
        
        # Filtro SQL
        indices_sql = filtrar_por_municipio_e_modalidade(metadata, municipio, modalidade)
        
        if not indices_sql:
            cache.append({
                "indice_par": indice_par,
                "id_gold": id_gold,
                "candidatos_top50": [],
                "sem_candidatos_sql": True,
            })
            continue
        
        # TF-IDF top-50
        candidatos = recuperacao_tfidf(objeto, indices_sql, metadata, idf_dict, k=50)
        
        cache.append({
            "indice_par": indice_par,
            "id_gold": id_gold,
            "municipio": municipio,
            "modalidade": modalidade,
            "objeto": objeto,
            "candidatos_top50": candidatos,
            "sem_candidatos_sql": False,
            "n_candidatos_sql": len(indices_sql),
        })
    
    # Salvar cache
    print()
    print("Salvando cache...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    
    # Estatísticas
    tamanho_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    com_candidatos = sum(1 for c in cache if not c.get("sem_candidatos_sql", False))
    
    print(f"   Cache salvo em: {CACHE_PATH}")
    print(f"   Tamanho: {tamanho_mb:.1f} MB")
    print(f"   Notícias com candidatos: {com_candidatos}/{len(cache)}")
    print()
    
    print("=" * 80)
    print(" CACHE GERADO COM SUCESSO!")
    print("=" * 80)
    print()
    print("Agora você pode usar este cache em scripts de Tournament:")
    print("  - Carrega cache ao invés de recalcular TF-IDF")
    print("  - Economia de tempo: ~90%")
    print()


if __name__ == "__main__":
    main()
