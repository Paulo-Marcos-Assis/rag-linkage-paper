#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_generate_cache_test.py
==========================
Gera cache de candidatos TF-IDF top-50 para o TEST SET (214 notícias).

Idêntico ao 03_generate_cache.py, mas processa o dataset de teste.
Usa o mesmo modelo TF-IDF treinado no conjunto completo de processos.

Uso:
    python scripts/03_generate_cache_test.py
"""

import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))
from utils import tokenize, tfidf_cosine_similarity, filtrar_por_municipio_e_modalidade

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data/test/dataset_214_noticias_test.json"
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
IDF_PATH = BASE_DIR / "models/tfidf_model/idf_dict.pkl"
CACHE_PATH = BASE_DIR / "models/test/cache_tfidf_top50_test.pkl"


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
    print(" GERANDO CACHE TF-IDF TOP-50 (TEST SET - 214 NOTÍCIAS)")
    print("=" * 80)
    print()
    
    # Verificar se dataset existe
    if not DATASET_PATH.exists():
        print(f"ERRO: Dataset não encontrado em {DATASET_PATH}")
        print()
        print("Execute primeiro:")
        print("  python scripts/convert_test_csv_to_json.py")
        print()
        sys.exit(1)
    
    # Carregar dados
    print("Carregando dados...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    noticias = data["noticias"]
    
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    with open(IDF_PATH, "rb") as f:
        idf_dict = pickle.load(f)
    
    print(f"   {len(noticias)} notícias (TEST SET)")
    print(f"   {len(metadata):,} processos (base completa)")
    print(f"   IDF carregado ({len(idf_dict):,} termos)")
    print()
    
    # Criar diretório de saída
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Processar cada notícia
    print("Processando TF-IDF para cada notícia do TEST SET...")
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
    print("Salvando cache TEST SET...")
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
    print(" CACHE TEST SET GERADO COM SUCESSO!")
    print("=" * 80)
    print()
    print("Agora você pode executar o pipeline no TEST SET:")
    print("  python scripts/06_single_call_pipeline.py \\")
    print("    --cache models/test/cache_tfidf_top50_test.pkl \\")
    print("    --top 10")
    print()


if __name__ == "__main__":
    main()
