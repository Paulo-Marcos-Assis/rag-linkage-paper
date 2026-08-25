#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_tfidf_rank.py
=====================
Script simples para explorar os top-k candidatos do rank TF-IDF.

Permite visualizar os candidatos recuperados para qualquer notícia do dataset,
mostrando o objeto da notícia e os top-k candidatos do TF-IDF.

Uso:
    # Buscar pelo número que aparece nos logs (1-500)
    python scripts/explore_tfidf_rank.py --numero 431
    python scripts/explore_tfidf_rank.py -n 431 --k 20
    
    # Buscar pelo ID do processo gold
    python scripts/explore_tfidf_rank.py --id 158312
    python scripts/explore_tfidf_rank.py --id 158312 --k 20
    
    # Buscar pelo índice do par (0-499)
    python scripts/explore_tfidf_rank.py --indice 499
"""

import argparse
import pickle
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Caminhos
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
DATASET_PATH = BASE_DIR / "data/dataset_500_noticias.json"


def carregar_cache_tfidf() -> list[dict]:
    """Carrega cache TF-IDF."""
    if not CACHE_PATH.exists():
        print(f"ERRO: Cache não encontrado em {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def carregar_dataset() -> dict:
    """Carrega dataset de notícias."""
    if not DATASET_PATH.exists():
        print(f"ERRO: Dataset não encontrado em {DATASET_PATH}")
        sys.exit(1)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encontrar_noticia_por_id(cache: list[dict], id_gold: str) -> dict | None:
    """Encontra notícia no cache pelo ID gold."""
    for item in cache:
        if str(item.get("id_gold", "")) == str(id_gold):
            return item
    return None


def encontrar_noticia_por_indice(cache: list[dict], indice: int) -> dict | None:
    """Encontra notícia no cache pelo índice do par."""
    for item in cache:
        if int(item.get("indice_par", -1)) == indice:
            return item
    return None


def encontrar_noticia_por_numero(cache: list[dict], numero: int) -> dict | None:
    """
    Encontra notícia pelo número de processamento (1-500).
    Este é o número que aparece nos logs: "PROCESSANDO NOTICIA 431/500"
    """
    if numero < 1 or numero > len(cache):
        return None
    # O número 1 corresponde ao índice 0, número 2 ao índice 1, etc.
    indice = numero - 1
    return cache[indice]


def exibir_candidatos(item: dict, k: int = 10):
    """Exibe os top-k candidatos do TF-IDF para uma notícia."""
    
    id_gold = item.get("id_gold", "N/A")
    indice_par = item.get("indice_par", "N/A")
    municipio = item.get("municipio", "N/A")
    modalidade = item.get("modalidade", "N/A")
    objeto = item.get("objeto", "")
    candidatos = item.get("candidatos_top50", [])
    
    # Verificar se há candidatos
    if item.get("sem_candidatos_sql", False) or not candidatos:
        print(f"\n{'='*80}")
        print(f"NOTÍCIA {int(indice_par)+1}/500")
        print(f"ID Gold: {id_gold}")
        print(f"Município: {municipio}")
        print(f"Modalidade: {modalidade}")
        print(f"{'='*80}")
        print("\n⚠️  SEM CANDIDATOS SQL - Esta notícia não possui candidatos recuperados.")
        return
    
    # Limitar k ao número de candidatos disponíveis
    k_real = min(k, len(candidatos))
    candidatos_topk = candidatos[:k_real]
    
    # Verificar se o gold está no rank
    rank_gold = None
    for rank, cand in enumerate(candidatos_topk, 1):
        if str(cand.get('id_processo', '')) == str(id_gold):
            rank_gold = rank
            break
    
    # Cabeçalho
    print(f"\n{'='*80}")
    print(f"PROCESSANDO NOTÍCIA {int(indice_par)+1}/500")
    print(f"ID Gold: {id_gold}")
    print(f"Município: {municipio}")
    print(f"Modalidade: {modalidade}")
    
    if rank_gold:
        print(f"Gold no rank TF-IDF: {rank_gold}/{k_real}")
    else:
        print(f"Gold NÃO encontrado no top-{k_real} TF-IDF")
    
    print(f"{'='*80}")
    
    # Objeto da notícia
    print(f"\n{'='*80}")
    print(f"OBJETO DA NOTÍCIA:")
    print(objeto)
    
    # Candidatos
    print(f"\nCANDIDATOS ({k_real}):")
    for i, cand in enumerate(candidatos_topk, 1):
        id_cand = cand.get('id_processo', 'N/A')
        obj_cand = str(cand.get('objeto', '')).strip()
        
        # Truncar objeto se muito longo
        if len(obj_cand) > 100:
            obj_cand_display = obj_cand[:100] + "..."
        else:
            obj_cand_display = obj_cand
        
        # Marcar se é o gold
        marcador = " ✓ [GOLD]" if str(id_cand) == str(id_gold) else ""
        
        print(f"  {i}. ID={id_cand} | {obj_cand_display}{marcador}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Explora os top-k candidatos do rank TF-IDF para uma notícia"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--numero", "-n", type=int, 
                       help="Número da notícia nos logs (1-500). Ex: 'PROCESSANDO NOTICIA 431/500' -> use --numero 431")
    group.add_argument("--id", type=str, 
                       help="ID do processo gold da notícia. Ex: --id 158312")
    group.add_argument("--indice", type=int, 
                       help="Índice do par (0-499). Equivalente a --numero menos 1")
    
    parser.add_argument("--k", type=int, default=10,
                        help="Número de candidatos a exibir (padrão: 10, máx: 50)")
    
    args = parser.parse_args()
    
    # Validar k
    if args.k < 1 or args.k > 50:
        print("ERRO: k deve estar entre 1 e 50")
        sys.exit(1)
    
    # Carregar cache
    print("Carregando cache TF-IDF...")
    cache = carregar_cache_tfidf()
    print(f"{len(cache)} notícias carregadas\n")
    
    # Encontrar notícia
    if args.numero:
        if args.numero < 1 or args.numero > len(cache):
            print(f"ERRO: Número deve estar entre 1 e {len(cache)}")
            sys.exit(1)
        item = encontrar_noticia_por_numero(cache, args.numero)
        if not item:
            print(f"ERRO: Notícia número {args.numero} não encontrada no cache")
            sys.exit(1)
    elif args.id:
        item = encontrar_noticia_por_id(cache, args.id)
        if not item:
            print(f"ERRO: Notícia com ID gold '{args.id}' não encontrada no cache")
            sys.exit(1)
    else:  # args.indice
        if args.indice < 0 or args.indice >= len(cache):
            print(f"ERRO: Índice deve estar entre 0 e {len(cache)-1}")
            sys.exit(1)
        item = encontrar_noticia_por_indice(cache, args.indice)
        if not item:
            print(f"ERRO: Notícia com índice {args.indice} não encontrada no cache")
            sys.exit(1)
    
    # Exibir candidatos
    exibir_candidatos(item, k=args.k)


if __name__ == "__main__":
    main()
