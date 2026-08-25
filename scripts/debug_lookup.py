#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_lookup.py
===============
Ferramenta de consulta para debug: busca processos licitatorios por ID ou texto.

Uso:
    # Buscar por ID especifico
    python scripts/debug_lookup.py --id 243371

    # Buscar multiplos IDs
    python scripts/debug_lookup.py --id 243371 243088 243370 243087

    # Buscar por texto no objeto
    python scripts/debug_lookup.py --busca "enxoval residencial terapeutico"

    # Buscar por texto com limite de resultados
    python scripts/debug_lookup.py --busca "alimentos pereciveis" --top 10

    # Ver candidatos de uma noticia especifica do cache
    python scripts/debug_lookup.py --noticia 0

    # Ver candidatos de uma noticia pelo indice_par
    python scripts/debug_lookup.py --noticia 42
"""

import pickle
import json
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"


def carregar_metadata():
    print(f"Carregando metadata ({META_PATH.stat().st_size / 1e6:.1f} MB)...")
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def carregar_cache():
    print(f"Carregando cache TF-IDF...")
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def buscar_por_id(metadata: list[dict], ids: list[str]):
    ids_set = set(str(i) for i in ids)
    encontrados = {}

    for meta in metadata:
        pid = str(meta.get("id_processo_licitatorio", ""))
        if pid in ids_set:
            encontrados[pid] = meta

    for pid in ids:
        pid = str(pid)
        if pid not in encontrados:
            print(f"\n[NAO ENCONTRADO] ID={pid}")
            continue

        meta = encontrados[pid]
        print(f"\n{'='*80}")
        print(f"ID:          {meta.get('id_processo_licitatorio', 'N/A')}")
        print(f"Municipio:   {meta.get('municipio', 'N/A')}")
        print(f"Modalidade:  {meta.get('modalidade', 'N/A')}")
        print(f"UF:          {meta.get('uf', 'N/A')}")
        print(f"Ente:        {meta.get('ente', 'N/A')}")
        print(f"Edital:      {meta.get('numero_edital', 'N/A')}")
        print(f"\nOBJETO COMPLETO:")
        print(f"  {meta.get('descricao_objeto', 'N/A')}")
        print(f"{'='*80}")


def buscar_por_texto(metadata: list[dict], query: str, top: int = 20):
    query_lower = query.lower()
    resultados = []

    for meta in metadata:
        objeto = meta.get("descricao_objeto", "") or ""
        if query_lower in objeto.lower():
            resultados.append(meta)

    print(f"\n{len(resultados)} resultados para '{query}' (mostrando top {top})\n")

    for meta in resultados[:top]:
        print(f"{'='*80}")
        print(f"ID:         {meta.get('id_processo_licitatorio', 'N/A')}")
        print(f"Municipio:  {meta.get('municipio', 'N/A')}")
        print(f"Modalidade: {meta.get('modalidade', 'N/A')}")
        print(f"\nOBJETO COMPLETO:")
        print(f"  {meta.get('descricao_objeto', 'N/A')}")
        print(f"{'='*80}\n")

    if len(resultados) > top:
        print(f"... e mais {len(resultados) - top} resultados. Use --top {len(resultados)} para ver todos.")


def buscar_noticia_cache(cache: list[dict], metadata: list[dict], noticia_idx: int):
    if noticia_idx >= len(cache):
        print(f"[ERRO] Indice {noticia_idx} fora do range. Cache tem {len(cache)} noticias (0 a {len(cache)-1}).")
        return

    entry = cache[noticia_idx]
    candidatos = entry.get("candidatos_top50", [])
    id_gold = entry.get("id_gold", "N/A")

    print(f"\n{'#'*80}")
    print(f"NOTICIA [{noticia_idx}]")
    print(f"  indice_par:  {entry.get('indice_par', 'N/A')}")
    print(f"  id_gold:     {id_gold}")
    print(f"  municipio:   {entry.get('municipio', 'N/A')}")
    print(f"  modalidade:  {entry.get('modalidade', 'N/A')}")
    print(f"  objeto:      {entry.get('objeto', 'N/A')}")
    print(f"  candidatos:  {len(candidatos)}")
    print(f"{'#'*80}")

    # Buscar objeto completo do gold no metadata
    meta_index = {str(m.get("id_processo_licitatorio", "")): m for m in metadata}
    gold_meta = meta_index.get(str(id_gold))
    if gold_meta:
        print(f"\nOBJETO GOLD COMPLETO (ID={id_gold}):")
        print(f"  {gold_meta.get('descricao_objeto', 'N/A')}")

    print(f"\nCANDIDATOS TOP-{len(candidatos)} (objeto completo):\n")

    for rank, cand in enumerate(candidatos, 1):
        pid = str(cand.get("id_processo", "N/A"))
        objeto = cand.get("objeto", "N/A")
        score = cand.get("score", 0)
        is_gold = pid == str(id_gold)
        marker = "  <-- GOLD" if is_gold else ""

        print(f"  Rank {rank:2d} | Score={score:.4f} | ID={pid}{marker}")
        print(f"         {objeto}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Debug lookup para processos licitatorios")
    parser.add_argument("--id", nargs="+", help="Buscar por ID(s) do processo")
    parser.add_argument("--busca", type=str, help="Buscar por texto no objeto")
    parser.add_argument("--top", type=int, default=20, help="Limite de resultados (default: 20)")
    parser.add_argument("--noticia", type=int, help="Ver candidatos completos de uma noticia do cache pelo indice")
    args = parser.parse_args()

    if not any([args.id, args.busca, args.noticia is not None]):
        parser.print_help()
        sys.exit(0)

    if args.id:
        metadata = carregar_metadata()
        buscar_por_id(metadata, args.id)

    elif args.busca:
        metadata = carregar_metadata()
        buscar_por_texto(metadata, args.busca, top=args.top)

    elif args.noticia is not None:
        metadata = carregar_metadata()
        cache = carregar_cache()
        buscar_noticia_cache(cache, metadata, args.noticia)


if __name__ == "__main__":
    main()
