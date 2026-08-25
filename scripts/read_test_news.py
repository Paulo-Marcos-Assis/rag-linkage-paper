#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_test_news.py
=================
Lê e exibe uma notícia específica do dataset de teste (214 notícias).

Uso:
    python scripts/read_test_news.py 62
    python scripts/read_test_news.py 62 --full
"""

import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data/test/dataset_214_noticias_test.json"


def main():
    parser = argparse.ArgumentParser(description="Lê notícia do test set por índice")
    parser.add_argument("indice", type=int, help="Índice da notícia (0-213)")
    parser.add_argument("--full", action="store_true", help="Mostra texto completo da notícia")
    args = parser.parse_args()
    
    # Carregar dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    noticias = data["noticias"]
    
    # Validar índice
    if args.indice < 0 or args.indice >= len(noticias):
        print(f"ERRO: Índice deve estar entre 0 e {len(noticias)-1}")
        return
    
    # Buscar notícia
    noticia = None
    for n in noticias:
        if int(n["indice_par"]) == args.indice:
            noticia = n
            break
    
    if not noticia:
        print(f"ERRO: Notícia com índice {args.indice} não encontrada")
        return
    
    # Exibir
    print("=" * 80)
    print(f" NOTÍCIA TEST SET #{args.indice + 1}/214")
    print("=" * 80)
    print()
    
    print("IDENTIFICAÇÃO:")
    print(f"  Índice par:              {noticia['indice_par']}")
    print(f"  ID processo gold:        {noticia['id_processo_gold']}")
    print()
    
    print("ATRIBUTOS GOLD (injetados):")
    print(f"  Município:               {noticia['municipio_gold']}")
    print(f"  Modalidade:              {noticia['modalidade_gold']}")
    print(f"  Objeto (gold):           {noticia['objeto_gold'][:100]}...")
    print()
    
    print("ATRIBUTOS EXTRAÍDOS (para busca):")
    print(f"  Município extraído:      {noticia['municipio_extraido']}")
    print(f"  Modalidade extraída:     {noticia['modalidade_extraida']}")
    print(f"  Objeto extraído:         {noticia['objeto_extraido']}")
    print()
    
    if 'titulo' in noticia:
        print("TÍTULO DA NOTÍCIA SINTÉTICA:")
        print(f"  {noticia['titulo']}")
        print()
    
    print("TEXTO DA NOTÍCIA SINTÉTICA:")
    texto = noticia.get('texto_completo', '')
    
    if args.full:
        # Texto completo
        print(texto)
    else:
        # Primeiros 500 caracteres
        if len(texto) > 500:
            print(texto[:500] + "...")
            print()
            print(f"[Texto truncado. Use --full para ver completo ({len(texto)} caracteres)]")
        else:
            print(texto)
    
    print()
    print("=" * 80)
    
    # Análise do objeto
    objeto_extraido = noticia['objeto_extraido']
    objeto_gold = noticia['objeto_gold']
    
    print(" ANÁLISE DO OBJETO")
    print("=" * 80)
    print()
    print("OBJETO EXTRAÍDO (usado na busca TF-IDF):")
    print(f"  {objeto_extraido}")
    print(f"  Tamanho: {len(objeto_extraido)} caracteres")
    print(f"  Maiúsculas: {'SIM' if objeto_extraido.isupper() else 'NÃO'}")
    print()
    
    print("OBJETO GOLD (do processo licitatório):")
    print(f"  {objeto_gold}")
    print(f"  Tamanho: {len(objeto_gold)} caracteres")
    print()
    
    # Verificar se são idênticos
    if objeto_extraido.lower() == objeto_gold.lower():
        print("⚠️  ALERTA: Objeto extraído é IDÊNTICO ao objeto gold (case-insensitive)")
        print("   Isso explica por que o TF-IDF tem score perfeito!")
    elif objeto_extraido.upper() == objeto_gold.upper():
        print("⚠️  ALERTA: Objeto extraído é IDÊNTICO ao objeto gold (apenas diferença de case)")
        print("   Isso explica por que o TF-IDF tem score perfeito!")
    else:
        print("✓ Objeto extraído é diferente do gold (paráfrase)")
    
    print()


if __name__ == "__main__":
    main()
