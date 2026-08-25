#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_analyze_duplicates.py
========================
Análise detalhada do impacto de duplicatas nos resultados.

Investiga:
1. Quantas notícias têm gold em descrição duplicada
2. Impacto nas métricas (Precision@1, Match Rate)
3. Distribuição de duplicatas por município/modalidade
4. Gera mapeamento de duplicatas

Uso:
    python scripts/04_analyze_duplicates.py
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "data/dataset_500_noticias.json"
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
OUTPUT_PATH = BASE_DIR / "data/duplicatas_mapping.json"


def main():
    print("=" * 80)
    print(" ANÁLISE DETALHADA DE DUPLICATAS")
    print("=" * 80)
    print()
    
    # Carregar dados
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    noticias = data["noticias"]
    
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Dados carregados:")
    print(f"   Notícias: {len(noticias)}")
    print(f"   Processos: {len(metadata):,}")
    print()
    
    # 1. Mapear descrições duplicadas
    print("1. Mapeando descrições duplicadas...")
    desc_to_ids = defaultdict(list)
    
    for m in metadata:
        desc = m.get("descricao_objeto", "").strip()
        id_proc = m.get("id_processo_licitatorio", "")
        if desc and id_proc:
            desc_to_ids[desc].append(id_proc)
    
    # Filtrar apenas duplicadas
    duplicadas = {desc: ids for desc, ids in desc_to_ids.items() if len(ids) > 1}
    
    total_processos_duplicados = sum(len(ids) for ids in duplicadas.values())
    
    print(f"   Descrições únicas: {len(desc_to_ids) - len(duplicadas):,}")
    print(f"   Descrições duplicadas: {len(duplicadas):,}")
    print(f"   Processos afetados: {total_processos_duplicados:,} ({100*total_processos_duplicados/len(metadata):.1f}%)")
    print()
    
    # 2. Criar mapeamento id -> grupo
    print("2. Criando mapeamento de grupos equivalentes...")
    id_to_group = {}
    
    for ids in duplicadas.values():
        for id_proc in ids:
            id_to_group[str(id_proc)] = [str(i) for i in ids]
    
    print(f"   {len(id_to_group):,} IDs mapeados")
    print()
    
    # 3. Analisar impacto nas notícias
    print("3. Analisando impacto nas 500 notícias...")
    
    noticias_com_gold_duplicado = 0
    
    for noticia in noticias:
        id_gold = str(noticia.get("id_processo_gold", ""))
        
        if id_gold in id_to_group:
            noticias_com_gold_duplicado += 1
    
    print(f"   Notícias com gold em descrição duplicada: {noticias_com_gold_duplicado}/{len(noticias)} ({100*noticias_com_gold_duplicado/len(noticias):.1f}%)")
    print()
    
    # 4. Distribuição
    print("4. Distribuição de duplicatas:")
    dist = defaultdict(int)
    for ids in duplicadas.values():
        dist[len(ids)] += 1
    
    for n, count in sorted(dist.items()):
        print(f"   {count:,} descrições aparecem {n} vezes")
    print()
    
    # 5. Salvar mapeamento
    print("Salvando mapeamento de duplicatas...")
    output_data = {
        "total_processos": len(metadata),
        "processos_duplicados": total_processos_duplicados,
        "descricoes_duplicadas": len(duplicadas),
        "noticias_afetadas": noticias_com_gold_duplicado,
        "id_to_group": id_to_group,
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   Salvo em: {OUTPUT_PATH}")
    print(f"   Tamanho: {OUTPUT_PATH.stat().st_size / (1024*1024):.1f} MB")
    print()
    
    print("=" * 80)
    print(" ANÁLISE CONCLUÍDA!")
    print("=" * 80)
    print()
    print("Use este mapeamento para avaliar resultados considerando duplicatas.")
    print()


if __name__ == "__main__":
    main()
