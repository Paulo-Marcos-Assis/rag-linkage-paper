#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_map_identical_objects.py
============================
Mapeia processos licitatórios com objetos textualmente idênticos.

Problema:
- Dois processos diferentes (ex: PE52/2022 e PE67/2022)
- Mesmo município, mesma modalidade
- Objeto textualmente IDÊNTICO
- A LLM não consegue diferenciar (e não deveria)

Solução:
- Mapear grupos de IDs com objeto idêntico
- Aceitar qualquer ID do grupo como resposta válida
- Ajustar métricas para refletir acurácia real

Uso:
    python scripts/07_map_identical_objects.py
"""

import pickle
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
OUTPUT_PATH = BASE_DIR / "data/identical_objects_mapping.json"


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparação, removendo diferenças irrelevantes.
    
    Remove:
    - Acentos (ã→a, é→e, ç→c, etc.)
    - Espaços extras
    - Pontuação final (., !, ?, ;, :)
    - Múltiplos espaços
    - Espaços antes/depois
    - Case-insensitive
    """
    if not texto:
        return ""
    
    # Converter para lowercase para comparação case-insensitive
    texto = texto.lower()
    
    # Remover acentos usando NFD (Normalization Form Decomposed)
    # NFD separa caracteres acentuados em base + acento
    # Depois filtra apenas caracteres ASCII (remove os acentos)
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(char for char in texto if unicodedata.category(char) != 'Mn')
    
    # Remover espaços extras no início/fim
    texto = texto.strip()
    
    # Remover pontuação final repetida (ex: "..." -> "")
    texto = re.sub(r'[.!?;:]+\s*$', '', texto)
    
    # Normalizar múltiplos espaços para um único espaço
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto

print("="*80)
print("MAPEAMENTO DE OBJETOS IDÊNTICOS")
print("="*80)

# Carregar cache
print(f"\nCarregando cache: {CACHE_PATH}")
with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)
print(f"Cache carregado: {len(cache)} notícias")

# Coletar todos os candidatos únicos
print("\nColetando candidatos únicos...")
candidatos_unicos = {}

for item in cache:
    candidatos = item.get("candidatos_top50", [])
    for cand in candidatos:
        id_proc = cand.get("id_processo")
        if id_proc and id_proc not in candidatos_unicos:
            candidatos_unicos[id_proc] = {
                "id": id_proc,
                "objeto": cand.get("objeto", "").strip(),
                "municipio": cand.get("municipio", "N/A"),
                "modalidade": cand.get("modalidade", "N/A"),
            }

print(f"Total de candidatos únicos: {len(candidatos_unicos)}")

# Mapear objeto normalizado → lista de IDs com texto original
print("\nMapeando objetos idênticos (com normalização)...")
objeto_normalizado_to_ids = defaultdict(list)

for id_proc, info in candidatos_unicos.items():
    obj_original = info["objeto"]
    if obj_original:  # Ignorar objetos vazios
        obj_normalizado = normalizar_texto(obj_original)
        objeto_normalizado_to_ids[obj_normalizado].append({
            "id": id_proc,
            "municipio": info["municipio"],
            "modalidade": info["modalidade"],
            "objeto_original": obj_original,  # Manter original para referência
        })

# Filtrar apenas objetos com múltiplos IDs
# Usar o primeiro objeto original como chave (representativo do grupo)
identical_groups = {}
for obj_norm, ids_list in objeto_normalizado_to_ids.items():
    if len(ids_list) > 1:
        # Ordenar por ID para consistência
        ids_list_sorted = sorted(ids_list, key=lambda x: x["id"])
        
        # Usar o objeto original do primeiro item como chave
        obj_key = ids_list_sorted[0]["objeto_original"]
        
        # Remover campo objeto_original da lista final (não necessário no JSON)
        ids_final = [{k: v for k, v in item.items() if k != "objeto_original"} 
                     for item in ids_list_sorted]
        
        identical_groups[obj_key] = ids_final

print(f"\nEncontrados {len(identical_groups)} objetos com múltiplos IDs")

# Estatísticas
total_processos_afetados = sum(len(ids) for ids in identical_groups.values())
max_grupo = max(len(ids) for ids in identical_groups.values()) if identical_groups else 0

print(f"Total de processos afetados: {total_processos_afetados}")
print(f"Maior grupo: {max_grupo} processos com mesmo objeto")

# Mostrar exemplos
print("\n" + "="*80)
print("EXEMPLOS DE GRUPOS (primeiros 5):")
print("="*80)
for i, (obj, ids_list) in enumerate(list(identical_groups.items())[:5], 1):
    print(f"\n{i}. Objeto: {obj[:100]}...")
    print(f"   IDs ({len(ids_list)}):")
    for info in ids_list:
        print(f"     - {info['id']} | {info['municipio']} | {info['modalidade']}")

# Salvar
print(f"\nSalvando em: {OUTPUT_PATH}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(identical_groups, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print("CONCLUÍDO")
print("="*80)
print(f"\nArquivo salvo: {OUTPUT_PATH}")
print(f"Grupos mapeados: {len(identical_groups)}")
print(f"Processos afetados: {total_processos_afetados}")
