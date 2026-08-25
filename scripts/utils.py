#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Funções auxiliares compartilhadas entre os scripts do pipeline RAG.
"""

import re
import numpy as np
from collections import Counter


def tokenize(text: str) -> list:
    """
    Tokenização simples para TF-IDF.
    
    Args:
        text: Texto a ser tokenizado
        
    Returns:
        Lista de tokens em lowercase
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def tfidf_cosine_similarity(s1: str, s2: str, idf_dict: dict) -> float:
    """
    Calcula similaridade TF-IDF cosine entre dois textos.
    
    Args:
        s1: Primeiro texto (query)
        s2: Segundo texto (documento)
        idf_dict: Dicionário {term: idf_score}
        
    Returns:
        Score de similaridade [0, 1]
    """
    tokens1 = tokenize(s1)
    tokens2 = tokenize(s2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Term Frequency
    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    
    # Normalização TF
    max_tf1 = max(tf1.values())
    max_tf2 = max(tf2.values())
    
    tf1_norm = {k: v/max_tf1 for k, v in tf1.items()}
    tf2_norm = {k: v/max_tf2 for k, v in tf2.items()}
    
    # TF-IDF
    tfidf1 = {k: v * idf_dict.get(k, 1.0) for k, v in tf1_norm.items()}
    tfidf2 = {k: v * idf_dict.get(k, 1.0) for k, v in tf2_norm.items()}
    
    # Cosine similarity
    all_terms = set(tfidf1.keys()) | set(tfidf2.keys())
    
    dot = sum(tfidf1.get(t, 0) * tfidf2.get(t, 0) for t in all_terms)
    norm1 = np.sqrt(sum(v**2 for v in tfidf1.values()))
    norm2 = np.sqrt(sum(v**2 for v in tfidf2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def filtrar_por_municipio_e_modalidade(
    metadata: list[dict], 
    municipio: str, 
    modalidade: str
) -> list[int]:
    """
    Filtra processos por município e modalidade (Filtro SQL).
    
    Args:
        metadata: Lista de metadados dos processos
        municipio: Nome do município
        modalidade: Nome da modalidade
        
    Returns:
        Lista de índices dos processos que passaram no filtro
    """
    municipio = municipio.strip().lower()
    modalidade = modalidade.strip().lower()
    
    indices = []
    for i, meta in enumerate(metadata):
        meta_municipio = meta.get("municipio", "").strip().lower()
        meta_modalidade = meta.get("modalidade", "").strip().lower()
        
        if meta_municipio == municipio and meta_modalidade == modalidade:
            indices.append(i)
    
    return indices


def avaliar_com_duplicatas(
    id_escolhido: str,
    id_gold: str,
    duplicatas_map: dict
) -> bool:
    """
    Verifica se acertou considerando duplicatas.
    
    Processos com descrição idêntica são considerados equivalentes.
    
    Args:
        id_escolhido: ID do processo escolhido
        id_gold: ID do processo correto (gold standard)
        duplicatas_map: Mapeamento {id: [ids_equivalentes]}
        
    Returns:
        True se acertou (mesmo ID ou ID equivalente)
    """
    id_escolhido = str(id_escolhido)
    id_gold = str(id_gold)
    
    if id_escolhido == id_gold:
        return True
    
    # Verificar se pertencem ao mesmo grupo de duplicatas
    grupo_gold = duplicatas_map.get(id_gold, [id_gold])
    
    return id_escolhido in grupo_gold


def avaliar_com_identical_objects(
    id_escolhido: str,
    id_gold: str,
    duplicatas_map: dict,
    identical_map: dict
) -> bool:
    """
    Avalia considerando duplicatas e objetos idênticos (após normalização).

    Retorna True se:
    - id_escolhido == id_gold (direto)
    - id_escolhido é duplicata de id_gold
    - ambos pertencem ao mesmo grupo de objetos idênticos normalizados
    """
    if avaliar_com_duplicatas(id_escolhido, id_gold, duplicatas_map):
        return True
    for ids_list in identical_map.values():
        ids_no_grupo = [item["id"] for item in ids_list]
        if id_escolhido in ids_no_grupo and id_gold in ids_no_grupo:
            return True
    return False


def calcular_metricas(ranks: list[int], total: int) -> dict:
    """
    Calcula métricas de avaliação.
    
    Args:
        ranks: Lista de posições do gold standard (1-indexed, None se não encontrado)
        total: Total de notícias avaliadas
        
    Returns:
        Dicionário com métricas
    """
    ranks_validos = [r for r in ranks if r is not None]
    
    metricas = {
        "total": total,
        "encontrados": len(ranks_validos),
        "match_rate": len(ranks_validos) / total if total > 0 else 0,
        "recall_at_1": sum(1 for r in ranks if r == 1) / total if total > 0 else 0,
        "mrr": np.mean([1/r for r in ranks_validos]) if ranks_validos else 0,
        "rank_medio": np.mean(ranks_validos) if ranks_validos else None,
        "rank_mediano": np.median(ranks_validos) if ranks_validos else None,
    }
    
    return metricas
