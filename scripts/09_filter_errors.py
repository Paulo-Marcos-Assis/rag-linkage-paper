#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_filter_errors.py
===================
Filtra casos de erro do resultado JSON do pipeline single-call.

Extrai apenas as notícias onde acertou_llm=false para análise detalhada.
Opcionalmente carrega o cache TF-IDF para mostrar detalhes completos.

Uso:
    python scripts/09_filter_errors.py results/single_call_top10_n500_gpt-oss_20b.json
    python scripts/09_filter_errors.py results/single_call_top10_n500_gpt-oss_20b.json --verbose
    python scripts/09_filter_errors.py results/single_call_top10_n500_gpt-oss_20b.json --output results/errors_analysis.json
"""

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
DUPLICATAS_PATH = BASE_DIR / "data/duplicatas_mapping.json"
IDENTICAL_OBJECTS_PATH = BASE_DIR / "data/identical_objects_mapping.json"
META_PATH = BASE_DIR / "models/vector_store/metadata.pkl"


def carregar_cache() -> dict:
    """Carrega cache TF-IDF indexado por indice_par."""
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    
    # Indexar por indice_par para acesso rápido
    cache_dict = {}
    for item in cache:
        indice = str(item.get("indice_par", ""))
        cache_dict[indice] = item
    
    return cache_dict


def carregar_metadata() -> dict:
    """Carrega metadata completo indexado por id_processo_licitatorio."""
    if not META_PATH.exists():
        return {}
    with open(META_PATH, "rb") as f:
        metadata_list = pickle.load(f)
    
    # Indexar por id_processo_licitatorio para acesso rápido
    metadata_dict = {}
    for item in metadata_list:
        id_proc = str(item.get("id_processo_licitatorio", ""))
        if id_proc:
            metadata_dict[id_proc] = item
    
    return metadata_dict


def carregar_duplicatas() -> dict:
    """Carrega mapeamento de duplicatas."""
    if not DUPLICATAS_PATH.exists():
        return {}
    with open(DUPLICATAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("id_to_group", {})


def carregar_identical_objects() -> dict:
    """Carrega mapeamento de objetos idênticos."""
    if not IDENTICAL_OBJECTS_PATH.exists():
        return {}
    with open(IDENTICAL_OBJECTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encontrar_grupo_identical_objects(id_proc: str, identical_map: dict) -> tuple[str, list[str]] | None:
    """Retorna (objeto_texto, lista_de_ids) se o ID pertence a um grupo."""
    for obj_text, ids_list in identical_map.items():
        ids_no_grupo = [item["id"] for item in ids_list]
        if str(id_proc) in ids_no_grupo:
            return (obj_text, ids_list)
    return None


def analisar_erro(detalhe: dict, cache_dict: dict, duplicatas_map: dict, identical_map: dict, metadata_dict: dict, numero_noticia: int) -> dict:
    """
    Analisa um caso de erro em detalhes.
    
    Retorna informações enriquecidas sobre o erro.
    """
    indice_par = str(detalhe.get("indice_par", ""))
    id_gold = str(detalhe.get("id_processo_gold", ""))
    id_escolhido = str(detalhe.get("id_escolhido_llm", ""))
    rank_tfidf_gold = detalhe.get("rank_tfidf_topn")
    
    # Buscar dados do cache
    item_cache = cache_dict.get(indice_par, {})
    
    # Encontrar rank do escolhido
    rank_tfidf_escolhido = None
    if id_escolhido and id_escolhido != "None":
        candidatos = item_cache.get("candidatos_top50", [])
        for i, cand in enumerate(candidatos[:10], 1):
            if str(cand.get("id_processo", "")) == id_escolhido:
                rank_tfidf_escolhido = i
                break
    
    analise = {
        "numero_noticia": numero_noticia,
        "indice_par": indice_par,
        "id_gold": id_gold,
        "id_escolhido": id_escolhido,
        "rank_tfidf_gold": rank_tfidf_gold,
        "rank_tfidf_escolhido": rank_tfidf_escolhido,
        "municipio": item_cache.get("municipio", "N/A"),
        "modalidade": item_cache.get("modalidade", "N/A"),
        "numero_edital": item_cache.get("numero_edital", "N/A"),
        "objeto_noticia": item_cache.get("objeto", "N/A"),
        "zero_llm": detalhe.get("zero_llm", False),
        "falha_llm": detalhe.get("falha_llm", False),
    }
    
    # Informações sobre o gold
    candidatos = item_cache.get("candidatos_top50", [])
    objeto_gold_encontrado = False
    for cand in candidatos:  # Buscar em todos os 50 candidatos
        if str(cand.get("id_processo", "")) == id_gold:
            analise["objeto_gold"] = cand.get("objeto", "N/A")
            objeto_gold_encontrado = True
            break
    
    # Se não encontrou nos candidatos, buscar no metadata completo
    if not objeto_gold_encontrado and id_gold in metadata_dict:
        analise["objeto_gold"] = metadata_dict[id_gold].get("descricao_objeto", "N/A")
    
    # Informações sobre o escolhido (se houver)
    if id_escolhido and id_escolhido != "None":
        for cand in candidatos[:10]:  # Escolhido sempre está no top 10
            if str(cand.get("id_processo", "")) == id_escolhido:
                analise["objeto_escolhido"] = cand.get("objeto", "N/A")
                break
    
    # Verificar se gold está em duplicatas
    if id_gold in duplicatas_map:
        analise["gold_tem_duplicatas"] = True
        analise["grupo_duplicatas_gold"] = duplicatas_map[id_gold]
    else:
        analise["gold_tem_duplicatas"] = False
    
    # Verificar se gold está em objetos idênticos
    grupo_ident_gold = encontrar_grupo_identical_objects(id_gold, identical_map)
    if grupo_ident_gold:
        obj_text, ids_list = grupo_ident_gold
        analise["gold_tem_objetos_identicos"] = True
        analise["grupo_identical_gold"] = {
            "objeto": obj_text[:200],
            "total_ids": len(ids_list),
            "ids": [item["id"] for item in ids_list]
        }
    else:
        analise["gold_tem_objetos_identicos"] = False
    
    # Verificar se escolhido está em objetos idênticos
    if id_escolhido and id_escolhido != "None":
        grupo_ident_escolhido = encontrar_grupo_identical_objects(id_escolhido, identical_map)
        if grupo_ident_escolhido:
            obj_text, ids_list = grupo_ident_escolhido
            analise["escolhido_tem_objetos_identicos"] = True
            analise["grupo_identical_escolhido"] = {
                "objeto": obj_text[:200],
                "total_ids": len(ids_list),
                "ids": [item["id"] for item in ids_list]
            }
        else:
            analise["escolhido_tem_objetos_identicos"] = False
    
    # Listar todos os candidatos top 10
    analise["candidatos_top10"] = []
    for i, cand in enumerate(candidatos[:10], 1):
        analise["candidatos_top10"].append({
            "rank": i,
            "id": cand.get("id_processo", "N/A"),
            "objeto": cand.get("objeto", "N/A")[:150],
        })
    
    return analise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filtra e analisa casos de erro do pipeline single-call"
    )
    parser.add_argument("json_file", help="Arquivo JSON de resultados")
    parser.add_argument("--output", "-o", help="Arquivo de saída (padrão: results/errors_analysis.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostra detalhes no terminal")
    parser.add_argument("--limit", type=int, help="Limita número de erros a processar")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("FILTRO DE CASOS DE ERRO")
    print("=" * 80)
    print()
    
    # Carregar JSON
    print(f"Carregando resultados: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    detalhes = data.get("detalhes", [])
    print(f"  Total de notícias: {len(detalhes)}")
    
    # Filtrar erros
    erros = [d for d in detalhes if not d.get("acertou_llm", False)]
    print(f"  Casos de erro: {len(erros)}")
    print()
    
    if args.limit:
        erros = erros[:args.limit]
        print(f"  Limitando análise a {args.limit} erros")
        print()
    
    # Carregar dados auxiliares
    print("Carregando dados auxiliares...")
    cache_dict = carregar_cache()
    metadata_dict = carregar_metadata()
    duplicatas_map = carregar_duplicatas()
    identical_map = carregar_identical_objects()
    print(f"  Cache TF-IDF: {len(cache_dict)} notícias")
    print(f"  Metadata completo: {len(metadata_dict)} processos")
    print(f"  Duplicatas: {len(duplicatas_map)} IDs")
    print(f"  Objetos idênticos: {len(identical_map)} grupos")
    print()
    
    # Analisar erros
    print("Analisando erros...")
    erros_analisados = []
    
    for i, erro in enumerate(erros, 1):
        # Encontrar número da notícia (1-500) baseado no indice_par
        numero_noticia = int(erro.get("indice_par", 0)) + 1
        
        if args.verbose:
            print(f"\n{'=' * 80}")
            print(f"ERRO {i}/{len(erros)} - Notícia {numero_noticia}/500")
            print(f"{'=' * 80}")
        
        analise = analisar_erro(erro, cache_dict, duplicatas_map, identical_map, metadata_dict, numero_noticia)
        erros_analisados.append(analise)
        
        if args.verbose:
            print(f"Índice: {analise['indice_par']}")
            print(f"Município: {analise['municipio']}")
            print(f"Modalidade: {analise['modalidade']}")
            print(f"\nObjeto da notícia:")
            print(f"  {analise['objeto_noticia'][:200]}...")
            print(f"\nID Gold: {analise['id_gold']}")
            if "objeto_gold" in analise:
                print(f"  Objeto: {analise['objeto_gold'][:150]}...")
            print(f"  Rank TF-IDF: {analise['rank_tfidf_gold']}")
            
            if analise.get("zero_llm"):
                print(f"\nLLM respondeu: 0 (nenhum candidato corresponde)")
            elif analise.get("falha_llm"):
                print(f"\nLLM falhou (timeout ou resposta inválida)")
            else:
                print(f"\nID Escolhido: {analise['id_escolhido']}")
                print(f"  Rank TF-IDF: {analise['rank_tfidf_escolhido']}")
                if "objeto_escolhido" in analise:
                    print(f"  Objeto: {analise['objeto_escolhido'][:150]}...")
            
            if analise.get("gold_tem_duplicatas"):
                print(f"\n⚠ Gold tem duplicatas: {analise['grupo_duplicatas_gold']}")
            
            if analise.get("gold_tem_objetos_identicos"):
                info = analise["grupo_identical_gold"]
                print(f"\n⚠ Gold tem objetos idênticos ({info['total_ids']} IDs)")
                print(f"  IDs: {info['ids']}")
    
    print()
    print(f"Total de erros analisados: {len(erros_analisados)}")
    
    # Estatísticas
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DOS ERROS")
    print("=" * 80)
    
    zeros = sum(1 for e in erros_analisados if e.get("zero_llm"))
    falhas = sum(1 for e in erros_analisados if e.get("falha_llm"))
    escolhas_erradas = len(erros_analisados) - zeros - falhas
    
    print(f"\nTipos de erro:")
    print(f"  Escolha errada: {escolhas_erradas} ({100*escolhas_erradas/len(erros_analisados):.1f}%)")
    print(f"  LLM respondeu 0: {zeros} ({100*zeros/len(erros_analisados):.1f}%)")
    print(f"  Falha LLM: {falhas} ({100*falhas/len(erros_analisados):.1f}%)")
    
    # Distribuição por rank TF-IDF
    print(f"\nDistribuição por rank TF-IDF do gold:")
    rank_dist = defaultdict(int)
    for e in erros_analisados:
        rank = e.get("rank_tfidf_gold")
        if rank is None:
            rank_dist["Não encontrado"] += 1
        else:
            rank_dist[f"Rank {rank}"] += 1
    
    for rank, count in sorted(rank_dist.items()):
        print(f"  {rank}: {count}")
    
    # Gold com duplicatas/objetos idênticos
    gold_dup = sum(1 for e in erros_analisados if e.get("gold_tem_duplicatas"))
    gold_ident = sum(1 for e in erros_analisados if e.get("gold_tem_objetos_identicos"))
    
    print(f"\nGold com mapeamentos:")
    print(f"  Com duplicatas: {gold_dup} ({100*gold_dup/len(erros_analisados):.1f}%)")
    print(f"  Com objetos idênticos: {gold_ident} ({100*gold_ident/len(erros_analisados):.1f}%)")
    
    # Salvar
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = BASE_DIR / "results/errors_analysis.json"
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_erros": len(erros_analisados),
            "estatisticas": {
                "escolhas_erradas": escolhas_erradas,
                "zeros_llm": zeros,
                "falhas_llm": falhas,
                "gold_com_duplicatas": gold_dup,
                "gold_com_objetos_identicos": gold_ident,
            },
            "distribuicao_rank_tfidf": dict(rank_dist),
            "erros": erros_analisados,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados salvos em: {output_path}")
    print()


if __name__ == "__main__":
    main()
