#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_check_id_equivalence.py
===========================
Verifica se dois ou mais IDs são considerados equivalentes pelo sistema.

Considera equivalentes IDs que:
1. São duplicatas (mesmo processo, IDs diferentes) - via duplicatas_mapping.json
2. Têm objetos textualmente idênticos - via identical_objects_mapping.json

Uso:
    python scripts/08_check_id_equivalence.py 97273 143734
    python scripts/08_check_id_equivalence.py 97273 143734 187497
    python scripts/08_check_id_equivalence.py --gold 187497 --escolhido 97273
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DUPLICATAS_PATH = BASE_DIR / "data/duplicatas_mapping.json"
IDENTICAL_OBJECTS_PATH = BASE_DIR / "data/identical_objects_mapping.json"


def carregar_duplicatas() -> dict:
    """Carrega mapeamento de duplicatas."""
    if not DUPLICATAS_PATH.exists():
        print(f"AVISO: Arquivo de duplicatas não encontrado: {DUPLICATAS_PATH}")
        return {}
    with open(DUPLICATAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("id_to_group", {})


def carregar_identical_objects() -> dict:
    """Carrega mapeamento de objetos idênticos."""
    if not IDENTICAL_OBJECTS_PATH.exists():
        print(f"AVISO: Arquivo de objetos idênticos não encontrado: {IDENTICAL_OBJECTS_PATH}")
        return {}
    with open(IDENTICAL_OBJECTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encontrar_grupo_duplicatas(id_proc: str, duplicatas_map: dict) -> list[str] | None:
    """
    Retorna lista de IDs duplicados do mesmo grupo, ou None se não for duplicata.
    """
    if str(id_proc) in duplicatas_map:
        return duplicatas_map[str(id_proc)]
    return None


def encontrar_grupo_identical_objects(id_proc: str, identical_map: dict) -> tuple[str, list[str]] | None:
    """
    Retorna (objeto_texto, lista_de_ids) se o ID pertence a um grupo de objetos idênticos.
    Retorna None caso contrário.
    """
    for obj_text, ids_list in identical_map.items():
        ids_no_grupo = [item["id"] for item in ids_list]
        if str(id_proc) in ids_no_grupo:
            return (obj_text, ids_list)
    return None


def verificar_equivalencia(ids: list[str], duplicatas_map: dict, identical_map: dict) -> dict:
    """
    Verifica se todos os IDs fornecidos são equivalentes.
    
    Retorna dict com:
    - sao_equivalentes: bool
    - tipo_equivalencia: "duplicatas" | "identical_objects" | "ambos" | None
    - grupo_duplicatas: list[str] | None
    - grupo_identical: dict | None
    """
    if len(ids) < 2:
        return {
            "sao_equivalentes": False,
            "erro": "Forneça pelo menos 2 IDs para comparar"
        }
    
    # Normalizar IDs para string
    ids = [str(id_) for id_ in ids]
    
    # Verificar duplicatas - TODOS os IDs devem estar no mesmo grupo
    primeiro_grupo_dup = None
    duplicatas_equivalentes = False
    
    for id_ in ids:
        grupo = encontrar_grupo_duplicatas(id_, duplicatas_map)
        if grupo:
            grupo_sorted = tuple(sorted(grupo))
            if primeiro_grupo_dup is None:
                primeiro_grupo_dup = grupo_sorted
            elif primeiro_grupo_dup != grupo_sorted:
                # IDs pertencem a grupos diferentes
                primeiro_grupo_dup = None
                break
        else:
            # Este ID não está em nenhum grupo de duplicatas
            primeiro_grupo_dup = None
            break
    
    # Todos os IDs estão no mesmo grupo de duplicatas?
    if primeiro_grupo_dup is not None:
        # Verificar se TODOS os IDs fornecidos estão nesse grupo
        if all(str(id_) in primeiro_grupo_dup for id_ in ids):
            duplicatas_equivalentes = True
    
    # Verificar objetos idênticos - TODOS os IDs devem estar no mesmo grupo
    primeiro_grupo_ident = None
    identical_info = None
    identical_equivalentes = False
    
    for id_ in ids:
        resultado = encontrar_grupo_identical_objects(id_, identical_map)
        if resultado:
            obj_text, ids_list = resultado
            ids_no_grupo = tuple(sorted([item["id"] for item in ids_list]))
            if primeiro_grupo_ident is None:
                primeiro_grupo_ident = ids_no_grupo
                identical_info = {"objeto": obj_text, "ids": ids_list}
            elif primeiro_grupo_ident != ids_no_grupo:
                # IDs pertencem a grupos diferentes
                primeiro_grupo_ident = None
                identical_info = None
                break
        else:
            # Este ID não está em nenhum grupo de objetos idênticos
            primeiro_grupo_ident = None
            identical_info = None
            break
    
    # Todos os IDs estão no mesmo grupo de objetos idênticos?
    if primeiro_grupo_ident is not None:
        # Verificar se TODOS os IDs fornecidos estão nesse grupo
        if all(str(id_) in primeiro_grupo_ident for id_ in ids):
            identical_equivalentes = True
    
    # Determinar resultado
    sao_equivalentes = duplicatas_equivalentes or identical_equivalentes
    
    tipo = None
    if duplicatas_equivalentes and identical_equivalentes:
        tipo = "ambos"
    elif duplicatas_equivalentes:
        tipo = "duplicatas"
    elif identical_equivalentes:
        tipo = "identical_objects"
    
    resultado = {
        "sao_equivalentes": sao_equivalentes,
        "tipo_equivalencia": tipo,
    }
    
    if duplicatas_equivalentes and primeiro_grupo_dup:
        resultado["grupo_duplicatas"] = list(primeiro_grupo_dup)
    
    if identical_equivalentes and identical_info:
        resultado["grupo_identical"] = identical_info
    
    return resultado


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verifica se IDs são equivalentes (duplicatas ou objetos idênticos)"
    )
    parser.add_argument("ids", nargs="*", help="IDs para verificar equivalência")
    parser.add_argument("--gold", type=str, help="ID gold (alternativa)")
    parser.add_argument("--escolhido", type=str, help="ID escolhido (alternativa)")
    
    args = parser.parse_args()
    
    # Coletar IDs
    ids = list(args.ids)
    if args.gold:
        ids.append(args.gold)
    if args.escolhido:
        ids.append(args.escolhido)
    
    if len(ids) < 2:
        print("ERRO: Forneça pelo menos 2 IDs para comparar")
        print("\nExemplos:")
        print("  python scripts/08_check_id_equivalence.py 97273 143734")
        print("  python scripts/08_check_id_equivalence.py --gold 187497 --escolhido 97273")
        sys.exit(1)
    
    print("=" * 80)
    print("VERIFICAÇÃO DE EQUIVALÊNCIA DE IDs")
    print("=" * 80)
    print()
    
    # Carregar dados
    print("Carregando mapeamentos...")
    duplicatas_map = carregar_duplicatas()
    identical_map = carregar_identical_objects()
    print(f"  Duplicatas: {len(duplicatas_map)} IDs mapeados")
    print(f"  Objetos idênticos: {len(identical_map)} grupos")
    print()
    
    # Verificar
    print(f"IDs a verificar: {', '.join(ids)}")
    print()
    
    resultado = verificar_equivalencia(ids, duplicatas_map, identical_map)
    
    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print()
    
    if resultado["sao_equivalentes"]:
        print("✓ SIM - Os IDs são EQUIVALENTES")
        print(f"  Tipo: {resultado['tipo_equivalencia']}")
        print()
        
        if "grupo_duplicatas" in resultado:
            print("  Grupo de duplicatas:")
            for id_ in resultado["grupo_duplicatas"]:
                marca = "→" if id_ in ids else " "
                print(f"    {marca} {id_}")
            print()
        
        if "grupo_identical" in resultado:
            info = resultado["grupo_identical"]
            print("  Grupo de objetos idênticos:")
            print(f"    Objeto: {info['objeto'][:100]}...")
            print(f"    IDs no grupo ({len(info['ids'])}):")
            for item in info["ids"]:
                marca = "→" if item["id"] in ids else " "
                print(f"      {marca} {item['id']} | {item['municipio']} | {item['modalidade']}")
            print()
        
        print("CONCLUSÃO: Se o sistema escolheu um desses IDs e o gold é outro,")
        print("           isso seria considerado um ACERTO (HIT).")
    else:
        print("✗ NÃO - Os IDs NÃO são equivalentes")
        print()
        print("CONCLUSÃO: Se o sistema escolheu um ID diferente do gold,")
        print("           isso seria considerado um ERRO (MISS).")
        print()
        
        # Mostrar informações individuais
        print("Informações individuais:")
        for id_ in ids:
            print(f"\n  ID {id_}:")
            
            grupo_dup = encontrar_grupo_duplicatas(id_, duplicatas_map)
            if grupo_dup:
                print(f"    - Pertence a grupo de duplicatas: {grupo_dup}")
            else:
                print(f"    - Não é duplicata")
            
            grupo_ident = encontrar_grupo_identical_objects(id_, identical_map)
            if grupo_ident:
                obj_text, ids_list = grupo_ident
                print(f"    - Pertence a grupo de objetos idênticos ({len(ids_list)} IDs)")
                print(f"      Objeto: {obj_text[:80]}...")
            else:
                print(f"    - Não tem objeto idêntico a outros")
    
    print()


if __name__ == "__main__":
    main()
