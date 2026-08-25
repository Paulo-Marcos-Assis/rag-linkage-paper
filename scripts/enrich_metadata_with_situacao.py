#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_metadata_with_situacao.py
=================================
Enriquece metadata.pkl com o campo 'situacao' do CSV v2.

Garante correspondência correta entre registros usando id_processo_licitatorio.

Uso:
    python scripts/enrich_metadata_with_situacao.py
    python scripts/enrich_metadata_with_situacao.py --dry-run  # Apenas validação
"""

import pickle
import csv
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent

# Caminhos
METADATA_PATH = BASE_DIR / "models/vector_store/metadata.pkl"
CSV_PATH = BASE_DIR / "data/325K_processo_licitatorio_with_relations_v2.csv"
OUTPUT_PATH = BASE_DIR / "models/vector_store/metadata_with_situacao.pkl"
BACKUP_PATH = BASE_DIR / "models/vector_store/metadata_backup.pkl"


def carregar_metadata():
    """Carrega metadata.pkl atual."""
    print(f"Carregando metadata de: {METADATA_PATH}")
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f"  {len(metadata)} registros carregados")
    return metadata


def carregar_csv_como_dict():
    """Carrega CSV v2 e cria dicionário id_processo -> situacao."""
    print(f"\nCarregando CSV de: {CSV_PATH}")
    
    situacao_map = {}
    total_linhas = 0
    com_situacao = 0
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_linhas += 1
            id_proc = row.get("id_processo_licitatorio", "").strip()
            situacao = row.get("situacao", "").strip()
            
            if id_proc:
                situacao_map[id_proc] = situacao
                if situacao:
                    com_situacao += 1
    
    print(f"  {total_linhas} linhas processadas")
    print(f"  {len(situacao_map)} IDs únicos")
    print(f"  {com_situacao} com situacao preenchida ({100*com_situacao/total_linhas:.1f}%)")
    
    return situacao_map


def validar_correspondencia(metadata, situacao_map):
    """
    Valida correspondência entre metadata e CSV usando múltiplos campos.
    
    Verifica:
    1. IDs existem em ambos
    2. Campos comuns (municipio, modalidade, objeto) batem
    """
    print("\n" + "="*80)
    print("VALIDAÇÃO DE CORRESPONDÊNCIA")
    print("="*80)
    
    ids_metadata = set(str(m.get("id_processo_licitatorio", "")) for m in metadata)
    ids_csv = set(situacao_map.keys())
    
    # IDs em comum
    ids_comuns = ids_metadata & ids_csv
    ids_so_metadata = ids_metadata - ids_csv
    ids_so_csv = ids_csv - ids_metadata
    
    print(f"\nAnálise de IDs:")
    print(f"  IDs no metadata.pkl: {len(ids_metadata)}")
    print(f"  IDs no CSV v2: {len(ids_csv)}")
    print(f"  IDs em comum: {len(ids_comuns)} ({100*len(ids_comuns)/len(ids_metadata):.1f}%)")
    print(f"  Apenas no metadata: {len(ids_so_metadata)}")
    print(f"  Apenas no CSV: {len(ids_so_csv)}")
    
    if ids_so_metadata:
        print(f"\n⚠️  AVISO: {len(ids_so_metadata)} IDs do metadata não encontrados no CSV")
        print(f"  Primeiros 5: {list(ids_so_metadata)[:5]}")
    
    # Validar amostra de registros
    print(f"\nValidando amostra de 10 registros...")
    
    erros = []
    for i, meta_reg in enumerate(metadata[:10]):
        id_proc = str(meta_reg.get("id_processo_licitatorio", ""))
        
        if id_proc not in situacao_map:
            erros.append(f"  Registro {i}: ID {id_proc} não encontrado no CSV")
            continue
        
        print(f"\n  ✓ Registro {i}: ID {id_proc}")
        print(f"    Município: {meta_reg.get('municipio', 'N/A')}")
        print(f"    Modalidade: {meta_reg.get('modalidade', 'N/A')}")
        print(f"    Situacao (CSV): '{situacao_map[id_proc]}'")
    
    if erros:
        print("\n⚠️  Erros encontrados:")
        for erro in erros:
            print(erro)
        return False
    
    print("\n✅ Validação concluída com sucesso!")
    return True


def enriquecer_metadata(metadata, situacao_map):
    """Adiciona campo 'situacao' a cada registro do metadata."""
    print("\n" + "="*80)
    print("ENRIQUECENDO METADATA")
    print("="*80)
    
    metadata_enriquecido = []
    
    stats = {
        "total": len(metadata),
        "com_situacao": 0,
        "sem_situacao": 0,
        "id_nao_encontrado": 0,
    }
    
    for meta_reg in metadata:
        # Criar cópia do registro
        novo_reg = meta_reg.copy()
        
        id_proc = str(meta_reg.get("id_processo_licitatorio", ""))
        
        if id_proc in situacao_map:
            situacao = situacao_map[id_proc]
            novo_reg["situacao"] = situacao
            
            if situacao:
                stats["com_situacao"] += 1
            else:
                stats["sem_situacao"] += 1
        else:
            # ID não encontrado no CSV - adicionar campo vazio
            novo_reg["situacao"] = ""
            stats["id_nao_encontrado"] += 1
        
        metadata_enriquecido.append(novo_reg)
    
    print(f"\nEstatísticas:")
    print(f"  Total de registros: {stats['total']}")
    print(f"  Com situacao preenchida: {stats['com_situacao']} ({100*stats['com_situacao']/stats['total']:.1f}%)")
    print(f"  Com situacao vazia: {stats['sem_situacao']} ({100*stats['sem_situacao']/stats['total']:.1f}%)")
    print(f"  ID não encontrado no CSV: {stats['id_nao_encontrado']} ({100*stats['id_nao_encontrado']/stats['total']:.1f}%)")
    
    # Distribuição de valores de situacao
    situacoes = [reg["situacao"] for reg in metadata_enriquecido if reg["situacao"]]
    if situacoes:
        print(f"\nValores de situacao (top 10):")
        for valor, count in Counter(situacoes).most_common(10):
            print(f"  '{valor}': {count}")
    
    return metadata_enriquecido


def salvar_metadata(metadata_enriquecido, fazer_backup=True):
    """Salva metadata enriquecido."""
    print("\n" + "="*80)
    print("SALVANDO METADATA ENRIQUECIDO")
    print("="*80)
    
    # Backup do original
    if fazer_backup and METADATA_PATH.exists():
        print(f"\nCriando backup: {BACKUP_PATH}")
        import shutil
        shutil.copy2(METADATA_PATH, BACKUP_PATH)
        print("  ✓ Backup criado")
    
    # Salvar novo metadata
    print(f"\nSalvando metadata enriquecido: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(metadata_enriquecido, f)
    
    tamanho_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✓ Salvo com sucesso ({tamanho_mb:.1f} MB)")
    
    # Verificar integridade
    print("\nVerificando integridade do arquivo salvo...")
    with open(OUTPUT_PATH, "rb") as f:
        verificacao = pickle.load(f)
    
    print(f"  ✓ {len(verificacao)} registros carregados")
    print(f"  ✓ Primeiro registro tem campo 'situacao': {'situacao' in verificacao[0]}")
    
    if verificacao[0].get("situacao") is not None:
        print(f"  ✓ Valor: '{verificacao[0]['situacao']}'")


def main():
    parser = argparse.ArgumentParser(
        description="Enriquece metadata.pkl com campo situacao do CSV v2"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas valida, não salva alterações"
    )
    args = parser.parse_args()
    
    print("="*80)
    print(" ENRIQUECIMENTO DE METADATA COM CAMPO SITUACAO")
    print("="*80)
    
    # 1. Carregar dados
    metadata = carregar_metadata()
    situacao_map = carregar_csv_como_dict()
    
    # 2. Validar correspondência
    if not validar_correspondencia(metadata, situacao_map):
        print("\n❌ Validação falhou. Abortando.")
        return 1
    
    # 3. Enriquecer
    metadata_enriquecido = enriquecer_metadata(metadata, situacao_map)
    
    # 4. Salvar (se não for dry-run)
    if args.dry_run:
        print("\n" + "="*80)
        print("🔍 DRY-RUN: Nenhuma alteração foi salva")
        print("="*80)
        print("\nPara aplicar as alterações, execute sem --dry-run:")
        print("  python scripts/enrich_metadata_with_situacao.py")
    else:
        salvar_metadata(metadata_enriquecido)
        
        print("\n" + "="*80)
        print("✅ CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"\nArquivos gerados:")
        print(f"  Backup: {BACKUP_PATH}")
        print(f"  Novo metadata: {OUTPUT_PATH}")
        print(f"\nPróximos passos:")
        print(f"  1. Validar o novo arquivo: {OUTPUT_PATH}")
        print(f"  2. Se estiver OK, substituir o original:")
        print(f"     mv {OUTPUT_PATH} {METADATA_PATH}")
    
    return 0


if __name__ == "__main__":
    exit(main())
