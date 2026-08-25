#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_vector_store.py
========================
Constrói o vector store FAISS a partir dos processos licitatórios (325k registros).

Estratégia:
- Embedding: neuralmind/bert-large-portuguese-cased (BERTimbau Large)
- Campo vetorizado: descricao_objeto
- Registros com descricao_objeto nulo/vazio são descartados
- Metadados completos armazenados em paralelo

Uso:
    python scripts/01_build_vector_store.py
    python scripts/01_build_vector_store.py --batch-size 128 --limit 1000  # teste
"""

import os
import sys
import csv
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "data/processos_325k.csv"
OUTPUT_DIR = BASE_DIR / "models/vector_store"

BERT_MODEL_NAME = "neuralmind/bert-large-portuguese-cased"
MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Constrói o FAISS vector store")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_csv(csv_path: Path, limit: int = None) -> tuple[list[str], list[dict]]:
    """Carrega CSV e retorna textos e metadados alinhados."""
    textos = []
    metadados = []
    
    null_values = {"", "null", "none", "nan"}
    
    print(f"\n[1/4] Carregando CSV: {csv_path}")
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_lido = 0
        total_descartado = 0
        
        for row in reader:
            if limit and len(textos) >= limit:
                break
            
            total_lido += 1
            obj = row.get("descricao_objeto", "").strip()
            
            if obj.lower() in null_values:
                total_descartado += 1
                continue
            
            textos.append(obj)
            metadados.append({
                "id_processo_licitatorio": row.get("id_processo_licitatorio", ""),
                "numero_edital": row.get("numero_edital", ""),
                "municipio": row.get("municipio", ""),
                "ente": row.get("ente", ""),
                "unidade_gestora": row.get("unidade_gestora", ""),
                "uf": row.get("uf", ""),
                "modalidade": row.get("modalidade", ""),
                "descricao_objeto": obj,
            })
    
    print(f"    Total lido:        {total_lido:,}")
    print(f"    Descartados (null):{total_descartado:,}")
    print(f"    Para vetorizar:    {len(textos):,}")
    
    return textos, metadados


def gerar_embeddings(textos: list[str], batch_size: int, device: str) -> np.ndarray:
    """Gera embeddings com BERTimbau Large."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("\n[ERRO] torch ou transformers não instalado.")
        print("Execute: pip install -r requirements.txt")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[2/4] Carregando modelo BERT: {BERT_MODEL_NAME}")
    print(f"    Dispositivo: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.to(device)
    model.eval()

    total = len(textos)
    print(f"\n[3/4] Gerando embeddings para {total:,} registros")
    print(f"    Batch size: {batch_size} | Max seq length: {MAX_SEQ_LENGTH} tokens")

    all_embeddings = []
    tempo_inicio = time.time()

    for i in tqdm(range(0, total, batch_size), desc="Embeddings", unit="batch"):
        batch_textos = textos[i : i + batch_size]

        encoded = tokenizer(
            batch_textos,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # Mean pooling
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Normalização L2
        normalized = F.normalize(mean_embeddings, p=2, dim=1)

        all_embeddings.append(normalized.cpu().float().numpy())

    embeddings = np.vstack(all_embeddings)

    tempo_total = time.time() - tempo_inicio
    velocidade = total / tempo_total

    print(f"\n    Tempo total: {tempo_total/60:.1f} min")
    print(f"    Velocidade: {velocidade:.0f} registros/segundo")
    print(f"    Shape: {embeddings.shape}")

    return embeddings


def construir_faiss(embeddings: np.ndarray):
    """Constrói índice FAISS."""
    try:
        import faiss
    except ImportError:
        print("\n[ERRO] faiss-cpu não instalado.")
        print("Execute: pip install -r requirements.txt")
        sys.exit(1)
    
    dim = embeddings.shape[1]
    print(f"\n[4/4] Construindo índice FAISS (IndexFlatIP, dim={dim})")
    
    index = faiss.IndexFlatIP(dim)
    index_with_ids = faiss.IndexIDMap(index)
    
    ids = np.arange(len(embeddings), dtype=np.int64)
    index_with_ids.add_with_ids(embeddings.astype(np.float32), ids)
    
    print(f"    Vetores indexados: {index_with_ids.ntotal:,}")
    
    return index_with_ids


def salvar(index, metadados: list[dict]):
    """Salva índice FAISS e metadados."""
    import faiss
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5/5] Salvando artefatos em: {OUTPUT_DIR}/")
    
    index_path = OUTPUT_DIR / "faiss_index.bin"
    meta_path = OUTPUT_DIR / "metadata.pkl"
    
    faiss.write_index(index, str(index_path))
    print(f"    Índice FAISS: {index_path.name} ({index_path.stat().st_size/1e6:.1f} MB)")
    
    with open(meta_path, "wb") as f:
        pickle.dump(metadados, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Metadados: {meta_path.name} ({meta_path.stat().st_size/1e6:.1f} MB)")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(" BUILD VECTOR STORE — Processos Licitatórios")
    print("=" * 60)
    
    if args.limit:
        print(f"\nMODO TESTE: limitado a {args.limit} registros\n")
    
    textos, metadados = load_csv(CSV_PATH, limit=args.limit)
    
    if not textos:
        print("[ERRO] Nenhum registro válido encontrado no CSV.")
        sys.exit(1)
    
    embeddings = gerar_embeddings(textos, batch_size=args.batch_size, device=args.device)
    
    index = construir_faiss(embeddings)
    
    salvar(index, metadados)
    
    print("\n" + "=" * 60)
    print(" CONCLUÍDO COM SUCESSO")
    print(f" Total vetorizado: {len(metadados):,} processos licitatórios")
    print("=" * 60)


if __name__ == "__main__":
    main()
