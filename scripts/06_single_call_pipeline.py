
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_single_call_pipeline.py
==========================
Pipeline de reranking com UMA ÚNICA chamada LLM por notícia.

DIFERENÇA vs 05_tournament_pipeline.py:
- Usa top-N candidatos do cache TF-IDF (padrão: 10)
- Faz apenas 1 chamada LLM por notícia (sem torneio)
- ~11x mais rápido que o tournament 10×5
- Tradeoff: Recall@10 ~91% vs Recall@50 ~96%

Uso:
    python scripts/06_single_call_pipeline.py --limit 10
    python scripts/06_single_call_pipeline.py --top 20
    python scripts/06_single_call_pipeline.py --top 10 --limit 100 --verbose
    RERANK_MODEL=qwen2.5:14b python scripts/06_single_call_pipeline.py
"""

import os
import json
import argparse
import pickle
import time
import signal
import re
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import avaliar_com_duplicatas

BASE_DIR = Path(__file__).parent.parent

# Caminhos padrão (podem ser sobrescritos via argparse)
DEFAULT_CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
DUPLICATAS_PATH = BASE_DIR / "data/duplicatas_mapping.json"
IDENTICAL_OBJECTS_PATH = BASE_DIR / "data/identical_objects_mapping.json"

# LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-oss:20b")

# Configuração
TOP_N = 10                  # quantos candidatos TF-IDF alimentar ao LLM
MAX_RETRIES = 1             # 1 = uma tentativa, sem retry; 0 = desativa LLM
LLM_TIMEOUT_SECONDS = 300



def carregar_cache_tfidf(cache_path: Path, limit: int = None) -> list[dict]:
    """Carrega cache TF-IDF."""
    if not cache_path.exists():
        print(f"ERRO: Cache nao encontrado em {cache_path}")
        sys.exit(1)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if limit:
        cache = cache[:limit]
    return cache


def carregar_duplicatas() -> dict:
    """Carrega mapa de duplicatas."""
    if not DUPLICATAS_PATH.exists():
        return {}
    with open(DUPLICATAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_identical_objects() -> dict:
    """Carrega mapa de objetos idênticos."""
    if not IDENTICAL_OBJECTS_PATH.exists():
        print(f"AVISO: Mapa de objetos idênticos não encontrado em {IDENTICAL_OBJECTS_PATH}")
        return {}
    with open(IDENTICAL_OBJECTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def avaliar_com_identical_objects(id_escolhido: str, id_gold: str, duplicatas_map: dict, identical_map: dict) -> bool:
    """
    Avalia considerando:
    1. Duplicatas (mesmo processo, IDs diferentes)
    2. Objetos idênticos (processos diferentes, texto igual)
    
    Retorna True se:
    - id_escolhido == id_gold (direto)
    - id_escolhido é duplicata de id_gold
    - ambos pertencem ao mesmo grupo de objetos idênticos
    """
    # Avaliação normal com duplicatas
    if avaliar_com_duplicatas(id_escolhido, id_gold, duplicatas_map):
        return True
    
    # Verificar se ambos pertencem ao mesmo grupo de objetos idênticos
    for obj_text, ids_list in identical_map.items():
        ids_no_grupo = [item["id"] for item in ids_list]
        if id_escolhido in ids_no_grupo and id_gold in ids_no_grupo:
            return True  # mesmo objeto, edital diferente: Agora é aceitável
    
    return False


def carregar_checkpoint(checkpoint_path: Path) -> dict | None:
    """Carrega checkpoint salvo anteriormente, se existir."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def salvar_checkpoint(checkpoint_path: Path, checkpoint: dict):
    """Salva checkpoint após cada notícia processada."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def single_call_rerank(
    objeto_noticia: str,
    candidatos: list[dict],
    llm,
    top_n: int = TOP_N,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False,
) -> int | None:
    """
    Faz uma única chamada LLM para selecionar o melhor candidato entre os top-N.

    Returns:
        int  -> índice (0-based) do candidato selecionado
        None -> LLM respondeu 0 (nenhum corresponde) ou falha
    """
    candidatos_usados = candidatos[:top_n]

    if not candidatos_usados:
        return None

    if len(candidatos_usados) == 1:
        return 0

    if verbose:
        print(f"\n{'='*80}")
        print(f"OBJETO DA NOTICIA:")
        print(f"{objeto_noticia[:200]}..." if len(objeto_noticia) > 200 else objeto_noticia)
        print(f"\nCANDIDATOS ({len(candidatos_usados)}):")

    candidatos_text = ""
    for i, cand in enumerate(candidatos_usados, 1):
        obj_text = str(cand.get("objeto", "")).strip()
        candidatos_text += f"{i}. {obj_text}\n\n"
        if verbose:
            truncado = f"{obj_text[:100]}..." if len(obj_text) > 100 else obj_text
            print(f"  {i}. ID={cand.get('id_processo', 'N/A')} | {truncado}")

    prompt = f"""
Você é um sistema de reranking semântico de alta precisão.

===============================================================================
CONTEXTO DA TAREFA
===============================================================================

Você está na ETAPA FINAL de um pipeline de recuperação semântica.

ETAPAS ANTERIORES:
1. Uma notícia sobre licitação foi publicada
2. Um sistema TF-IDF buscou na base de 315.052 processos licitatórios de SC
3. O TF-IDF retornou os {len(candidatos_usados)} candidatos mais similares semanticamente

SUA TAREFA AGORA:
Você receberá o objeto da notícia e os {len(candidatos_usados)} candidatos pré-filtrados.

Sua missão é identificar qual desses candidatos (se houver) corresponde
ao MESMO processo licitatório descrito na notícia.

**Atente-se ao top 1 do rank, pois tem grandes chances de ser  o gold. mas não limite-se a ele, pois é possivel que não seja.**

===============================================================================
OBJETO DA NOTÍCIA
===============================================================================
{objeto_noticia}

===============================================================================
CANDIDATOS ({len(candidatos_usados)})
===============================================================================
{candidatos_text}

=======================================
O candidato correto pode estar em QUALQUER posição (1, 2, 3... ou nenhum).

Se nenhum candidato corresponder, retorne 0.

===============================================================================
SAÍDA
===============================================================================

Retorne APENAS UM valor:


1-{len(candidatos_usados)} -> índice do único candidato mais compatível
OU: 0  
(0 Significa que nenhum candidato corresponde)

Sem explicações.
Sem texto extra.
Sem múltiplos índices.
"""

    from langchain_core.messages import HumanMessage

    def timeout_handler(signum, frame):
        raise TimeoutError("LLM timeout")

    signal.signal(signal.SIGALRM, timeout_handler)

    for tentativa in range(1, max_retries + 1):
        try:
            if verbose:
                print(f"\n[Tentativa {tentativa}/{max_retries}]", flush=True)

            signal.alarm(LLM_TIMEOUT_SECONDS)

            inicio = time.time()
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                resultado = response.content.strip()
            finally:
                signal.alarm(0)

            tempo = time.time() - inicio

            if verbose:
                print(f"\n[LLM RESPOSTA] Tempo: {tempo:.2f}s")
                print(f"Resposta bruta: '{resultado}'", flush=True)

            numeros = re.findall(r'\b(\d+)\b', resultado)

            for num in numeros:
                num_int = int(num)

                if num_int == 0:
                    if verbose:
                        print("LLM indicou: nenhum candidato corresponde (0)", flush=True)
                    return "zero"

                if 1 <= num_int <= len(candidatos_usados):
                    idx = num_int - 1
                    if verbose:
                        print(f"\n[SELECIONADO] Candidato {num_int}: ID={candidatos_usados[idx].get('id_processo', 'N/A')}")
                    return idx

            if verbose:
                print(f"Resposta inválida: nenhum número válido em '{resultado}' (esperava 1-{len(candidatos_usados)})", flush=True)

        except TimeoutError:
            print(f"TIMEOUT (tentativa {tentativa}/{max_retries})", flush=True)
        except Exception as e:
            print(f"ERRO: {type(e).__name__}: {str(e)}", flush=True)

        time.sleep(tentativa * 2)

    print("FALHA DEFINITIVA DO RERANKING", flush=True)
    return "falha"


def avaliar_single_call(
    cache_tfidf: list[dict],
    llm,
    duplicatas_map: dict,
    identical_map: dict,
    top_n: int = TOP_N,
    verbose: bool = False,
    checkpoint_path: Path = None,
):
    """Avalia pipeline single-call com suporte a objetos idênticos."""

    # Tentar retomar de checkpoint
    checkpoint = None
    inicio_idx = 0
    if checkpoint_path:
        checkpoint = carregar_checkpoint(checkpoint_path)
        if checkpoint:
            inicio_idx = checkpoint["proximo_idx"]
            print(f"\n[CHECKPOINT] Retomando do indice {inicio_idx} ({inicio_idx}/{len(cache_tfidf)} ja processados)")
            sys.stdout.flush()

    stats = checkpoint["stats"] if checkpoint else {
        "total": len(cache_tfidf),
        "top_n": top_n,
        "sem_candidatos_sql": 0,
        "com_candidatos": 0,
        "falhas_llm": 0,       # timeout / invalid response
        "zeros_llm": 0,        # LLM respondeu 0 (nenhum match)
    }

    metricas_tfidf = checkpoint["metricas_tfidf"] if checkpoint else {
        "acertos_top1": 0,
        "encontrados_topn": 0,
        "ranks": [],
    }

    metricas_llm = checkpoint["metricas_llm"] if checkpoint else {
        "acertos": 0,
        "ranks": [],
    }

    custo_total = checkpoint["custo_total"] if checkpoint else 0
    detalhes = checkpoint["detalhes"] if checkpoint else []

    cache_restante = cache_tfidf[inicio_idx:]

    use_tqdm = not verbose
    iterator = tqdm(
        cache_restante, desc="Single-call rerank", unit="notícia",
        disable=not use_tqdm, initial=inicio_idx, total=len(cache_tfidf)
    ) if use_tqdm else cache_restante

    for noticia_idx, item in enumerate(iterator, start=inicio_idx):
        indice_par = item.get("indice_par")
        id_gold = str(item.get("id_gold", ""))
        candidatos = item.get("candidatos_top50", [])
        objeto = item.get("objeto", "")

        # Sem candidatos
        if item.get("sem_candidatos_sql", False) or not candidatos:
            stats["sem_candidatos_sql"] += 1
            detalhes.append({
                "indice_par": indice_par,
                "id_processo_gold": id_gold,
                "sem_candidatos_sql": True,
            })
            if checkpoint_path:
                salvar_checkpoint(checkpoint_path, {
                    "proximo_idx": noticia_idx + 1,
                    "stats": stats,
                    "metricas_tfidf": metricas_tfidf,
                    "metricas_llm": metricas_llm,
                    "custo_total": custo_total,
                    "detalhes": detalhes,
                })
            continue

        stats["com_candidatos"] += 1

        # Avaliar TF-IDF baseline (top-N)
        candidatos_topn = candidatos[:top_n]
        rank_tfidf = None
        for rank, cand in enumerate(candidatos_topn, 1):
            if avaliar_com_identical_objects(cand['id_processo'], id_gold, duplicatas_map, identical_map):
                rank_tfidf = rank
                metricas_tfidf["encontrados_topn"] += 1
                metricas_tfidf["ranks"].append(rank)
                if rank == 1:
                    metricas_tfidf["acertos_top1"] += 1
                break

        if verbose:
            print(f"\n\n{'='*80}")
            print(f"PROCESSANDO NOTICIA {noticia_idx+1}/{len(cache_tfidf)}")
            print(f"ID Gold: {id_gold}")
            print(f"Municipio: {item.get('municipio', 'N/A')}")
            print(f"Modalidade: {item.get('modalidade', 'N/A')}")
            numero_edital = item.get('numero_edital', 'N/A')
            if numero_edital and numero_edital != 'N/A':
                print(f"Edital: {numero_edital}")
            if rank_tfidf:
                print(f"Gold no rank TF-IDF: {rank_tfidf}/{top_n}")
            else:
                print(f"Gold NAO encontrado no top-{top_n} TF-IDF")
            print(f"{'='*80}")

        # Single LLM call
        resultado_rerank = single_call_rerank(
            objeto, candidatos, llm,
            top_n=top_n, verbose=verbose
        )
        custo_total += 1

        rank_llm = None
        acertou_llm = False
        id_escolhido = None
        idx_escolhido = None

        if resultado_rerank == "zero":
            stats["zeros_llm"] += 1
            if verbose:
                print(f"\n[RESULTADO] LLM respondeu 0 - nenhum candidato corresponde", flush=True)
        elif resultado_rerank == "falha":
            stats["falhas_llm"] += 1
            if verbose:
                print(f"\n[RESULTADO] FALHA - LLM nao retornou resposta valida", flush=True)
        else:
            idx_escolhido = resultado_rerank
            id_escolhido = candidatos[idx_escolhido]['id_processo']
            objeto_escolhido = candidatos[idx_escolhido].get('objeto', '')[:150]
            acertou_llm = avaliar_com_identical_objects(id_escolhido, id_gold, duplicatas_map, identical_map)

            objeto_gold = None
            for cand in candidatos_topn:
                if avaliar_com_identical_objects(cand['id_processo'], id_gold, duplicatas_map, identical_map):
                    objeto_gold = cand.get('objeto', '')[:150]
                    break

            if acertou_llm:
                metricas_llm["acertos"] += 1
                rank_llm = 1
                metricas_llm["ranks"].append(1)
                if verbose:
                    print(f"\n[RESULTADO] ACERTO!")
                    print(f"  ID escolhido: {id_escolhido}")
                    print(f"  Objeto: {objeto_escolhido}...")
                    print(f"  == ID gold: {id_gold}")
            else:
                if verbose:
                    print(f"\n[RESULTADO] ERRO!")
                    print(f"  ID escolhido: {id_escolhido}")
                    print(f"  Objeto escolhido: {objeto_escolhido}...")
                    print(f"  != ID gold: {id_gold}")
                    if objeto_gold:
                        print(f"  Objeto gold: {objeto_gold}...")
                    if rank_tfidf:
                        print(f"  Rank TF-IDF do gold: {rank_tfidf}")

        detalhes.append({
            "indice_par": indice_par,
            "id_processo_gold": id_gold,
            "rank_tfidf_topn": rank_tfidf,
            "rank_llm": rank_llm,
            "acertou_llm": acertou_llm,
            "id_escolhido_llm": id_escolhido,
            "falha_llm": resultado_rerank == "falha",
            "zero_llm": resultado_rerank == "zero",
        })

        if checkpoint_path:
            salvar_checkpoint(checkpoint_path, {
                "proximo_idx": noticia_idx + 1,
                "stats": stats,
                "metricas_tfidf": metricas_tfidf,
                "metricas_llm": metricas_llm,
                "custo_total": custo_total,
                "detalhes": detalhes,
            })

    # Métricas finais
    n = stats["com_candidatos"]

    metricas_tfidf_final = {
        f"match_rate_top{top_n}": metricas_tfidf["encontrados_topn"] / n if n > 0 else 0,
        "precision_at_1": metricas_tfidf["acertos_top1"] / n if n > 0 else 0,
        "mrr": np.mean([1 / r for r in metricas_tfidf["ranks"]]) if metricas_tfidf["ranks"] else 0,
    }

    metricas_llm_final = {
        "accuracy": metricas_llm["acertos"] / n if n > 0 else 0,
        "mrr": np.mean([1 / r for r in metricas_llm["ranks"]]) if metricas_llm["ranks"] else 0,
    }

    return (metricas_tfidf_final, metricas_llm_final, stats, custo_total, detalhes)


def main():
    parser = argparse.ArgumentParser(description="Single-call LLM reranking pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar número de notícias (ex: --limit 10)")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help=f"Quantos candidatos TF-IDF usar (padrão: {TOP_N})")
    parser.add_argument("--cache", type=str, default=None,
                        help="Caminho customizado para cache TF-IDF (ex: models/test/cache_tfidf_top50_test.pkl)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Modo verbose: mostra candidatos e resposta LLM")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Arquivo de checkpoint customizado")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Desativa salvamento de checkpoint")
    args = parser.parse_args()
    
    # Determinar cache path
    cache_path = Path(args.cache) if args.cache else DEFAULT_CACHE_PATH

    print("\n" + "=" * 80)
    print(" SINGLE-CALL LLM RERANKING PIPELINE")
    print("=" * 80)

    print(f"\nModelo LLM:  {RERANK_MODEL}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Cache:       {cache_path}")
    print(f"Top-N:       {args.top} candidatos por notícia")
    print(f"LLM calls:   1 por notícia")
    if args.verbose:
        print(f"Modo:        VERBOSE")
    sys.stdout.flush()

    # Carregar dados
    print("\nCarregando cache TF-IDF...")
    cache_tfidf = carregar_cache_tfidf(cache_path, args.limit)
    duplicatas_map = carregar_duplicatas()
    identical_map = carregar_identical_objects()

    print(f"{len(cache_tfidf)} notícias carregadas")
    n_dup_grupos = duplicatas_map.get('total_grupos', len(duplicatas_map.get('id_to_group', {})))
    n_dup_processos = duplicatas_map.get('total_processos_duplicados', 0)
    print(f"{n_dup_grupos} grupos de duplicatas ({n_dup_processos} processos)")
    print(f"{len(identical_map)} grupos de objetos idênticos mapeados")

    # Inicializar LLM
    print(f"\nInicializando LLM ({RERANK_MODEL})...")

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=RERANK_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    # Checkpoint
    checkpoint_path = None
    if not args.no_checkpoint:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            n = args.limit if args.limit else "all"
            dataset_suffix = "_test" if "test" in str(cache_path) else ""
            checkpoint_path = BASE_DIR / f"results/checkpoint_single_top{args.top}_n{n}{dataset_suffix}_{RERANK_MODEL.replace(':', '_')}.json"
        checkpoint_path.parent.mkdir(exist_ok=True)
        print(f"Checkpoint: {checkpoint_path}")
        sys.stdout.flush()

    # Executar
    tempo_inicio = time.time()

    (metricas_tfidf, metricas_llm, stats, custo, detalhes) = avaliar_single_call(
        cache_tfidf, llm, duplicatas_map, identical_map,
        top_n=args.top,
        verbose=args.verbose,
        checkpoint_path=checkpoint_path,
    )

    tempo_total = time.time() - tempo_inicio

    # Resultados
    print("\n" + "=" * 80)
    print(" RESULTADOS FINAIS")
    print("=" * 80)

    print("\nESTATÍSTICAS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nTF-IDF (top-{args.top}):")
    for k, v in metricas_tfidf.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nLLM SINGLE-CALL ({RERANK_MODEL}):")
    for k, v in metricas_llm.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    ganho = metricas_llm["accuracy"] - metricas_tfidf["precision_at_1"]

    print("\nCOMPARAÇÃO:")
    print(f"  ganho accuracy vs TF-IDF@1: {ganho:+.4f}")

    print("\nCUSTO:")
    print(f"  chamadas_llm: {custo}")
    print(f"  tempo_total_min: {tempo_total/60:.1f}")

    # Salvar JSON
    dataset_suffix = "_test" if "test" in str(cache_path) else ""
    output_path = BASE_DIR / f"results/single_call_top{args.top}_n{len(cache_tfidf)}{dataset_suffix}_{RERANK_MODEL.replace(':', '_')}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "configuracao": {
                "modelo_llm": RERANK_MODEL,
                "ollama_host": OLLAMA_HOST,
                "top_n": args.top,
                "max_retries": MAX_RETRIES,
                "llm_timeout_s": LLM_TIMEOUT_SECONDS,
            },
            "metricas_tfidf": metricas_tfidf,
            "metricas_llm": metricas_llm,
            "estatisticas": stats,
            "custo": {
                "chamadas_llm": custo,
                "tempo_total_s": tempo_total,
            },
            "detalhes": detalhes,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em:")
    print(f"   {output_path}")

    # Remover checkpoint ao concluir com sucesso
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint removido (execucao completa)")


if __name__ == "__main__":
    main()
