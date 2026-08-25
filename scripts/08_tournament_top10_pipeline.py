#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_tournament_top10_pipeline.py
================================
Pipeline de reranking com Tournament Selection em top-10 candidatos.

DIFERENÇA vs 06_single_call_pipeline.py:
- Usa top-10 candidatos do cache TF-IDF (igual ao 06)
- Faz 2 rounds de 5 candidatos (5+5) + 1 final
- Total: 3 chamadas LLM por notícia

DIFERENÇA vs 05_tournament_pipeline.py:
- Usa top-10 em vez de top-50
- 2 rounds (5+5) em vez de 10 rounds (5×10)
- Mais rápido, mas com recall menor

Uso:
    python scripts/08_tournament_top10_pipeline.py --limit 10
    python scripts/08_tournament_top10_pipeline.py --verbose
    RERANK_MODEL=qwen2.5:14b python scripts/08_tournament_top10_pipeline.py
"""

import os
import json
import argparse
import pickle
import time
import signal
import re
import sys
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import avaliar_com_duplicatas

BASE_DIR = Path(__file__).parent.parent

# Caminhos
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
DUPLICATAS_PATH = BASE_DIR / "data/duplicatas_mapping.json"
IDENTICAL_OBJECTS_PATH = BASE_DIR / "data/identical_objects_mapping.json"

# LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-oss:20b")

# Configuração
TOP_N = 10                  # quantos candidatos TF-IDF usar
GROUP_SIZE = 5              # tamanho de cada grupo no torneio
MAX_RETRIES = 1             # tentativas por round
LLM_TIMEOUT_SECONDS = 300
SEED = 42                   # reprodutibilidade

# Seed para reprodutibilidade
random.seed(SEED)
np.random.seed(SEED)


def carregar_cache_tfidf(limit: int = None) -> list[dict]:
    """Carrega cache TF-IDF."""
    if not CACHE_PATH.exists():
        print(f"ERRO: Cache não encontrado em {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "rb") as f:
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


def carregar_checkpoint(checkpoint_path: Path) -> dict | None:
    """Carrega checkpoint se existir."""
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


def avaliar_com_identical_objects(id_escolhido: str, id_gold: str, duplicatas_map: dict, identical_map: dict) -> bool:
    """
    Avalia considerando:
    1. Duplicatas (mesmo processo, IDs diferentes)
    2. Objetos idênticos (processos diferentes, texto igual)
    """
    # Avaliação normal com duplicatas
    if avaliar_com_duplicatas(id_escolhido, id_gold, duplicatas_map):
        return True
    
    # Verificar se ambos pertencem ao mesmo grupo de objetos idênticos
    for obj_text, ids_list in identical_map.items():
        ids_no_grupo = [item["id"] for item in ids_list]
        if id_escolhido in ids_no_grupo and id_gold in ids_no_grupo:
            return True
    
    return False


def chamar_llm_round(
    llm,
    objeto_noticia: str,
    candidatos: list[dict],
    round_name: str = "Round",
    max_retries: int = 1,
    debug: bool = False
) -> tuple[int, str, float, bool]:
    """
    Chama LLM para selecionar 1 candidato entre N.
    
    Returns:
        (indice_escolhido, resposta_bruta, tempo_segundos, falhou)
        - indice_escolhido: 0 se nenhum, 1-N se escolheu
        - falhou: True se timeout/erro
    """
    
    # Montar prompt
    candidatos_texto = "\n".join([
        f"  {i+1}. ID={c['id_processo']} | {c['objeto']}"
        for i, c in enumerate(candidatos)
    ])
    
    prompt = f"""Você é um especialista em linkage de registros de licitações públicas.

===============================================================================
TAREFA
===============================================================================

Identifique qual candidato corresponde EXATAMENTE ao objeto da notícia.

OBJETO DA NOTÍCIA:
{objeto_noticia}

CANDIDATOS ({len(candidatos)}):
{candidatos_texto}

===============================================================================
INSTRUÇÕES
===============================================================================

Retorne APENAS o número do candidato que corresponde EXATAMENTE ao objeto.

Se NENHUM candidato corresponder com certeza, retorne 0.

CRITÉRIOS:
- Mesmo objeto/serviço/produto
- Mesmo escopo
- Mesma finalidade
- Pequenas diferenças (quantidade, período, lote) INVALIDAM o match

NÃO assuma que os primeiros candidatos são melhores.
Objetos parecidos podem ser licitações DIFERENTES.

{"ATENÇÃO: Esta é a RODADA FINAL. Todos os candidatos já foram pré-selecionados." if "FINAL" in round_name else ""}

===============================================================================
SAÍDA
===============================================================================

Retorne APENAS UM número:
0  -> nenhum candidato corresponde
1-{len(candidatos)} -> índice do candidato correto

Sem explicações. Sem texto extra.
"""

    from langchain_core.messages import HumanMessage

    for tentativa in range(1, max_retries + 1):
        try:
            if debug:
                print(f"\n[Tentativa {tentativa}/{max_retries}]")

            def timeout_handler(signum, frame):
                raise TimeoutError("LLM timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(LLM_TIMEOUT_SECONDS)

            inicio = time.time()

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                resultado = response.content.strip()
            finally:
                signal.alarm(0)

            tempo = time.time() - inicio

            if debug:
                print(f"\n[LLM RESPOSTA] Tempo: {tempo:.2f}s")
                print(f"Resposta bruta: '{resultado}'")

            # Parse da resposta
            match = re.search(r'\b([0-9]+)\b', resultado)
            if not match:
                if tentativa < max_retries:
                    if debug:
                        print(f"[ERRO] Resposta inválida. Tentando novamente...")
                    continue
                else:
                    if debug:
                        print(f"[ERRO] Resposta inválida após {max_retries} tentativas.")
                    return (0, resultado, tempo, True)

            escolha = int(match.group(1))

            if escolha < 0 or escolha > len(candidatos):
                if tentativa < max_retries:
                    if debug:
                        print(f"[ERRO] Índice fora do range. Tentando novamente...")
                    continue
                else:
                    if debug:
                        print(f"[ERRO] Índice inválido após {max_retries} tentativas.")
                    return (0, resultado, tempo, True)

            return (escolha, resultado, tempo, False)

        except TimeoutError:
            if debug:
                print(f"[TIMEOUT] Tentativa {tentativa}/{max_retries}")
            if tentativa >= max_retries:
                return (0, "TIMEOUT", 0, True)
            time.sleep(2 ** tentativa)

        except Exception as e:
            if debug:
                print(f"[ERRO] {e}")
            if tentativa >= max_retries:
                return (0, f"ERRO: {e}", 0, True)
            time.sleep(2 ** tentativa)

    return (0, "MAX_RETRIES", 0, True)


def avaliar_tournament_top10(
    par: dict,
    llm,
    duplicatas_map: dict,
    identical_map: dict,
    verbose: bool = False
) -> dict:
    """
    Avalia uma notícia usando Tournament Selection em top-10.
    
    Processo:
    1. Pega top-10 do TF-IDF
    2. Embaralha para evitar viés
    3. Round 1: Grupo 1-5 → escolhe 1
    4. Round 2: Grupo 6-10 → escolhe 1
    5. Final: 2 finalistas → escolhe 1
    
    Returns:
        dict com métricas e detalhes
    """
    
    id_gold = str(par.get("id_gold", ""))
    objeto_noticia = par.get("objeto", "")
    candidatos_top50 = par.get("candidatos_top50", [])
    
    # Pegar apenas top-10
    candidatos_top10 = candidatos_top50[:TOP_N]
    
    if not candidatos_top10:
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": None,
            "acertou_llm": False,
            "id_escolhido_llm": None,
            "rank_llm": None,
            "falha_llm": False,
            "zero_llm": False,
            "tempo_total_s": 0,
            "chamadas_llm": 0
        }
    
    # Verificar se gold está no top-10
    rank_gold_top10 = None
    for i, cand in enumerate(candidatos_top10, 1):
        if str(cand["id_processo"]) == id_gold:
            rank_gold_top10 = i
            break
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PROCESSANDO NOTICIA {par.get('indice', '?')}/{par.get('total', '?')}")
        print(f"ID Gold: {id_gold}")
        print(f"Município: {par.get('municipio', 'N/A')}")
        print(f"Modalidade: {par.get('modalidade', 'N/A')}")
        if rank_gold_top10:
            print(f"Gold no rank TF-IDF: {rank_gold_top10}/{TOP_N}")
        else:
            print(f"Gold NAO encontrado no top-{TOP_N} TF-IDF")
        print(f"{'='*80}")
        print(f"\n{'='*80}")
        print(f"OBJETO DA NOTICIA:")
        print(objeto_noticia)
        print(f"\nCANDIDATOS ({len(candidatos_top10)}):")
        for i, c in enumerate(candidatos_top10, 1):
            print(f"  {i}. ID={c['id_processo']} | {c['objeto'][:100]}...")
    
    # Embaralhar para evitar viés
    indices_originais = list(range(len(candidatos_top10)))
    random.shuffle(indices_originais)
    candidatos_embaralhados = [candidatos_top10[i] for i in indices_originais]
    
    tempo_total = 0
    chamadas_llm = 0
    
    # Round 1: Primeiros 5
    grupo1 = candidatos_embaralhados[:GROUP_SIZE]
    escolha1, resp1, tempo1, falha1 = chamar_llm_round(
        llm, objeto_noticia, grupo1, "Round 1", MAX_RETRIES, verbose
    )
    tempo_total += tempo1
    chamadas_llm += 1
    
    if verbose:
        print(f"\n[ROUND 1] Escolha: {escolha1} | Tempo: {tempo1:.2f}s")
    
    if falha1:
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": rank_gold_top10,
            "acertou_llm": False,
            "id_escolhido_llm": None,
            "rank_llm": None,
            "falha_llm": True,
            "zero_llm": False,
            "tempo_total_s": tempo_total,
            "chamadas_llm": chamadas_llm
        }
    
    # Round 2: Últimos 5
    grupo2 = candidatos_embaralhados[GROUP_SIZE:]
    escolha2, resp2, tempo2, falha2 = chamar_llm_round(
        llm, objeto_noticia, grupo2, "Round 2", MAX_RETRIES, verbose
    )
    tempo_total += tempo2
    chamadas_llm += 1
    
    if verbose:
        print(f"\n[ROUND 2] Escolha: {escolha2} | Tempo: {tempo2:.2f}s")
    
    if falha2:
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": rank_gold_top10,
            "acertou_llm": False,
            "id_escolhido_llm": None,
            "rank_llm": None,
            "falha_llm": True,
            "zero_llm": False,
            "tempo_total_s": tempo_total,
            "chamadas_llm": chamadas_llm
        }
    
    # Montar finalistas
    finalistas = []
    if escolha1 > 0:
        finalistas.append(grupo1[escolha1 - 1])
    if escolha2 > 0:
        finalistas.append(grupo2[escolha2 - 1])
    
    if not finalistas:
        # Ambos os rounds disseram "0"
        if verbose:
            print(f"\n[RESULTADO] ZERO - Nenhum candidato selecionado nos rounds")
        
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": rank_gold_top10,
            "acertou_llm": False,
            "id_escolhido_llm": None,
            "rank_llm": None,
            "falha_llm": False,
            "zero_llm": True,
            "tempo_total_s": tempo_total,
            "chamadas_llm": chamadas_llm
        }
    
    if len(finalistas) == 1:
        # Apenas 1 finalista, não precisa de round final
        id_escolhido = str(finalistas[0]["id_processo"])
        
        if verbose:
            print(f"\n[FINALISTA ÚNICO] ID={id_escolhido}")
        
        acertou = avaliar_com_identical_objects(id_escolhido, id_gold, duplicatas_map, identical_map)
        
        if verbose:
            if acertou:
                print(f"\n[RESULTADO] ACERTO!")
            else:
                print(f"\n[RESULTADO] ERRO!")
                print(f"  ID escolhido: {id_escolhido}")
                print(f"  != ID gold: {id_gold}")
        
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": rank_gold_top10,
            "acertou_llm": acertou,
            "id_escolhido_llm": id_escolhido,
            "rank_llm": 1 if acertou else None,
            "falha_llm": False,
            "zero_llm": False,
            "tempo_total_s": tempo_total,
            "chamadas_llm": chamadas_llm
        }
    
    # Round Final: 2 finalistas
    escolha_final, resp_final, tempo_final, falha_final = chamar_llm_round(
        llm, objeto_noticia, finalistas, "FINAL", MAX_RETRIES, verbose
    )
    tempo_total += tempo_final
    chamadas_llm += 1
    
    if verbose:
        print(f"\n[ROUND FINAL] Escolha: {escolha_final} | Tempo: {tempo_final:.2f}s")
    
    if falha_final or escolha_final == 0:
        return {
            "id_gold": id_gold,
            "rank_tfidf_top10": rank_gold_top10,
            "acertou_llm": False,
            "id_escolhido_llm": None,
            "rank_llm": None,
            "falha_llm": falha_final,
            "zero_llm": (escolha_final == 0),
            "tempo_total_s": tempo_total,
            "chamadas_llm": chamadas_llm
        }
    
    # Resultado final
    id_escolhido = str(finalistas[escolha_final - 1]["id_processo"])
    acertou = avaliar_com_identical_objects(id_escolhido, id_gold, duplicatas_map, identical_map)
    
    if verbose:
        print(f"\n[SELECIONADO] ID={id_escolhido}")
        if acertou:
            print(f"\n[RESULTADO] ACERTO!")
        else:
            print(f"\n[RESULTADO] ERRO!")
            print(f"  ID escolhido: {id_escolhido}")
            print(f"  != ID gold: {id_gold}")
    
    return {
        "id_gold": id_gold,
        "rank_tfidf_top10": rank_gold_top10,
        "acertou_llm": acertou,
        "id_escolhido_llm": id_escolhido,
        "rank_llm": 1 if acertou else None,
        "falha_llm": False,
        "zero_llm": False,
        "tempo_total_s": tempo_total,
        "chamadas_llm": chamadas_llm
    }


def main():
    parser = argparse.ArgumentParser(description="Tournament Top-10 Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de notícias")
    parser.add_argument("--verbose", action="store_true", help="Modo verboso")
    parser.add_argument("--checkpoint", type=str, default=None, help="Arquivo de checkpoint customizado")
    parser.add_argument("--no-checkpoint", action="store_true", help="Desativa salvamento de checkpoint")
    args = parser.parse_args()

    print("="*80)
    print("TOURNAMENT TOP-10 PIPELINE")
    print("="*80)
    print(f"\nModelo LLM: {RERANK_MODEL}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Top-N: {TOP_N}")
    print(f"Group Size: {GROUP_SIZE}")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Timeout: {LLM_TIMEOUT_SECONDS}s")
    print(f"Seed: {SEED}")

    # Carregar dados
    print(f"\nCarregando cache: {CACHE_PATH}")
    cache = carregar_cache_tfidf(args.limit)
    print(f"Cache carregado: {len(cache)} notícias")

    print(f"\nCarregando duplicatas: {DUPLICATAS_PATH}")
    duplicatas_map = carregar_duplicatas()
    if duplicatas_map:
        total_grupos = duplicatas_map.get("total_grupos", 0)
        total_processos = duplicatas_map.get("total_processos_duplicados", 0)
        print(f"Duplicatas carregadas: {total_grupos} grupos, {total_processos} processos")
    else:
        print("Nenhuma duplicata carregada")

    print(f"\nCarregando objetos idênticos: {IDENTICAL_OBJECTS_PATH}")
    identical_map = carregar_identical_objects()
    if identical_map:
        total_grupos_id = len(identical_map)
        total_processos_id = sum(len(v) for v in identical_map.values())
        print(f"Objetos idênticos carregados: {total_grupos_id} grupos, {total_processos_id} processos")
    else:
        print("Nenhum objeto idêntico carregado")

    # Definir checkpoint
    if args.no_checkpoint:
        checkpoint_path = None
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        n_suffix = f"n{len(cache)}" if args.limit else "nall"
        model_suffix = RERANK_MODEL.replace(":", "_")
        checkpoint_path = BASE_DIR / f"results/checkpoint_tournament_top10_{n_suffix}_{model_suffix}.json"
    
    # Carregar checkpoint se existir
    proximo_idx = 0
    resultados = []
    tempo_acumulado = 0
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n⚠️  CHECKPOINT ENCONTRADO: {checkpoint_path}")
        checkpoint = carregar_checkpoint(checkpoint_path)
        if checkpoint:
            proximo_idx = checkpoint.get("proximo_idx", 0)
            resultados = checkpoint.get("detalhes", [])
            tempo_acumulado = checkpoint.get("tempo_acumulado", 0)
            print(f"Retomando do índice {proximo_idx}/{len(cache)}")
            print(f"Já processadas: {len(resultados)} notícias")
    
    if checkpoint_path:
        print(f"\nCheckpoint: {checkpoint_path}")
    else:
        print(f"\n⚠️  Checkpoint DESATIVADO")

    # Inicializar LLM
    print(f"\nInicializando LLM: {RERANK_MODEL}")
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=RERANK_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0
    )

    # Processar notícias
    print(f"\n{'='*80}")
    print("PROCESSANDO NOTÍCIAS")
    print(f"{'='*80}\n")

    tempo_inicio = time.time()

    for i in tqdm(range(proximo_idx, len(cache)), desc="Processando", initial=proximo_idx, total=len(cache)):
        par = cache[i]
        par["indice"] = i
        par["total"] = len(cache)
        
        resultado = avaliar_tournament_top10(
            par, llm, duplicatas_map, identical_map, args.verbose
        )
        resultado["indice_par"] = str(i)
        resultado["id_processo_gold"] = resultado.pop("id_gold")
        resultado["rank_tfidf_topn"] = resultado.pop("rank_tfidf_top10")
        
        resultados.append(resultado)
        
        # Salvar checkpoint
        if checkpoint_path:
            salvar_checkpoint(checkpoint_path, {
                "proximo_idx": i + 1,
                "detalhes": resultados,
                "tempo_acumulado": tempo_acumulado + (time.time() - tempo_inicio)
            })

    tempo_total = tempo_acumulado + (time.time() - tempo_inicio)

    # Calcular métricas
    total = len(resultados)
    com_candidatos = sum(1 for r in resultados if r["rank_tfidf_topn"] is not None or r["id_escolhido_llm"] is not None)
    sem_candidatos = total - com_candidatos
    
    acertos_llm = sum(1 for r in resultados if r["acertou_llm"])
    falhas_llm = sum(1 for r in resultados if r["falha_llm"])
    zeros_llm = sum(1 for r in resultados if r["zero_llm"])
    
    # TF-IDF metrics
    gold_no_top10 = sum(1 for r in resultados if r["rank_tfidf_topn"] is not None)
    match_rate_top10 = gold_no_top10 / total if total > 0 else 0
    
    tfidf_rank1 = sum(1 for r in resultados if r["rank_tfidf_topn"] == 1)
    precision_at_1 = tfidf_rank1 / total if total > 0 else 0
    
    # MRR TF-IDF
    mrr_tfidf = 0
    for r in resultados:
        if r["rank_tfidf_topn"]:
            mrr_tfidf += 1.0 / r["rank_tfidf_topn"]
    mrr_tfidf = mrr_tfidf / total if total > 0 else 0
    
    # LLM metrics
    accuracy_llm = acertos_llm / com_candidatos if com_candidatos > 0 else 0
    mrr_llm = acertos_llm / com_candidatos if com_candidatos > 0 else 0  # MRR = 1.0 quando acerta
    
    # Chamadas LLM
    total_chamadas = sum(r["chamadas_llm"] for r in resultados)
    
    # Salvar resultados
    output_file = BASE_DIR / f"results/tournament_top10_n{total}_{RERANK_MODEL.replace(':', '_')}.json"
    output_data = {
        "configuracao": {
            "modelo_llm": RERANK_MODEL,
            "ollama_host": OLLAMA_HOST,
            "top_n": TOP_N,
            "group_size": GROUP_SIZE,
            "max_retries": MAX_RETRIES,
            "llm_timeout_s": LLM_TIMEOUT_SECONDS,
            "seed": SEED
        },
        "metricas_tfidf": {
            "match_rate_top10": round(match_rate_top10, 4),
            "precision_at_1": round(precision_at_1, 4),
            "mrr": round(mrr_tfidf, 4)
        },
        "metricas_llm": {
            "accuracy": round(accuracy_llm, 4),
            "mrr": round(mrr_llm, 4)
        },
        "estatisticas": {
            "total": total,
            "top_n": TOP_N,
            "sem_candidatos_sql": sem_candidatos,
            "com_candidatos": com_candidatos,
            "falhas_llm": falhas_llm,
            "zeros_llm": zeros_llm
        },
        "custo": {
            "chamadas_llm": total_chamadas,
            "tempo_total_s": tempo_total
        },
        "detalhes": resultados
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(" RESULTADOS FINAIS")
    print(f"{'='*80}\n")
    
    print("ESTATÍSTICAS:")
    print(f"  total: {total}")
    print(f"  top_n: {TOP_N}")
    print(f"  sem_candidatos_sql: {sem_candidatos}")
    print(f"  com_candidatos: {com_candidatos}")
    print(f"  falhas_llm: {falhas_llm}")
    print(f"  zeros_llm: {zeros_llm}")
    
    print(f"\nTF-IDF (top-{TOP_N}):")
    print(f"  match_rate_top{TOP_N}: {match_rate_top10:.4f}")
    print(f"  precision_at_1: {precision_at_1:.4f}")
    print(f"  mrr: {mrr_tfidf:.4f}")
    
    print(f"\nLLM TOURNAMENT TOP-10 ({RERANK_MODEL}):")
    print(f"  accuracy: {accuracy_llm:.4f}")
    print(f"  mrr: {mrr_llm:.4f}")
    
    print(f"\nCOMPARAÇÃO:")
    print(f"  ganho accuracy vs TF-IDF@1: {accuracy_llm - precision_at_1:+.4f}")
    
    print(f"\nCUSTO:")
    print(f"  chamadas_llm: {total_chamadas}")
    print(f"  média chamadas/notícia: {total_chamadas/total:.1f}")
    print(f"  tempo_total_min: {tempo_total/60:.1f}")
    print(f"  tempo_médio/notícia: {tempo_total/total:.1f}s")
    
    print(f"\nResultados salvos em: {output_file}")


if __name__ == "__main__":
    main()
