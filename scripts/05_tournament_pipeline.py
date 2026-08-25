
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_tournament_pipeline.py
=========================
Pipeline Tournament Selection usando CACHE TF-IDF pré-calculado.

OBJETIVO:
- Carregar top-50 TF-IDF do cache
- Aplicar reranking via LLM Tournament Selection
- Avaliar Recall@1 final
- Detectar explicitamente falhas do LLM

CONFIGURAÇÃO VENCEDORA:
- Tournament 10×5 (10 grupos de 5 candidatos)
- LLM: qwen2.5:7b
- Accuracy: 70.2% (vs 66.6% TF-IDF baseline)

Uso:
    python scripts/05_tournament_pipeline.py --limit 10
    python scripts/05_tournament_pipeline.py  # todas as 500 notícias
    RERANK_MODEL=qwen2.5:14b python scripts/05_tournament_pipeline.py
"""

import os
import json
import argparse
import pickle
import time
import signal
import re
import random
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import avaliar_com_duplicatas, avaliar_com_identical_objects

BASE_DIR = Path(__file__).parent.parent

# Caminhos
CACHE_PATH = BASE_DIR / "models/cache_tfidf_top50.pkl"
DUPLICATAS_PATH = BASE_DIR / "data/duplicatas_mapping.json"
IDENTICAL_OBJECTS_PATH = BASE_DIR / "data/identical_objects_mapping.json"

# LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-oss:20b")

# Configuração Tournament
TOURNAMENT_SIZE = 5  # 10×5: 10 grupos de 5 candidatos
SELECTIONS_PER_ROUND = 1  # 1 vencedor por grupo

MAX_RETRIES = 1  # number of attempts per round; 1 = single attempt with no retry; 0 = skip all attempts (always fails)
LLM_TIMEOUT_SECONDS = 300

# Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

signal.signal(signal.SIGHUP, signal.SIG_IGN)


def carregar_cache_tfidf(limit: int = None) -> list[dict]:
    """Carrega cache TF-IDF."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache não encontrado: {CACHE_PATH}\n"
            f"Execute primeiro: python scripts/03_generate_cache.py"
        )

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    if limit:
        cache = cache[:limit]

    return cache


def carregar_duplicatas() -> dict:
    """Carrega mapeamento de duplicatas."""
    if not DUPLICATAS_PATH.exists():
        print("Arquivo de duplicatas não encontrado.")
        return {}

    with open(DUPLICATAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("id_to_group", {})


def tournament_round(
    objeto_noticia: str,
    candidatos: list[dict],
    llm,
    n_select: int = 1,
    max_retries: int = MAX_RETRIES,
    debug: bool = False,
    verbose: bool = False,
    round_name: str = "",
):
    """
    Executa uma rodada do tournament reranking.
    
    Returns:
        list[int] -> índices selecionados
        None -> falha definitiva do LLM
    """
    if not candidatos:
        return None

    if len(candidatos) <= n_select:
        return list(range(len(candidatos)))

    if verbose:
        print(f"\n{'='*80}")
        print(f"RODADA: {round_name}")
        print(f"{'='*80}")
        print(f"\nOBJETO DA NOTICIA:")
        print(f"{objeto_noticia[:200]}..." if len(objeto_noticia) > 200 else objeto_noticia)
        print(f"\nCANDIDATOS ({len(candidatos)}):")

    candidatos_text = ""
    for i, cand in enumerate(candidatos, 1):
        obj_text = str(cand.get("objeto", "")).strip()
        candidatos_text += f"{i}. {obj_text}\n\n"
        if verbose:
            print(f"  {i}. ID={cand.get('id_processo', 'N/A')} | {obj_text[:100]}..." if len(obj_text) > 100 else f"  {i}. ID={cand.get('id_processo', 'N/A')} | {obj_text}")

    prompt = f"""
Você é um sistema de reranking semântico de alta precisão.

Sua tarefa é identificar se EXISTE exatamente 1 candidato que corresponde ao MESMO objeto da notícia.

Não escolha por similaridade geral. Escolha apenas por correspondência específica do objeto contratado.

===============================================================================
OBJETO DA NOTÍCIA
===============================================================================
{objeto_noticia}

===============================================================================
CANDIDATOS ({len(candidatos)})
===============================================================================
{candidatos_text}

===============================================================================
REGRAS DE DECISÃO
===============================================================================

Compare o objeto da notícia com CADA candidato.

Considere correspondência apenas se o núcleo contratual for o mesmo.

Validar principalmente:

1. Objeto principal contratado
   - produto, serviço ou contratação central

2. Finalidade
   - para que foi contratado

3. Tipo contratual
   - empenho, registro de preços, compra direta, locação, manutenção etc.

4. Beneficiário / entidade / alvo exato
   - nome, unidade, pessoa, órgão, setor, local, programa

5. Escopo
   - item específico vs categoria ampla
   - unidade específica vs todas unidades

6. Identificadores fortes
   - nome
   - número
   - lote
   - fase
   - etapa
   - quantidade
   - período
   - referência temporal
   - código

7. Restrições textuais
   Palavras que restringem ou ampliam o objeto mudam a decisão.

===============================================================================
NÃO CONSIDERE COMO MATCH APENAS POR:
===============================================================================

- mesmo tema
- mesmo contexto
- mesmo órgão
- mesmo município
- mesma família textual
- prefixos repetidos
- boilerplate administrativo
- estrutura textual parecida
- palavras parcialmente semelhantes

Contexto parecido NÃO implica mesmo objeto.

===============================================================================
REGRAS CRÍTICAS
===============================================================================

1. Analise TODOS os candidatos antes de decidir.
   Não pare no primeiro aparentemente bom.

2. Não faça early lock.
   Compare os melhores candidatos entre si antes da resposta final.

3. Ignore texto administrativo repetido.
   Priorize os discriminadores reais do objeto.

4. Pequenas diferenças invalidam match:
   - nome diferente
   - beneficiário diferente
   - quantidade diferente
   - período diferente
   - escopo diferente
   - lote/fase diferente
   - item contratado diferente
   - restrição textual diferente

5. Falso negativo é melhor que falso positivo.
   Se houver dúvida real, responda 0.

6. Se múltiplos candidatos forem semanticamente próximos,
   escolha apenas o que for IDENTICAMENTE o mesmo objeto.

{"" if "FINAL" not in round_name else """
===============================================================================
ATENÇÃO ESPECIAL — RODADA FINAL
===============================================================================

Todos os candidatos já são semanticamente próximos.

Não escolha por proximidade geral.

Decida usando microdiscriminadores:

- beneficiário exato
- nome exato
- quantidade/cardinalidade
- período/mês/referência temporal
- lote/fase/etapa
- escopo
- restrições adicionais
- item contratual exato

Ignore prefixos repetidos.

Se dois candidatos forem parecidos, compare o trecho realmente distintivo.

Exemplo conceitual:
Mesmo texto base + mês diferente = pode ser outro registro.
Mesmo texto base + quantidade diferente = pode ser outro objeto.
Mesmo texto base + beneficiário diferente = NÃO é match.
"""}

===============================================================================
SAÍDA
===============================================================================

Retorne APENAS UM valor:

0  -> nenhum candidato corresponde com segurança
1-{len(candidatos)} -> índice do único candidato correto

Sem explicações.
Sem texto extra.
Sem ranking.
Sem múltiplos índices.
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

            if verbose:
                print(f"\n[LLM RESPOSTA] Tempo: {tempo:.2f}s")
                print(f"Resposta bruta: '{resultado}'")
            elif debug:
                print(f"Tempo: {tempo:.2f}s | Resposta: {resultado}")

            numeros = re.findall(r'\b(\d+)\b', resultado)
            indices = []

            for num in numeros:
                num_int = int(num)

                if num_int == 0:
                    if debug:
                        print("LLM indicou: nenhum candidato corresponde (0)")
                    return None

                if 1 <= num_int <= len(candidatos):
                    idx = num_int - 1
                    if idx not in indices:
                        indices.append(idx)

            if len(indices) >= n_select:
                if verbose:
                    print(f"\n[SELECIONADOS] Indices: {indices[:n_select]}")
                    for idx in indices[:n_select]:
                        print(f"  -> Candidato {idx+1}: ID={candidatos[idx].get('id_processo', 'N/A')}")
                return indices[:n_select]

            if indices:
                if verbose or debug:
                    print(f"Selecoes insuficientes: LLM retornou {len(indices)} de {n_select} esperados ('{resultado}') - aceitando parcialmente")
                return indices

            if verbose or debug:
                print(f"Resposta inválida: nenhum número válido em '{resultado}' (esperava 1-{len(candidatos)})") 

        except TimeoutError:
            if verbose:
                print(f"\n[ERRO] TIMEOUT (tentativa {tentativa}/{max_retries})")
            else:
                print(f"TIMEOUT (tentativa {tentativa}/{max_retries})")
        except Exception as e:
            if verbose:
                print(f"\n[ERRO] {type(e).__name__}: {str(e)}")
            else:
                print(f"ERRO: {type(e).__name__}: {str(e)}")

        time.sleep(tentativa * 2)

    if verbose:
        print(f"\n[FALHA] FALHA DEFINITIVA DO RERANKING após {max_retries} tentativas")
    else:
        print("FALHA DEFINITIVA DO RERANKING")
    return None


def tournament_selection_10x5(objeto_noticia: str, candidatos_top50: list[dict], llm, verbose: bool = False, noticia_idx: int = 0):
    """
    Tournament Selection 10×5:
    - 10 grupos de 5 candidatos
    - 1 vencedor por grupo
    - Final com 10 candidatos
    """
    if not candidatos_top50:
        return None, 0

    if len(candidatos_top50) == 1:
        return 0, 0

    # Embaralhar para evitar viés
    indices_originais = list(range(len(candidatos_top50)))
    random.shuffle(indices_originais)

    candidatos_embaralhados = [candidatos_top50[i] for i in indices_originais]

    # Dividir em grupos
    grupos = []
    for i in range(0, min(50, len(candidatos_embaralhados)), TOURNAMENT_SIZE):
        grupo = candidatos_embaralhados[i:i + TOURNAMENT_SIZE]
        grupos.append(grupo)

    finalistas_locais = []
    custo = 0

    # Rodadas eliminatórias
    if verbose:
        print(f"\n{'#'*80}")
        print(f"NOTICIA {noticia_idx+1} - FASE ELIMINATORIA")
        print(f"Total de grupos: {len(grupos)}")
        print(f"{'#'*80}")

    for grupo_idx, grupo in enumerate(grupos, 1):
        round_name = f"Noticia {noticia_idx+1} - Grupo {grupo_idx}/{len(grupos)}"
        indices_selecionados = tournament_round(
            objeto_noticia, grupo, llm, n_select=SELECTIONS_PER_ROUND,
            verbose=verbose, round_name=round_name
        )

        custo += 1

        if indices_selecionados is None:
            continue

        for idx in indices_selecionados:
            finalistas_locais.append(grupo[idx])

    if not finalistas_locais:
        if verbose:
            print(f"\n[AVISO] Nenhum finalista selecionado")
        return None, custo

    # Final
    if verbose:
        print(f"\n{'#'*80}")
        print(f"NOTICIA {noticia_idx+1} - RODADA FINAL")
        print(f"Finalistas: {len(finalistas_locais)}")
        print(f"{'#'*80}")

    round_name = f"Noticia {noticia_idx+1} - FINAL"
    indices_final = tournament_round(objeto_noticia, finalistas_locais, llm, n_select=1,
                                    verbose=verbose, round_name=round_name)

    custo += 1

    if indices_final is None:
        return None, custo

    vencedor_local = finalistas_locais[indices_final[0]]

    # Mapear índice global
    idx_global = None
    for i, cand in enumerate(candidatos_top50):
        if cand['id_processo'] == vencedor_local['id_processo']:
            idx_global = i
            break

    if verbose:
        print(f"\n[VENCEDOR FINAL] ID={vencedor_local.get('id_processo', 'N/A')}")
        print(f"Posicao no ranking TF-IDF: {idx_global+1 if idx_global is not None else 'N/A'}")

    return idx_global, custo


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


def avaliar_tournament(cache_tfidf: list[dict], llm, duplicatas_map: dict, identical_map: dict, verbose: bool = False, checkpoint_path: Path = None):
    """Avalia Tournament Selection."""

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
        "sem_candidatos_sql": 0,
        "com_candidatos": 0,
        "tournament_usado": 0,
        "falhas_llm": 0,
    }

    metricas_tfidf = checkpoint["metricas_tfidf"] if checkpoint else {
        "acertos_top1": 0,
        "encontrados_top50": 0,
        "ranks": [],
    }

    metricas_tournament = checkpoint["metricas_tournament"] if checkpoint else {
        "acertos": 0,
        "ranks": [],
    }

    custo_total = checkpoint["custo_total"] if checkpoint else 0
    detalhes = checkpoint["detalhes"] if checkpoint else []

    cache_restante = cache_tfidf[inicio_idx:]

    use_tqdm = not verbose
    iterator = tqdm(cache_restante, desc="Tournament", unit="notícia", disable=not use_tqdm, initial=inicio_idx, total=len(cache_tfidf)) if use_tqdm else cache_restante

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
            continue

        stats["com_candidatos"] += 1

        # TF-IDF
        rank_tfidf = None
        for rank, cand in enumerate(candidatos, 1):
            if avaliar_com_duplicatas(cand['id_processo'], id_gold, duplicatas_map):
                rank_tfidf = rank
                metricas_tfidf["encontrados_top50"] += 1
                metricas_tfidf["ranks"].append(rank)
                if rank == 1:
                    metricas_tfidf["acertos_top1"] += 1
                break

        # Tournament
        if verbose:
            print(f"\n\n{'='*80}")
            print(f"PROCESSANDO NOTICIA {noticia_idx+1}/{len(cache_tfidf)}")
            print(f"ID Gold: {id_gold}")
            print(f"Municipio: {item.get('municipio', 'N/A')}")
            print(f"Modalidade: {item.get('modalidade', 'N/A')}")
            print(f"{'='*80}")

        idx_vencedor, custo = tournament_selection_10x5(objeto, candidatos, llm, 
                                                        verbose=verbose, noticia_idx=noticia_idx)

        custo_total += custo
        stats["tournament_usado"] += 1

        rank_tournament = None
        acertou_tournament = False
        id_escolhido = None

        if idx_vencedor is None:
            stats["falhas_llm"] += 1
            if verbose:
                print(f"\n[RESULTADO] FALHA - LLM nao conseguiu selecionar")
        else:
            id_escolhido = candidatos[idx_vencedor]['id_processo']
            objeto_escolhido = candidatos[idx_vencedor].get('objeto', '')[:150]
            acertou_tournament = avaliar_com_identical_objects(id_escolhido, id_gold, duplicatas_map, identical_map)

            # Buscar objeto do gold
            objeto_gold = None
            for cand in candidatos:
                if avaliar_com_identical_objects(cand['id_processo'], id_gold, duplicatas_map, identical_map):
                    objeto_gold = cand.get('objeto', '')[:150]
                    break

            if acertou_tournament:
                metricas_tournament["acertos"] += 1
                rank_tournament = 1
                metricas_tournament["ranks"].append(1)
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
            "rank_tfidf": rank_tfidf,
            "rank_tournament": rank_tournament,
            "acertou_tournament": acertou_tournament,
            "id_escolhido_tournament": id_escolhido,
            "falha_llm": idx_vencedor is None,
        })

        # Salvar checkpoint após cada notícia
        if checkpoint_path:
            salvar_checkpoint(checkpoint_path, {
                "proximo_idx": noticia_idx + 1,
                "stats": stats,
                "metricas_tfidf": metricas_tfidf,
                "metricas_tournament": metricas_tournament,
                "custo_total": custo_total,
                "detalhes": detalhes,
            })

    # Métricas finais
    n_tfidf = stats["com_candidatos"]
    n_tournament = stats["tournament_usado"]

    metricas_tfidf_final = {
        "match_rate": metricas_tfidf["encontrados_top50"] / n_tfidf if n_tfidf > 0 else 0,
        "precision_at_1": metricas_tfidf["acertos_top1"] / n_tfidf if n_tfidf > 0 else 0,
        "mrr": np.mean([1 / r for r in metricas_tfidf["ranks"]]) if metricas_tfidf["ranks"] else 0,
    }

    metricas_tournament_final = {
        "accuracy": metricas_tournament["acertos"] / n_tournament if n_tournament > 0 else 0,
        "mrr": np.mean([1 / r for r in metricas_tournament["ranks"]]) if metricas_tournament["ranks"] else 0,
    }

    return (metricas_tfidf_final, metricas_tournament_final, stats, custo_total, detalhes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Modo verbose: mostra detalhes de cada rodada do tournament")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Arquivo de checkpoint para retomar execucao interrompida (ex: results/checkpoint.json)")
    parser.add_argument("--no-checkpoint", action="store_true",
                       help="Desativa salvamento de checkpoint")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" TOURNAMENT SELECTION (CACHE TF-IDF)")
    print("=" * 80)

    print(f"\nModelo LLM: {RERANK_MODEL}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Seed: {SEED}")
    if args.verbose:
        print(f"Modo: VERBOSE (detalhes completos de cada rodada)")
    sys.stdout.flush()

    # Carregar cache
    print("\nCarregando cache TF-IDF...")
    cache_tfidf = carregar_cache_tfidf(args.limit)
    duplicatas_map = carregar_duplicatas()

    identical_map = {}
    if IDENTICAL_OBJECTS_PATH.exists():
        with open(IDENTICAL_OBJECTS_PATH, "r", encoding="utf-8") as f:
            identical_map = json.load(f)

    print(f"{len(cache_tfidf)} notícias carregadas")
    print(f"{len(duplicatas_map):,} duplicatas mapeadas")
    print(f"{len(identical_map):,} grupos de objetos idênticos mapeados")

    # Inicializar LLM
    print(f"\nInicializando LLM ({RERANK_MODEL})...")

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=RERANK_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
        timeout=300
    )

    # Definir path do checkpoint
    checkpoint_path = None
    if not args.no_checkpoint:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            n = args.limit if args.limit else "all"
            checkpoint_path = BASE_DIR / f"results/checkpoint_{n}_{RERANK_MODEL.replace(':', '_')}.json"
        checkpoint_path.parent.mkdir(exist_ok=True)
        print(f"Checkpoint: {checkpoint_path}")
        sys.stdout.flush()

    # Executar
    tempo_inicio = time.time()

    (metricas_tfidf, metricas_tournament, stats, custo, detalhes) = avaliar_tournament(
        cache_tfidf, llm, duplicatas_map, identical_map, verbose=args.verbose, checkpoint_path=checkpoint_path
    )

    tempo_total = time.time() - tempo_inicio

    # Resultados
    print("\n" + "=" * 80)
    print(" RESULTADOS FINAIS")
    print("=" * 80)

    print("\nESTATÍSTICAS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nTF-IDF:")
    for k, v in metricas_tfidf.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nTOURNAMENT:")
    for k, v in metricas_tournament.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    ganho = metricas_tournament["accuracy"] - metricas_tfidf["precision_at_1"]

    print("\nCOMPARAÇÃO:")
    print(f"  ganho accuracy: {ganho:+.4f}")

    print("\nCUSTO:")
    print(f"  chamadas_llm: {custo}")
    print(f"  tempo_total_min: {tempo_total/60:.1f}")

    # Salvar JSON
    output_path = BASE_DIR / f"results/tournament_n{len(cache_tfidf)}_{RERANK_MODEL.replace(':', '_')}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "configuracao": {
                "modelo_llm": RERANK_MODEL,
                "ollama_host": OLLAMA_HOST,
                "seed": SEED,
                "tournament_size": TOURNAMENT_SIZE,
                "selections_per_round": SELECTIONS_PER_ROUND,
            },
            "metricas_tfidf": metricas_tfidf,
            "metricas_tournament": metricas_tournament,
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
