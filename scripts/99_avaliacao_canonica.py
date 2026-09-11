#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
99_avaliacao_canonica.py
=========================
Avaliação canônica (identical-aware) de todos os artefatos existentes para a
dissertação (C2 do paper/EQM/tracking_experimentos_dissertacao.md).

Princípios:
  - Denominador SEMPRE o total de notícias (dev=500, teste=214);
    abstenção ("zero") e falha contam como erro.
  - Matching identical-aware (gold OU duplicata OU objeto idêntico) em tudo.
  - LLM (single-call e tournament): reporta apenas Top-1 accuracy.
    MRR NÃO é reportado p/ LLM: a saída é um único candidato (rank_llm 1 ou None)
    => o "MRR" dos JSONs é artefato de implementação, não métrica real.
  - Recuperação TF-IDF: Recall@k / p@1 / MRR / perdidos, strict e equivalent-aware.

Saídas:
  results/avaliacao_canonica.json  (detalhado)
  results/avaliacao_canonica.csv   (tabelas planas p/ a dissertação)

Sem nenhuma chamada de LLM.
"""
import json
import pickle
import csv
import collections


ROOT_VALUES = {
    "dups": "data/duplicatas_mapping.json",
    "ident": "data/identical_objects_mapping.json",
    "cache_dev": "models/cache_tfidf_top50.pkl",
    "cache_test": "models/test/cache_tfidf_top50_test.pkl",
    "resultados": [
        "results/single_call_top10_n500_gpt-oss_20b.json",
        "results/single_call_top10_n500_qwen2.5_7b.json",
        "results/single_call_top10_n500_qwen3.5_9b.json",
        "results/single_call_top10_n500_qwen3.8_27b.json",
        "results/single_call_top10_n500_gemma4_31b.json",
        "results/single_call_top10_n214_test_gpt-oss_20b.json",
        "results/single_call_top10_n214_test_qwen3.5_9b.json",
        "results/single_call_top10_n214_test_qwen3.8_27b.json",
        "results/single_call_top10_n214_test_gemma4_31b.json",
        "results/tournament_top10_n500_gpt-oss_20b.json",
        "results/tournament_n500_gpt-oss_20b.json",
        "results/tournament_top10_n500_qwen2.5_7b.json",
        "results/tournament_top10_n500_qwen3.8_27b.json",
        "results/tournament_top10_n500_gemma4_31b.json",
    ],
}


# ---------------------------------------------------------------------------
# Indexação dos mapas de equivalência
# ---------------------------------------------------------------------------
def carregar_mapas(): # noqa
    dups_raw = json.load(open(ROOT_VALUES["dups"]))
    ident_raw = json.load(open(ROOT_VALUES["ident"]))

    # duplicatas: id -> grupo (lista de ids)
    id_to_group = {str(k): [str(x) for x in v] for k, v in dups_raw.get("id_to_group", {}).items()}

    # objetos idênticos: chave (objeto normalizado) -> lista de {id,...}
    ids_by_key = {}
    obj_groups = collections.defaultdict(set)  # id -> chaves onde aparece
    for key, records in ident_raw.items():
        ids = [str(r["id"]) for r in records]
        ids_by_key[key] = set(ids)
        for i in ids:
            obj_groups[i].add(key)

    def equivalente(a, b):
        a, b = str(a), str(b)
        if a == b:
            return True
        grupo = id_to_group.get(a)
        if grupo and b in grupo:
            return True
        for key in obj_groups.get(a, ()):
            if b in ids_by_key[key]:
                return True
        return False

    return equivalente


# ---------------------------------------------------------------------------
# Recuperação TF-IDF
# ---------------------------------------------------------------------------
def metricas_recuperacao(cache, equivalente):
    """Retorna (strict, equiv) com Recall@k, p@1, MRR, perdidos, rank medio."""
    import statistics

    def calc(cands, gold):
        def pos(confronta):
            for i, c in enumerate(cands, 1):
                if confronta(c["id_processo"]):
                    return i
            return None
        rank_strict = pos(lambda cid: str(cid) == str(gold))
        rank_equiv = pos(lambda cid: equivalente(cid, gold))
        return rank_strict, rank_equiv

    rs, re = [], []
    for item in cache:
        gold = item["id_gold"]
        s, e = calc(item["candidatos_top50"], gold)
        rs.append(s)
        re.append(e)

    def fold(ranks, K):
        n = len(ranks)
        found = [r for r in ranks if r is not None]
        recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 5, 10, 20, 50)}
        p1 = recall[1]
        mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
        mean_rank = statistics.mean(found) if found else None
        return {
            "n_presentes": len(found),
            "perdidos_frac": (n - len(found)) / n,
            "recall_at_1": recall[1],
            "recall_at_5": recall[5],
            "recall_at_10": recall[10],
            "recall_at_20": recall[20],
            "recall_at_50": recall[50],
            "p_at_1": p1,
            "mrr": mrr,
            "rank_medio": mean_rank,
        }

    return fold(rs, 50), fold(re, 50)


def slice_top10(cache, equivalente):
    """p@1 e match@10 sobre a fatia candidatos[:10] (o que o LLM viu)."""
    n = len(cache)
    p1 = 0
    match10 = 0
    for item in cache:
        gold = item["id_gold"]
        top10 = item["candidatos_top50"][:10]
        if top10:
            for i, c in enumerate(top10, 1):
                if equivalente(c["id_processo"], gold):
                    if i == 1:
                        p1 += 1
                    match10 += 1
                    break
    return {"p_at_1_top10_equiv": p1 / n if n else 0.0, "match_at_10_top10_equiv": match10 / n if n else 0.0}


# ---------------------------------------------------------------------------
# Resultado LLM
# ---------------------------------------------------------------------------
def metricas_llm(caminho, equivalente, total_esperado, top10):
    d = json.load(open(caminho))
    detalhes = d.get("detalhes", [])
    n = len(detalhes)
    acertos = 0
    zeros = 0
    falhas = 0
    escolhidos = 0
    for r in detalhes:
        if r.get("zero_llm"):
            zeros += 1
        elif r.get("falha_llm"):
            falhas += 1
        elif r.get("id_escolhido_llm") is not None:
            escolhidos += 1
            if equivalente(r["id_escolhido_llm"], r["id_processo_gold"]):
                acertos += 1
    stats = d.get("estatisticas", {})
    return {
        "arquivo": caminho,
        "n_itens": n,
        "zerados": zeros,
        "falhas": falhas,
        "escolheu": escolhidos,
        "accuracy_sobre_total": acertos / n if n else 0.0,
        "accuracy_detalhe_candidatos": acertos / stats["com_candidatos"] if stats.get("com_candidatos") else 0.0,
        "abstencao_rate": zeros / n if n else 0.0,
        "falha_rate": falhas / n if n else 0.0,
        "chamadas_llm_total": d.get("custo", {}).get("chamadas_llm", d.get("custo_total")),
        "ganho_vs_tfidf_p1_top10_equiv": acertos / n - top10["p_at_1_top10_equiv"] if n else 0.0,
        "mrr_armazenado_artefato": d.get("metricas_llm", {}).get("mrr"),
    }


def main():
    equivalente = carregar_mapas()

    saida = {"metodologia": "identical-aware; denominador = total de notícias", "recuperacao": {}, "llm": {}}

    for nome, caminho in (("dev_500", "models/cache_tfidf_top50.pkl"), ("test_214", "models/test/cache_tfidf_top50_test.pkl")):
        cache = pickle.load(open(caminho, "rb"))
        strict, equiv = metricas_recuperacao(cache, equivalente)
        saida["recuperacao"][nome] = {"strict": strict, "equiv": equiv, "slice_top10": slice_top10(cache, equivalente)}

    dev_top10 = saida["recuperacao"]["dev_500"]["slice_top10"]
    test_top10 = saida["recuperacao"]["test_214"]["slice_top10"]
    top10_por_arquivo = {
        "flags/single_call_top10_n500_gpt-oss_20b.json": dev_top10,
        "flags/single_call_top10_n500_qwen2.5_7b.json": dev_top10,
        "flags/tournament_top10_n500_gpt-oss_20b.json": dev_top10,
        "results/single_call_top10_n214_test_gpt-oss_20b.json": test_top10,
        "results/tournament_n500_gpt-oss_20b.json": dev_top10,
        "results/tournament_top10_n500_qwen2.5_7b.json": dev_top10,
        "results/single_call_top10_n214_test_qwen3.5_9b.json": test_top10,
        "results/single_call_top10_n500_qwen3.5_9b.json": dev_top10,
        "results/single_call_top10_n500_qwen3.8_27b.json": dev_top10,
        "results/single_call_top10_n500_gemma4_31b.json": dev_top10,
        "results/tournament_top10_n500_qwen3.8_27b.json": dev_top10,
        "results/tournament_top10_n500_gemma4_31b.json": dev_top10,
        "results/single_call_top10_n214_test_qwen3.8_27b.json": test_top10,
        "results/single_call_top10_n214_test_gemma4_31b.json": test_top10,
    }
    for caminho in ROOT_VALUES["resultados"]:
        top10 = top10_por_arquivo.get(caminho, dev_top10)
        saida["llm"][caminho] = metricas_llm(caminho, equivalente, None, top10)

    json.dump(saida, open("results/avaliacao_canonica.json", "w"), ensure_ascii=False, indent=2)

    with open("results/avaliacao_canonica.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "dev_500_strict", "dev_500_equiv", "test_214_strict", "test_214_equiv"])
        r_d = saida["recuperacao"]["dev_500"]
        r_t = saida["recuperacao"]["test_214"]
        for k in ("n_presentes", "perdidos_frac", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50", "p_at_1", "mrr"):
            w.writerow([k, r_d["strict"][k], r_d["equiv"][k], r_t["strict"][k], r_t["equiv"][k]])
        w.writerow(["rank_medio", r_d["strict"]["rank_medio"], r_d["equiv"]["rank_medio"], r_t["strict"]["rank_medio"], r_t["equiv"]["rank_medio"]])
        for k in ("p_at_1_top10_equiv", "match_at_10_top10_equiv"):
            w.writerow([k, r_d["slice_top10"][k], "", r_t["slice_top10"][k], ""])
        w.writerow([])
        w.writerow(["arquivo", "n", "accuracy_total", "accuracy_com_candidatos", "abstencao", "falha", "ganho_vs_p1_tfidf", "mrr_artefato"])
        for caminho, m in saida["llm"].items():
            w.writerow([m["arquivo"], m["n_itens"], m["accuracy_sobre_total"], m["accuracy_detalhe_candidatos"], m["abstencao_rate"], m["falha_rate"], m["ganho_vs_tfidf_p1_top10_equiv"], m["mrr_armazenado_artefato"]])

    # resumo no console
    def pc(v):
        return f"{100*v:.1f}%"
    for nome, dados in saida["recuperacao"].items():
        print(f"[RECUPERACAO {nome}] strict R@1/R@10/R@50: {pc(dados['strict']['recall_at_1'])}/{pc(dados['strict']['recall_at_10'])}/{pc(dados['strict']['recall_at_50'])}"
              f" | equiv R@1={pc(dados['equiv']['recall_at_1'])} p@1={pc(dados['equiv']['p_at_1'])} MMR={dados['equiv']['mrr']:.3f}")
        print(f"   slice top10 equiv: p@1={pc(dados['slice_top10']['p_at_1_top10_equiv'])} match@10={pc(dados['slice_top10']['match_at_10_top10_equiv'])}")
    for caminho, m in saida["llm"].items():
        print(f"[LLM] {caminho}  acc_total={pc(m['accuracy_sobre_total'])} acc_cands={pc(m['accuracy_detalhe_candidatos'])} abst={pc(m['abstencao_rate'])} falha={pc(m['falha_rate'])} ganho={pc(m['ganho_vs_tfidf_p1_top10_equiv'])}")


if __name__ == "__main__":
    main()