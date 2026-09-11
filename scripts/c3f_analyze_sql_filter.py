#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3F — análise do impacto honesto do filtro SQL no TEST SET (214)
===============================================================
Após a re-extração de município/modalidade do texto (c3e_extract_mun_mod_test.py),
este script quantifica o dano do pressuposto do paper: que o filtro SQL recebe
atributos de município/modalidade (que, na verdade, foram copiados do gold).

Computa, para os 214 itens, com matching identical-aware (utils.avaliar_com_*):
  - acurácia da extração (município, modalidade) vs gold
  - quantos itens têm o processo ouro retido pelo filtro SQL usando os atributos EXTRAÍDOS
  - rebuild do cache TF-IDF top-50 "honesto" (filtro sobre atributos extraídos)
  - R@1/R@10/R@50/MRR/Reach strict e equivalent-aware, para comparar com o paper (94,4%)
  - duas políticas de filtro: "strict" (attr vazio -> elimina) e "relaxed" (attr vazio -> ignora só este campo)

Sem chamada de LLM.
"""
import json
import pickle
import sys
import collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import tfidf_cosine_similarity, filtrar_por_municipio_e_modalidade, avaliar_com_identical_objects, avaliar_com_duplicatas

BASE = Path(__file__).parent.parent
EXTRACAO = BASE / "data/test/extracao_c3_qwen35.json"
META = BASE / "models/vector_store/metadata.pkl"
IDF = BASE / "models/tfidf_model/idf_dict.pkl"
DUPS = BASE / "data/duplicatas_mapping.json"
IDENT = BASE / "data/identical_objects_mapping.json"
CACHE_HONESTO_STRICT = BASE / "models/test/cache_tfidf_top50_test_c3_honesto_strict.pkl"
CACHE_HONESTO_RELAX = BASE / "models/test/cache_tfidf_top50_test_c3_honesto_relaxed.pkl"
OUT_JSON = BASE / "results/c3f_analise_sql_filter_test.json"


def recuperacao_tfidf(objeto_query, indices, metadata, idf, k=50):
    scores = []
    for idx in indices:
        desc = metadata[idx].get("descricao_objeto", "")
        if not desc:
            continue
        scores.append({
            "idx": idx,
            "score": tfidf_cosine_similarity(objeto_query, desc, idf),
            "id_processo": metadata[idx].get("id_processo_licitatorio", ""),
            "objeto": desc,
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]


def build_cache(noticias, metadata, idf, dups, ident, policy):
    """policy: 'strict' (attr vazio elimina tudo) ou 'relaxed' (ignora attr vazio)."""
    cache = []
    for n in noticias:
        gold = str(n["id_processo_gold"])
        m_ext = (n["municipio_extraido"] or "").strip()
        mod_ext = (n["modalidade_extraida"] or "").strip()
        objeto = n.get("objeto_extraido", "")  # não reextraímos objeto no C3; usa gold para isolar o efeito do filtro SQL
        indices_sql = filtrar_por_municipio_e_modalidade(metadata, m_ext, mod_ext)
        # política relaxed: se algum attr vazio, filtra só pelo outro
        if policy == "relaxed":
            if not indices_sql:
                if m_ext and not mod_ext:
                    indices_sql = filtrar_por_municipio_e_modalidade(metadata, m_ext, "")
                elif not m_ext and mod_ext:
                    indices_sql = [i for i, m in enumerate(metadata) if m.get("modalidade","").strip().lower() == mod_ext.lower()]
        sem_cand = not indices_sql
        rec = {
            "indice_par": n["indice_par"],
            "id_gold": gold,
            "municipio_extraido": m_ext,
            "modalidade_extraido": mod_ext,
            "municipio_gold": n.get("municipio_gold"),
            "modalidade_gold": n.get("modalidade_gold"),
            "municipio_correto": bool(n.get("municipio_correto")),
            "modalidade_correta": bool(n.get("modalidade_correta")),
            "sem_candidatos_sql": sem_cand,
        }
        if sem_cand:
            rec.update({"candidatos_top50": [], "n_candidatos_sql": 0})
        else:
            cands = recuperacao_tfidf(objeto, indices_sql, metadata, idf, 50)
            rec.update({"candidatos_top50": cands, "n_candidatos_sql": len(indices_sql)})
        cache.append(rec)
    return cache


def metricas(retrieval_rank_of_gold):  # rank position of gold (strict or equiv), with None
    import statistics
    n = len(retrieval_rank_of_gold)
    found = [r for r in retrieval_rank_of_gold if r is not None]
    from collections import Counter as C
    cnt = C(1 if r is None else (1 if r == 1 else 0) for r in retrieval_rank_of_gold)
    return {
        "n": n,
        "n_encontrados": len(found),
        "perdidos_frac": (n - len(found)) / n,
        "recall_at_1": sum(1 for r in retrieval_rank_of_gold if r is not None and r == 1) / n,
        "recall_at_10": sum(1 for r in retrieval_rank_of_gold if r is not None and r <= 10) / n,
        "recall_at_50": sum(1 for r in retrieval_rank_of_gold if r is not None and r <= 50) / n,
        "p_at_1": sum(1 for r in retrieval_rank_of_gold if r is not None and r == 1) / n,
        "mrr": sum(1.0 / r for r in retrieval_rank_of_gold if r is not None) / n,
        "rank_medio_encontrados": round(statistics.mean(found), 2) if found else None,
    }


def gold_survives_sql(n, metadata, policy):
    gold = str(n["id_processo_gold"])
    m_ext = (n.get("municipio_extraido") or "").strip().lower()
    mod_ext = (n.get("modalidade_extraida") or "").strip().lower()
    # acha a linha do processo ouro
    gold_idx = None
    for idx, md in enumerate(metadata):
        if str(md.get("id_processo_licitatorio", "")) == gold:
            gold_idx = idx
            break
    if gold_idx is None:
        return None  # ouro não na base
    md_mun = (metadata[gold_idx].get("municipio") or "").strip().lower()
    md_mod = (metadata[gold_idx].get("modalidade") or "").strip().lower()
    mun_ok = (m_ext == "") or (m_ext == md_mun)
    mod_ok = (mod_ext == "") or (mod_ext == md_mod)
    if policy == "strict":
        return (m_ext != "" and m_ext == md_mun) and (mod_ext != "" and mod_ext == md_mod)
    else:  # relaxed: cada attr presente deve bater; attr vazia é ignorada (gold não eliminado por ela)
        return mun_ok and mod_ok


def main():
    ext = json.load(open(EXTRACAO, encoding="utf-8"))
    noticias = ext["noticias"]
    metadata = pickle.load(open(META, "rb"))
    idf = pickle.load(open(IDF, "rb"))
    dups = json.load(open(DUPS)).get("id_to_group", {})
    ident = json.load(open(IDENT))

    n = len(noticias)
    mun_ok = sum(1 for x in noticias if x.get("municipio_correto"))
    mod_ok = sum(1 for x in noticias if x.get("modalidade_correta"))
    ambos = sum(1 for x in noticias if x.get("municipio_correto") and x.get("modalidade_correta"))
    print(f"Extração (gemma3:12b) vs gold — {n} itens:")
    print(f"  município  correto: {mun_ok}/{n} ({mun_ok/n:.1%})")
    print(f"  modalidade correta: {mod_ok}/{n} ({mod_ok/n:.1%})")
    print(f"  ambos corretos    : {ambos}/{n} ({ambos/n:.1%})")
    # por que modalidade falha
    fails = [x for x in noticias if not x.get("modalidade_correta", False)]
    vacias = sum(1 for x in fails if not x.get("modalidade_extraida"))
    wrong = len(fails) - vacias
    print(f"  modalidade falhas: {len(fails)} (vazias/texto-não-nomeia: {vacias}; erradas: {wrong})")

    for policy in ("strict", "relaxed"):
        # gold surviva no filtro SQL?
        gold_retido = [gold_survives_sql(n, metadata, policy) for n in noticias]
        gold_retido = [True if g is None else g for g in gold_retido]  # None -> considera retido? não; trata como perdido se ausente
        print(f"\n=== Política filtro SQL: {policy} ===")
        print(f"  ouro retido pelo filtro (com attr extraído): {sum(gold_retido)}/{n}")
        cache = build_cache(noticias, metadata, idf, dups, ident, policy)
        pickle.dump(cache, open(CACHE_HONESTO_STRICT if policy == "strict" else CACHE_HONESTO_RELAX, "wb"))
        # ranks de gold dentro do top-50 (strict id e equivalence-aware)
        ranks_strict = []
        ranks_equiv = []
        for c in cache:
            gold = c["id_gold"]
            cands = c["candidatos_top50"]
            rs = next((i + 1 for i, cnd in enumerate(cands) if str(cnd["id_processo"]) == gold), None) if cands else None
            re_rank = next((i + 1 for i, cnd in enumerate(cands) if avaliar_com_identical_objects(str(cnd["id_processo"]), gold, dups, ident)), None) if cands else None
            ranks_strict.append(rs)
            ranks_equiv.append(re_rank)
        ms = metricas(ranks_strict)
        me = metricas(ranks_equiv)
        print(f"  RECUPERAÇÃO honesta top-10 strict: R@1={ms['recall_at_1']:.1%} R@10={ms['recall_at_10']:.1%} R@50={ms['recall_at_50']:.1%} MRR={ms['mrr']:.3f} perdidos={ms['perdidos_frac']:.1%}")
        print(f"  RECUPERAÇÃO honesta top-10 equiv  : R@1={me['recall_at_1']:.1%} R@10={me['recall_at_10']:.1%} R@50={me['recall_at_50']:.1%} MRR={me['mrr']:.3f}")
        json.dump({"politica": policy, "recuperacao_strict": ms, "recuperacao_equiv": me,
                   "gold_retido_pelo_filtro": sum(gold_retido)},
                  open(BASE / f"results/c3f_analise_{policy}.json", "w"), ensure_ascii=False, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()