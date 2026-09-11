#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 — re-extração de município e modalidade do TEST SET (214 notícias)
=====================================================================
C3 do paper/EQM/tracking_experimentos_dissertacao.md:

No dataset de teste atual, município/modalidade foram COPIADOS do gold
(fonte: fast_object_only_gpt-oss:20b), então o filtro SQL nunca foi
avaliado honestamente no teste. Este script re-extrai OS MESMOS atributos
a partir do TEXTO das notícias com um segundo modelo (qwen3.5:9b,
variabilidade D2), usando o MESMO FeatureExtractor do experimento das 500
(prompt completo, temperatura 0, primeiro elemento da lista = valor usado).

Não sobrescreve dataset_214_noticias_test.json — gera arquivo separado.

Uso:
    OLLAMA_HOST=https://ollama.ceos.ufsc.br OLLAMA_MODEL=qwen3.5:9b \
        python scripts/c3e_extract_mun_mod_test.py [--limit N]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import FeatureExtractor, SELECTED_MODEL, OLLAMA_HOST, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS

DATASET_PATH = BASE_DIR / "data/test/dataset_214_noticias_test.json"
OUTPUT_PATH = BASE_DIR / "data/test/extracao_c3_qwen35.json"
CHECKPOINT_PATH = BASE_DIR / "data/test/extracao_c3_qwen35_checkpoint.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limitar nº de notícias (smoke test)")
    args = parser.parse_args()

    print("=" * 70)
    print(" C3 — RE-EXTRAÇÃO município/modalidade do TEST SET (214) a partir do TEXTO")
    print("=" * 70)
    print(f"  Modelo: {SELECTED_MODEL}")
    print(f"  Host:   {OLLAMA_HOST}")
    print(f"  Temp:   {LLM_TEMPERATURE} | Timeout: {LLM_TIMEOUT_SECONDS}s")

    data = json.load(open(DATASET_PATH, encoding="utf-8"))
    noticias = data["noticias"]
    if args.limit:
        noticias = noticias[: args.limit]
        print(f"  SMOKE TEST: limit={args.limit}")

    # retomar checkpoint parcial, se houver
    feitos = {}
    if CHECKPOINT_PATH.exists():
        for r in json.load(open(CHECKPOINT_PATH, encoding="utf-8")).get("resultados", []):
            feitos[r["indice_par"]] = r
        print(f"  Retomando: {len(feitos)} já extraídos")

    extractor = FeatureExtractor()
    resultados = []

    for i, noticia in enumerate(noticias, 1):
        indice = noticia["indice_par"]
        if indice in feitos:
            resultados.append(feitos[indice])
            continue

        texto = noticia["texto_completo"]
        t0 = time.time()
        try:
            res = extractor.extract(texto, return_metrics=True)
            atr = res.get("atributos", {})
            met = res.get("metricas", {})
            tempo = round(time.time() - t0, 2)

            municipios = atr.get("municipio", []) or []
            modalidades = atr.get("modalidade", []) or []

            rec = {
                "indice_par": indice,
                "id_processo_gold": noticia["id_processo_gold"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "municipios_extraidos": municipios,
                "modalidades_extraidas": modalidades,
                "municipio_extraido": municipios[0] if municipios else "",
                "modalidade_extraida": modalidades[0] if modalidades else "",
                "municipio_correto": bool(municipios) and municipios[0].strip().lower() == noticia["municipio_gold"].strip().lower(),
                "modalidade_correta": bool(modalidades) and modalidades[0].strip().lower() == noticia["modalidade_gold"].strip().lower(),
                "erro": False,
                "tempo_inferencia_s": met.get("tempo_inferencia_s", tempo),
            }
        except Exception as e:
            rec = {
                "indice_par": indice,
                "id_processo_gold": noticia["id_processo_gold"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "municipios_extraidos": [],
                "modalidades_extraidas": [],
                "municipio_extraido": "",
                "modalidade_extraida": "",
                "municipio_correto": False,
                "modalidade_correta": False,
                "erro": True,
                "erro_msg": f"{type(e).__name__}: {e}",
                "tempo_inferencia_s": round(time.time() - t0, 2),
            }
            print(f"\n  [ERRO idx={indice}] {rec['erro_msg']}", flush=True)

        resultados.append(rec)
        feitos[indice] = rec

        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(noticias)}] idx={indice} mun='{rec.get('municipio_extraido','')}' "
                  f"({rec.get('municipio_correto')}) mod='{rec.get('modalidade_extraida','')}' "
                  f"({rec.get('modalidade_correta')}) tempo={rec.get('tempo_inferencia_s',0):.1f}s", flush=True)

        # checkpoint
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"resultados": resultados}, f, ensure_ascii=False)

    # estatísticas
    n = len(resultados)
    mun_ok = sum(1 for r in resultados if r["municipio_correto"])
    mod_ok = sum(1 for r in resultados if r["modalidade_correta"])
    ambos_ok = sum(1 for r in resultados if r["municipio_correto"] and r["modalidade_correta"])
    erros = sum(1 for r in resultados if r["erro"])
    tempos = [r["tempo_inferencia_s"] for r in resultados if r.get("tempo_inferencia_s")]
    stats = {
        "modelo": SELECTED_MODEL,
        "n_total": n,
        "n_municipio_correto": mun_ok,
        "taxa_municipio_correto": mun_ok / n if n else 0,
        "n_modalidade_correta": mod_ok,
        "taxa_modalidade_correta": mod_ok / n if n else 0,
        "n_ambos_corretos": ambos_ok,
        "taxa_ambos_corretos": ambos_ok / n if n else 0,
        "n_erros": erros,
        "tempo_total_s": round(sum(tempos), 2) if tempos else 0,
        "tempo_medio_s": round(sum(tempos) / len(tempos), 2) if tempos else 0,
        "n_municipios_na_lista_extraida": sum(len(r["municipios_extraidos"]) for r in resultados if not r["erro"]),
    }

    print("\n" + "=" * 70)
    print(" ESTATÍSTICAS DA RE-EXTRAÇÃO C3 (vs gold, match case-insensitive no 1º elemento)")
    print(f"  Município correto : {mun_ok}/{n} ({stats['taxa_municipio_correto']:.1%})")
    print(f"  Modalidade correta: {mod_ok}/{n} ({stats['taxa_modalidade_correta']:.1%})")
    print(f"  Ambos corretos    : {ambos_ok}/{n} ({stats['taxa_ambos_corretos']:.1%})")
    print(f"  Erros             : {erros}")
    print("=" * 70)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "descricao": "C3 — re-extração município/modalidade do TEST SET a partir do texto",
                "modelo": SELECTED_MODEL,
                "ollama_host": OLLAMA_HOST,
                "temperatura": LLM_TEMPERATURE,
                "estatisticas": stats,
            },
            "noticias": resultados,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Salvo em: {OUTPUT_PATH}")

    # limpa checkpoint ao concluir
    if erros == 0:
        CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()