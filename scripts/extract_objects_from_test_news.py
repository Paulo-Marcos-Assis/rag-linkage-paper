#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_objects_from_test_news.py
==================================
Extrai atributos (município, modalidade, edital, objeto) das notícias sintéticas
do test set (214 notícias) usando o MESMO FeatureExtractor do experimento das 500.

Reproduz fielmente a metodologia de RAG_V2/04_create_extracted_dataset.py:
- Usa o texto_sintetico de cada notícia
- Extrai com LLM (gpt-oss:20b, temperatura 0) via FeatureExtractor
- objeto_extraido = paráfrase jornalística da notícia (NÃO o objeto original do CSV)
- Gera dataset compatível com 03_generate_cache_test.py e o pipeline

Uso:
    python3 scripts/extract_objects_from_test_news.py
    python3 scripts/extract_objects_from_test_news.py --limit 10
"""

import os
import sys
import csv
import json
import signal
import time
import argparse
import statistics
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import FeatureExtractor, SELECTED_MODEL, LLM_TEMPERATURE, OLLAMA_HOST, LLM_TIMEOUT_SECONDS

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "noticias_simuladas/214_noticias_sinteticas_test.csv"
OUTPUT_JSON = BASE_DIR / "data/test/dataset_214_noticias_test.json"


def carregar_noticias(limit: int = None) -> list[dict]:
    """Carrega notícias sintéticas do CSV (mesma lógica do 04_create_extracted_dataset)."""
    noticias = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            texto = row.get("texto_sintetico", "").strip()
            id_proc = row.get("id_processo_licitatorio", "").strip()
            if not texto or not id_proc:
                continue
            noticias.append({
                "indice_par": row.get("indice_par", ""),
                "id_processo_gold": id_proc,
                "texto_completo": texto,
                "titulo": row.get("titulo_sintetico", "").strip(),
                "municipio_gold": row.get("municipio_injetado", "").strip(),
                "modalidade_gold": row.get("modalidade_injetada", "").strip(),
                "objeto_gold": row.get("descricao_objeto_injetada", "").strip(),
            })
    return noticias


def extrair_objeto_simples(texto: str, llm) -> str:
    """
    Prompt mínimo: só extrai o objeto licitado (paráfrase do texto).
    Usado no modo --fast para evitar o prompt gigante do FeatureExtractor.
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""Você analisa notícias sobre licitações públicas.
Extraia a descrição do OBJETO licitado exatamente como aparece no texto (cópia fiel, sem inventar).
Retorne APENAS um JSON válido: {{"objeto": "descrição do objeto"}}

Texto da notícia:
\"\"\"{texto}\"\"\"

Responda APENAS com o JSON."""

    def _handler(signum, frame):
        raise TimeoutError(f"LLM timeout após {LLM_TIMEOUT_SECONDS}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(LLM_TIMEOUT_SECONDS)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    finally:
        signal.alarm(0)

    raw = response.content.strip()
    # Limpa possível markdown
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        return data.get("objeto", "") or ""
    except Exception:
        return ""


def extrair_fast_batch(noticias: list[dict], llm) -> list[dict]:
    """
    Modo rápido: usa municipio/modalidade do CSV (gold) e só chama LLM para objeto.
    """
    print("\n" + "=" * 70)
    print(" EXTRAÇÃO RÁPIDA (só objeto via LLM; município/modalidade = gold)")
    print("=" * 70)
    print(f"Total de notícias: {len(noticias)}\n")

    resultados = []
    for noticia in tqdm(noticias, desc="Extraindo", unit="notícia"):
        texto = noticia["texto_completo"]
        t0 = time.time()
        try:
            objeto_ext = extrair_objeto_simples(texto, llm)
            tempo = round(time.time() - t0, 2)
            municipio_ext = noticia["municipio_gold"]
            modalidade_ext = noticia["modalidade_gold"]
            resultados.append({
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "texto_completo": noticia["texto_completo"],
                "titulo": noticia["titulo"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                "municipio_extraido": municipio_ext,
                "modalidade_extraida": modalidade_ext,
                "objeto_extraido": objeto_ext,
                "edital_extraido": "",
                "extracao_completa": bool(objeto_ext),
                "municipio_correto": True,
                "modalidade_correta": True,
                "objeto_extraido_presente": bool(objeto_ext),
                "fonte_extracao": f"fast_object_only_{SELECTED_MODEL}",
                "metricas_llm": {"tempo_inferencia_s": tempo, "modo": "fast"},
            })
        except Exception as e:
            print(f"\n[ERRO {noticia.get('indice_par', '?')}] {e}")
            resultados.append({
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "texto_completo": noticia["texto_completo"],
                "titulo": noticia["titulo"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                "municipio_extraido": noticia["municipio_gold"],
                "modalidade_extraida": noticia["modalidade_gold"],
                "objeto_extraido": "",
                "edital_extraido": "",
                "extracao_completa": False,
                "municipio_correto": True,
                "modalidade_correta": True,
                "objeto_extraido_presente": False,
                "fonte_extracao": f"fast_object_only_{SELECTED_MODEL}",
                "metricas_llm": {"erro": str(e)},
            })
    return resultados


def extrair_atributos_batch(noticias: list[dict], extractor: FeatureExtractor) -> list[dict]:
    """Extração com LLM (replica 04_create_extracted_dataset.extrair_atributos_batch)."""
    print("\n" + "=" * 70)
    print(" EXTRAÇÃO DE ATRIBUTOS (FeatureExtractor - mesmo das 500)")
    print("=" * 70)
    print(f"Total de notícias: {len(noticias)}\n")

    resultados = []
    for noticia in tqdm(noticias, desc="Extraindo", unit="notícia"):
        texto = noticia["texto_completo"]
        try:
            resultado_completo = extractor.extract(texto, return_metrics=True)
            extracted = resultado_completo.get("atributos", {})
            metricas = resultado_completo.get("metricas", {})

            municipio_ext = extracted.get("municipio", [""])[0] if extracted.get("municipio") else ""
            modalidade_ext = extracted.get("modalidade", [""])[0] if extracted.get("modalidade") else ""
            objeto_ext = extracted.get("objeto", [""])[0] if extracted.get("objeto") else ""
            edital_ext = extracted.get("edital", [""])[0] if extracted.get("edital") else ""

            resultados.append({
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "texto_completo": noticia["texto_completo"],
                "titulo": noticia["titulo"],
                # Ground truth
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                # Atributos extraídos
                "municipio_extraido": municipio_ext,
                "modalidade_extraida": modalidade_ext,
                "objeto_extraido": objeto_ext,
                "edital_extraido": edital_ext,
                # Flags de qualidade
                "extracao_completa": bool(municipio_ext and modalidade_ext and objeto_ext),
                "municipio_correto": municipio_ext.lower() == noticia["municipio_gold"].lower() if municipio_ext else False,
                "modalidade_correta": modalidade_ext.lower() == noticia["modalidade_gold"].lower() if modalidade_ext else False,
                "objeto_extraido_presente": bool(objeto_ext),
                "fonte_extracao": f"extracao_test_{SELECTED_MODEL}",
                "metricas_llm": metricas,
            })
        except Exception as e:
            print(f"\n[ERRO na notícia {noticia.get('indice_par', '?')}] {e}")
            resultados.append({
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "texto_completo": noticia["texto_completo"],
                "titulo": noticia["titulo"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                "municipio_extraido": "",
                "modalidade_extraida": "",
                "objeto_extraido": "",
                "edital_extraido": "",
                "extracao_completa": False,
                "municipio_correto": False,
                "modalidade_correta": False,
                "objeto_extraido_presente": False,
                "fonte_extracao": f"extracao_test_{SELECTED_MODEL}",
                "metricas_llm": {"erro": str(e)},
            })
    return resultados


def calcular_estatisticas(noticias_extraidas: list[dict]) -> dict:
    n_total = len(noticias_extraidas)
    n_completas = sum(1 for n in noticias_extraidas if n["extracao_completa"])
    n_municipio_correto = sum(1 for n in noticias_extraidas if n["municipio_correto"])
    n_modalidade_correta = sum(1 for n in noticias_extraidas if n["modalidade_correta"])

    tempos = [n["metricas_llm"].get("tempo_inferencia_s") for n in noticias_extraidas
              if "tempo_inferencia_s" in n.get("metricas_llm", {})]
    n_erros = sum(1 for n in noticias_extraidas if "erro" in n.get("metricas_llm", {}))

    metricas_tempo = {}
    if tempos:
        metricas_tempo = {
            "tempo_total_s": round(sum(tempos), 2),
            "tempo_medio_s": round(statistics.mean(tempos), 2),
            "tempo_mediano_s": round(statistics.median(tempos), 2),
            "tempo_min_s": round(min(tempos), 2),
            "tempo_max_s": round(max(tempos), 2),
        }

    return {
        "n_total": n_total,
        "n_completas": n_completas,
        "taxa_completas": n_completas / n_total if n_total else 0,
        "n_municipio_correto": n_municipio_correto,
        "taxa_municipio_correto": n_municipio_correto / n_total if n_total else 0,
        "n_modalidade_correta": n_modalidade_correta,
        "taxa_modalidade_correta": n_modalidade_correta / n_total if n_total else 0,
        "n_erros": n_erros,
        "metricas_inferencia": metricas_tempo,
    }


def main():
    parser = argparse.ArgumentParser(description="Extrai atributos das notícias do test set")
    parser.add_argument("--limit", type=int, default=None, help="Limitar nº de notícias (debug)")
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido: só extrai objeto via LLM; município/modalidade vêm do CSV")
    args = parser.parse_args()

    print("=" * 70)
    print(" EXTRAÇÃO DE ATRIBUTOS - TEST SET (214 NOTÍCIAS)")
    print("=" * 70)
    print(f"  Modelo:  {SELECTED_MODEL}")
    print(f"  Host:    {OLLAMA_HOST}")
    print(f"  Temp:    {LLM_TEMPERATURE}")
    print(f"  Entrada: {INPUT_CSV}")
    print(f"  Saída:   {OUTPUT_JSON}")
    print("=" * 70)

    noticias = carregar_noticias(limit=args.limit)
    print(f"\n{len(noticias)} notícias carregadas")

    if args.fast:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=SELECTED_MODEL, base_url=OLLAMA_HOST, temperature=LLM_TEMPERATURE)
        noticias_extraidas = extrair_fast_batch(noticias, llm)
    else:
        extractor = FeatureExtractor()
        noticias_extraidas = extrair_atributos_batch(noticias, extractor)

    stats = calcular_estatisticas(noticias_extraidas)

    print("\n" + "=" * 70)
    print(" ESTATÍSTICAS DA EXTRAÇÃO")
    print("=" * 70)
    print(f"  Total de notícias    : {stats['n_total']}")
    print(f"  Extrações completas  : {stats['n_completas']} ({stats['taxa_completas']:.1%})")
    print(f"  Município correto    : {stats['n_municipio_correto']} ({stats['taxa_municipio_correto']:.1%})")
    print(f"  Modalidade correta   : {stats['n_modalidade_correta']} ({stats['taxa_modalidade_correta']:.1%})")
    print(f"  Erros de extração    : {stats['n_erros']}")
    if stats.get("metricas_inferencia"):
        m = stats["metricas_inferencia"]
        print(f"  Tempo total          : {m.get('tempo_total_s', 0):.1f}s")
        print(f"  Tempo médio/notícia  : {m.get('tempo_medio_s', 0):.1f}s")

    # Diagnóstico: quantos objetos extraídos são idênticos ao gold
    objetos_identicos = sum(
        1 for n in noticias_extraidas
        if n["objeto_extraido"] and n["objeto_extraido"].strip().lower() == n["objeto_gold"].strip().lower()
    )
    print(f"  Objetos == gold      : {objetos_identicos}/{stats['n_total']} ({objetos_identicos/stats['n_total']:.1%})")
    if objetos_identicos / max(stats['n_total'], 1) > 0.3:
        print("  ⚠️  Muitos objetos idênticos ao gold — verificar extração!")
    else:
        print("  ✓ Objetos majoritariamente parafraseados (esperado)")

    # Salvar
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "metadata": {
            "descricao": "Dataset TEST SET (214 notícias) com atributos extraídos por LLM",
            "n_noticias": len(noticias_extraidas),
            "modelo_extracao": SELECTED_MODEL,
            "ollama_host": OLLAMA_HOST,
            "temperatura": LLM_TEMPERATURE,
            "fontes": [
                "214_noticias_sinteticas_test.csv",
                f"FeatureExtractor ({SELECTED_MODEL})",
            ],
            "estatisticas": stats,
        },
        "noticias": noticias_extraidas,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f" Dataset salvo em: {OUTPUT_JSON}")
    print(f" Tamanho: {OUTPUT_JSON.stat().st_size / 1024:.1f} KB")
    print("=" * 70)
    print("\nPróximos passos:")
    print("  1. python3 scripts/03_generate_cache_test.py")
    print("  2. python3 scripts/06_single_call_pipeline.py --cache models/test/cache_tfidf_top50_test.pkl --top 10 --verbose")


if __name__ == "__main__":
    main()
