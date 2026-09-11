#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 (lean/http) — re-extração de município/modalidade do TEST SET (214)
=====================================================================
Versão robusta: chamadas HTTP diretas ao Ollama API (/api/chat) com timeout
por requisição + retry, para sobreviver a latência variável do servidor
compartilhado. Prompt enxuto (só município/modalidade/edital/objeto).
Produz a MESMA estrutura de saída do FeatureExtractor → reaproveita checkpoint.

Uso:
    OLLAMA_HOST=https://ollama.ceos.ufsc.br OLLAMA_MODEL=qwen3.5:9b \
        python3 scripts/c3e_lean.py [opções]
    --limit N   (debug)
    --resume    (usa checkpoint; default on)
    --retries 2 (tentativas por item)
"""
import os
import json
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
PER_CALL_TIMEOUT = 90          # s, timeout do ChatOllama
RETRIES_DEFAULT = 3

DATASET_PATH = BASE_DIR / "data/test/dataset_214_noticias_test.json"
OUTPUT_PATH = BASE_DIR / "data/test/extracao_c3_qwen35.json"
CHECKPOINT_PATH = BASE_DIR / "data/test/extracao_c3_qwen35_checkpoint.json"

LEAN_PROMPT = """Você é um extrator de atributos de notícias sobre licitações públicas de Santa Catarina.
Extraia apenas o que está EXPLICITAMENTE escrito; não invente. Procure no TEXTO DA NOTÍCIA.

REGRAS:
- municipio: o município da licitação (apenas SC). Remova "cidade de"/"município de". Se não houver, [].
  Se houver vários e um for "o principal" (que publicou o edital), devolva só ele.
- modalidade: somente se CLARAMENTE identificada. Normalizações: plural→singular; "dispensa"→"dispensa de licitação";
  "concorrência pública"→"concorrência". Ignore "registro de preços", "pregão público", "concorrência" como competição, "concurso" de pessoas. Se não houver, [].
- edital: todos "NUMERO/ANO". Se não houver, [].
- objeto: descrição do objeto COMPLETA nas PALAVRAS da notícia. Se não houver, [].
Mantenha ordem de correspondência entre modalidade/edital/objeto.

RESPOSTA: APENAS JSON válido, sem texto extra.

TEXTO DA NOTÍCIA:
\"\"\"{text}\"\"\""""


def parse_json_lenient(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i = raw.find("{")
        if i < 0:
            raise
        depth = 0
        j = -1
        for k in range(i, len(raw)):
            if raw[k] == "{":
                depth += 1
            elif raw[k] == "}":
                depth -= 1
                if depth == 0:
                    j = k
                    break
        if j > 0:
            return json.loads(raw[i:j + 1])
        raise


def chat_once(llm, prompt: str) -> str:
    """Uma chamada ao LLM via langchain ChatOllama (trata streaming do qwen3.5)."""
    from langchain_core.messages import HumanMessage
    resp = llm.invoke([HumanMessage(content=prompt)])
    return (resp.content or "").strip()


def chat(llm, prompt: str, retries: int) -> dict:
    """Tenta até `retries` vezes; devolve dict parsed ou levanta na última falha."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            raw = chat_once(llm, prompt)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return parse_json_lenient(raw)
        except Exception as e:              # timeout, http, json
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt)
    raise RuntimeError(f"falhou após {retries} tentativas; último: {last}")


def parse(data: dict) -> dict:
    out = {}
    for key in ["municipio", "modalidade", "edital", "objeto"]:
        v = data.get(key, [])
        if isinstance(v, str):
            v = [v] if v.strip() else []
        elif not isinstance(v, list):
            v = []
        cleaned = []
        seen = set()
        for item in v:
            s = str(item).strip()
            s = s.strip('"').strip("'").strip()
            # descarta resíduos de formatação do modelo ("[ ]", "[", vazio...)
            if s == "" or (set(s) <= set("[] ") and s.strip("[] ") == ""):
                continue
            if s not in seen:
                cleaned.append(s)
                seen.add(s)
        out[key] = cleaned
    for key in ["ano_inicio", "ano_fim"]:
        out[key] = str(data.get(key, "")).strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=RETRIES_DEFAULT)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    print("=" * 70)
    print(" C3 — re-extração município/modalidade/objeto do TEST SET (langchain, timeout+retry)")
    print("=" * 70)
    print(f"  Modelo: {SELECTED_MODEL} | Host: {OLLAMA_HOST} | timeout/req: {PER_CALL_TIMEOUT}s | retries: {args.retries}")

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=SELECTED_MODEL, base_url=OLLAMA_HOST, temperature=0,
                     timeout=PER_CALL_TIMEOUT, max_retries=0)
    print("  ChatOllama pronto")

    data = json.load(open(DATASET_PATH, encoding="utf-8"))
    noticias = data["noticias"]
    if args.limit:
        noticias = noticias[:args.limit]
        print(f"  SMOKE: limit={args.limit}")

    feitos = {}
    if args.resume and CHECKPOINT_PATH.exists():
        feitos = {r["indice_par"]: r for r in json.load(open(CHECKPOINT_PATH, encoding="utf-8"))["resultados"]}
        print(f"  Retomando: {len(feitos)} já extraídos")

    resultados = []
    falhas = 0
    for i, noticia in enumerate(noticias, 1):
        indice = str(noticia["indice_par"])
        if indice in feitos:
            resultados.append(feitos[indice])
            continue

        texto = noticia["texto_completo"]
        t0 = time.time()
        try:
            data_out = chat(llm, LEAN_PROMPT.format(text=texto), args.retries)
            out = parse(data_out)
            municipios = out["municipio"]
            modalidades = out["modalidade"]
            objeto = out["objeto"]
            rec = {
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                "municipios_extraidos": municipios,
                "modalidades_extraidas": modalidades,
                "municipio_extraido": municipios[0] if municipios else "",
                "modalidade_extraida": modalidades[0] if modalidades else "",
                "objeto_extraido": objeto[0] if objeto else "",
                "edital_extraido": out["edital"][0] if out["edital"] else "",
                "municipio_correto": bool(municipios) and municipios[0].strip().lower() == noticia["municipio_gold"].strip().lower(),
                "modalidade_correta": bool(modalidades) and modalidades[0].strip().lower() == noticia["modalidade_gold"].strip().lower(),
                "extracao_completa": bool(municipios and modalidades and objeto),
                "fonte_extracao": f"lean_mun_mod_{SELECTED_MODEL}",
                "metricas_llm": {"tempo_inferencia_s": round(time.time() - t0, 2),
                                 "modelo": SELECTED_MODEL},
                "erro": False,
            }
        except Exception as e:
            falhas += 1
            rec = {
                "indice_par": noticia["indice_par"],
                "id_processo_gold": noticia["id_processo_gold"],
                "municipio_gold": noticia["municipio_gold"],
                "modalidade_gold": noticia["modalidade_gold"],
                "objeto_gold": noticia["objeto_gold"],
                "municipios_extraidos": [],
                "modalidades_extraidas": [],
                "municipio_extraido": "",
                "modalidade_extraida": "",
                "objeto_extraido": "",
                "edital_extraido": "",
                "municipio_correto": False,
                "modalidade_correta": False,
                "extracao_completa": False,
                "fonte_extracao": f"lean_mun_mod_{SELECTED_MODEL}",
                "metricas_llm": {"erro": f"{type(e).__name__}: {e}", "modelo": SELECTED_MODEL},
                "erro": True,
            }
            print(f"\n  [ERRO idx={indice}] {rec['metricas_llm']['erro']}", flush=True)

        resultados.append(rec)
        feitos[indice] = rec

        if i % 10 == 0 or i == 1 or (i % 5 == 0 and i <= 20):
            last = resultados[-1]
            print(f"  [{i}/{len(noticias)}] idx={indice} mun='{last.get('municipio_extraido','')}' "
                  f"({last.get('municipio_correto')}) mod='{last.get('modalidade_extraida','')}' "
                  f"({last.get('modalidade_correta')}) t={last['metricas_llm'].get('tempo_inferencia_s','-')}s "
                  f"erros={falhas}", flush=True)
        if i % 10 == 0:
            json.dump({"resultados": resultados}, open(CHECKPOINT_PATH, "w", encoding="utf-8"))

    json.dump({"resultados": resultados}, open(CHECKPOINT_PATH, "w", encoding="utf-8"))

    n = len(resultados)
    mun_ok = sum(1 for r in resultados if r.get("municipio_correto"))
    mod_ok = sum(1 for r in resultados if r.get("modalidade_correta"))
    ambos = sum(1 for r in resultados if r.get("municipio_correto") and r.get("modalidade_correta"))
    print("\n" + "=" * 70)
    print("  ESTATÍSTICAS (vs gold, match case-insensitive no 1º elemento)")
    print(f"  Município correto : {mun_ok}/{n} ({mun_ok/n:.1%})")
    print(f"  Modalidade correta: {mod_ok}/{n} ({mod_ok/n:.1%})")
    print(f"  Ambos corretos    : {ambos}/{n} ({ambos/n:.1%})")
    print(f"  Erros             : {falhas}")
    print("=" * 70)

    json.dump({
        "metadata": {"descricao": "C3 — re-extração do TEST SET", "modelo": SELECTED_MODEL,
                     "host": OLLAMA_HOST, "timeout_s": PER_CALL_TIMEOUT, "fonte_script": "scripts/c3e_lean.py"},
        "noticias": resultados,
    }, open(OUTPUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"Salvo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()