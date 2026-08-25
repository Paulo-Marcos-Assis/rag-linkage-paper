#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar_noticias_sinteticas_test.py
==================================
Gera 214 notícias sintéticas para o TEST SET do experimento de linkage.

Estratégia (idêntica ao training set):
    Para cada par (notícia_real[i], processo_licitatório[i]):
    - Pega a notícia real (texto original de fraude em licitação)
    - Pega os atributos do processo licitatório (descricao_objeto, modalidade, municipio)
    - Usa o LLM (gpt-oss:20b) para REESCREVER a notícia substituindo as informações
      originais de licitação pelos atributos do processo pareado

Resultado:
    - 214 notícias sintéticas com ground truth conhecido (id_processo_licitatorio)
    - Mapeamento JSON de pares para avaliação

Uso:
    python3 gerar_noticias_sinteticas_test.py                    # executa tudo
    python3 gerar_noticias_sinteticas_test.py --limit 5          # teste com 5 pares
    python3 gerar_noticias_sinteticas_test.py --start-from 100   # retomar a partir do par 100
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
LLM_TEMPERATURE = 0.3  # Leve criatividade para variação no texto

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Entradas (TEST SET)
NOTICIAS_CSV   = os.path.join(BASE_DIR, "214_noticias_test.csv")
PROCESSOS_CSV  = os.path.join(BASE_DIR, "214_processos_test.csv")

# Saídas (TEST SET)
OUTPUT_CSV     = os.path.join(BASE_DIR, "214_noticias_sinteticas_test.csv")
MAPPING_JSON   = os.path.join(BASE_DIR, "mapeamento_pares_test.json")
LOG_FILE       = os.path.join(BASE_DIR, "geracao_log_test.txt")
ERROS_JSON     = os.path.join(BASE_DIR, "erros_geracao_test.json")
STATUS_JSON    = os.path.join(BASE_DIR, "status_execucao_test.json")

# Configurações de retry
MAX_RETRIES    = 3
RETRY_DELAY    = 5  # segundos entre tentativas


# ==============================================================================
# PROMPT PARA O LLM
# ==============================================================================
def build_prompt(texto_noticia_original: str, titulo_original: str,
                 descricao_objeto: str, modalidade: str, municipio: str) -> str:
    return f"""Você é um jornalista investigativo especializado em licitações públicas no Brasil.

CONTEXTO (NÃO MENCIONAR NA RESPOSTA):
Este texto será utilizado em um experimento de Processamento de Linguagem Natural (PLN) com uso de RAG (Retrieval-Augmented Generation).
O objetivo é avaliar se um sistema consegue recuperar corretamente um processo licitatório a partir de uma notícia.

Para isso, a notícia deve conter pistas semânticas realistas, mas NÃO deve copiar literalmente o objeto descrito no documento da licitação.
Apenas a modalidade e o município devem ser mantidos como fornecidos.

Ao descrever o objeto da licitação, interprete o conteúdo do edital e reescreva como um jornalista faria:
- NÃO reproduza o texto técnico do edital literalmente.
- Interprete o conteúdo e reescreva de forma natural e compreensível.
- Explique, de forma indireta, o que está sendo contratado (finalidade prática da licitação).
- Se necessário, simplifique termos técnicos.
- Você pode resumir, reorganizar ou detalhar o objeto para torná-lo mais claro no contexto da notícia.
- Use variações linguísticas (sinônimos, paráfrases), mantendo o mesmo significado.

EXEMPLOS DE ESTILO (NÃO COPIAR, APENAS REFERÊNCIA):

Exemplo 1:
Descrição técnica:
"Aquisição de gêneros alimentícios para alimentação escolar dos alunos das Escolas de Ensino Fundamental, dos Centros de Educação Infantil da Rede Municipal de Ensino e Entidades Filantrópicas Municipais"
Forma jornalística:
"A licitação previa a compra de alimentos destinados à merenda escolar  de alunos da rede municipal e de entidades filantrópicas."

Exemplo 2:
Descrição técnica:
"Registro de preços para futura aquisição de medicamentos para atender a demanda dos postos de saúde do município"
Forma jornalística:
"O processo tratava do fornecimento de medicamentos destinados ao abastecimento dos postos de saúde do município."

Exemplo 3:
Descrição técnica:
"Registro de preços para aquisição de camisetas para campanhas educativas..."
Forma jornalística:
"A contratação envolvia a produção de fornecimento de camisetas voltadas a campanhas educativas da área da saúde do município."

----------------------------------------------------------------------
TAREFA:

Reescreva a notícia abaixo como uma nova matéria jornalística, mantendo:
- A estrutura narrativa (parágrafos, progressão da informação)
- O estilo jornalístico

----------------------------------------------------------------------
NOVOS ATRIBUTOS (USO OBRIGATÓRIO):

- Município: {municipio}
- Modalidade: {modalidade}
- Objeto da licitação: {descricao_objeto}

----------------------------------------------------------------------
REGRAS:

1. USO OBRIGATÓRIO DOS ATRIBUTOS:
   - A nova notícia DEVE refletir claramente os três atributos (município, modalidade e objeto), mesmo que a notícia original não contenha esses elementos.
   - Se a notícia original não mencionar algum desses aspectos, você deve introduzi-los de forma natural.

2. REESCRITA DO OBJETO (REGRA MAIS IMPORTANTE):
   - NÃO copie o texto de "descricao_objeto" literalmente.
   - Reescreva como um jornalista faria:
     → simplifique, resuma ou explique
     → use sinônimos ou descrições equivalentes
     → torne o texto natural e plausível em uma reportagem
   - O objetivo é manter o SIGNIFICADO, mas alterar a FORMA textual.

3. Varie a forma de introduzir o objeto na narrativa:
   - evite padrões repetitivos (ex: "tinha como objeto")
   - use construções diferentes e naturais

4. Mantenha coerência e plausibilidade jornalística:
   - adapte o texto ao novo contexto
   - integre os atributos de forma fluida

5. CONTEXTO DE FRAUDE:
   - Mantenha o tema de suspeita de irregularidade, fraude ou problema na licitação.
   - Isso é essencial para o experimento.

6. PROIBIÇÕES:
   - NÃO invente número de edital, valores (R$), datas ou nomes específicos não fornecidos.
   - NÃO copie trechos longos da notícia original.
   - NÃO explique o que está fazendo.

7. VARIAÇÃO SEMÂNTICA (CRÍTICO PARA O EXPERIMENTO):
   - Evite repetir exatamente as mesmas palavras do objeto.
   - Use variações linguísticas naturais (paráfrase).
   - Introduza pequenas reformulações que preservem o significado.
   - Isso é essencial para testar recuperação semântica (RAG).

8. Idioma: português brasileiro, linguagem jornalística.

9. FORMATO DE SAÍDA (IMPORTANTE):
   - Retorne um objeto JSON com dois campos: "titulo" e "texto"
   - O título deve ser uma manchete jornalística curta e impactante relacionada à notícia reescrita
   - O texto deve ser a notícia completa reescrita
   - Formato exato:
     {{
       "titulo": "Título da notícia sintética aqui",
       "texto": "Texto completo da notícia reescrita aqui..."
     }}

----------------------------------------------------------------------
NOTÍCIA ORIGINAL:

Título: {titulo_original}

Texto: {texto_noticia_original}

----------------------------------------------------------------------

Notícia reescrita (retorne APENAS o JSON):
"""

# ==============================================================================
# CLASSE GERADORA
# ==============================================================================
class GeradorNoticiasSinteticas:
    def __init__(self):
        print(f"Gerador configurado: Ollama em {OLLAMA_HOST} (modelo: {SELECTED_MODEL})")
        self.llm = None
        self._llm_initialized = False

    def _ensure_llm(self):
        if not self._llm_initialized:
            print(f"Conectando ao Ollama em {OLLAMA_HOST}...")
            try:
                self.llm = ChatOllama(
                    model=SELECTED_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=LLM_TEMPERATURE,
                    timeout=180,
                )
                self._llm_initialized = True
                print("Conexão com Ollama estabelecida com sucesso!")
            except Exception as e:
                print(f"ERRO ao conectar ao Ollama: {e}")
                self.llm = None
                self._llm_initialized = True

    def _reconectar_llm(self):
        """Força reconexão ao Ollama"""
        print("  → Tentando reconectar ao Ollama...")
        self._llm_initialized = False
        self.llm = None
        self._ensure_llm()

    def gerar_noticia(self, texto_original: str, titulo_original: str,
                      descricao_objeto: str, modalidade: str, municipio: str) -> dict:
        """
        Gera uma notícia sintética reescrevendo a original com novos atributos.
        Retorna dict com 'titulo' e 'texto', ou dict vazio em caso de erro.
        """
        self._ensure_llm()

        if not self.llm:
            print("[ERRO] LLM não inicializado.")
            return {}

        prompt = build_prompt(texto_original, titulo_original, descricao_objeto, modalidade, municipio)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            # Limpar blocos markdown se o LLM retornar com ```json
            if result.startswith("```"):
                lines = result.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                result = "\n".join(lines).strip()

            # Tentar parsear JSON
            try:
                parsed = json.loads(result)
                if "titulo" in parsed and "texto" in parsed:
                    return {
                        "titulo": parsed["titulo"].strip(),
                        "texto": parsed["texto"].strip()
                    }
                else:
                    print("[AVISO] JSON sem campos 'titulo' ou 'texto'. Usando texto bruto.")
                    return {"titulo": "", "texto": result}
            except json.JSONDecodeError:
                print("[AVISO] Resposta não é JSON válido. Usando texto bruto.")
                return {"titulo": "", "texto": result}

        except Exception as e:
            print(f"[ERRO na geração] {e}")
            return {}

    def gerar_noticia_com_retry(self, texto_original: str, titulo_original: str,
                                descricao_objeto: str, modalidade: str, municipio: str,
                                indice_par: int) -> tuple[dict, dict]:
        """
        Gera notícia com retry automático em caso de falha.
        Retorna: (resultado, info_erro)
        - resultado: dict com 'titulo' e 'texto' ou vazio se falhou
        - info_erro: dict com detalhes do erro ou None se sucesso
        """
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                resultado = self.gerar_noticia(
                    texto_original, titulo_original,
                    descricao_objeto, modalidade, municipio
                )
                
                if resultado:
                    return resultado, None
                else:
                    print(f"  ⚠ Tentativa {tentativa}/{MAX_RETRIES} falhou (resposta vazia)")
                    
            except Exception as e:
                erro_msg = str(e)
                print(f"  ⚠ Tentativa {tentativa}/{MAX_RETRIES} falhou: {erro_msg}")
                
                # Se erro de conexão, tentar reconectar
                if "connection" in erro_msg.lower() or "timeout" in erro_msg.lower():
                    self._reconectar_llm()
            
            # Aguardar antes da próxima tentativa (exceto na última)
            if tentativa < MAX_RETRIES:
                print(f"  → Aguardando {RETRY_DELAY}s antes da próxima tentativa...")
                time.sleep(RETRY_DELAY)
        
        # Todas as tentativas falharam
        info_erro = {
            "indice_par": indice_par,
            "titulo_original": titulo_original[:100],
            "municipio": municipio,
            "tentativas": MAX_RETRIES,
            "timestamp": datetime.now().isoformat(),
            "erro": "Falha após múltiplas tentativas"
        }
        
        return {}, info_erro


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def carregar_noticias(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def carregar_processos(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def salvar_csv(noticias_sinteticas: list[dict], path: str):
    if not noticias_sinteticas:
        return
    fieldnames = noticias_sinteticas[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(noticias_sinteticas)


def salvar_mapping(mapping: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def log(msg: str, log_path: str = LOG_FILE):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atualizar_status(indice_atual: int, total: int, sucessos: int, erros: int,
                     tempo_inicio: float, status_path: str = STATUS_JSON):
    """Atualiza arquivo de status para monitoramento em tempo real"""
    tempo_decorrido = time.time() - tempo_inicio
    progresso_pct = (indice_atual / total) * 100 if total > 0 else 0
    
    # Estimativa de tempo restante
    if indice_atual > 0:
        tempo_por_item = tempo_decorrido / indice_atual
        itens_restantes = total - indice_atual
        tempo_restante_seg = tempo_por_item * itens_restantes
    else:
        tempo_restante_seg = 0
    
    status = {
        "status": "em_execucao" if indice_atual < total else "concluido",
        "progresso": {
            "atual": indice_atual,
            "total": total,
            "percentual": round(progresso_pct, 2),
            "sucessos": sucessos,
            "erros": erros
        },
        "tempo": {
            "inicio": datetime.fromtimestamp(tempo_inicio).isoformat(),
            "decorrido_minutos": round(tempo_decorrido / 60, 2),
            "estimado_restante_minutos": round(tempo_restante_seg / 60, 2),
            "velocidade_items_por_minuto": round((indice_atual / tempo_decorrido) * 60, 2) if tempo_decorrido > 0 else 0
        },
        "ultima_atualizacao": datetime.now().isoformat()
    }
    
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


# ==============================================================================
# MAIN
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Gera notícias sintéticas para TEST SET do experimento de linkage")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N pares (para teste rápido)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Índice para retomar execução (0-indexed)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" GERAÇÃO DE NOTÍCIAS SINTÉTICAS — TEST SET (30%)")
    print("=" * 65)

    # 1. Carregar dados
    noticias  = carregar_noticias(NOTICIAS_CSV)
    processos = carregar_processos(PROCESSOS_CSV)

    print(f"\nNotícias carregadas:  {len(noticias)}")
    print(f"Processos carregados: {len(processos)}")

    n_pares = min(len(noticias), len(processos))
    if args.limit:
        n_pares = min(n_pares, args.limit)
        print(f"\n⚠  MODO TESTE: limitado a {n_pares} pares")

    # 2. Carregar resultados parciais (se retomando)
    noticias_sinteticas = []
    mapping = []
    erros_detalhados = []

    if args.start_from > 0 and os.path.exists(OUTPUT_CSV):
        noticias_sinteticas = carregar_noticias(OUTPUT_CSV)
        if os.path.exists(MAPPING_JSON):
            with open(MAPPING_JSON, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        if os.path.exists(ERROS_JSON):
            with open(ERROS_JSON, "r", encoding="utf-8") as f:
                erros_detalhados = json.load(f)
        print(f"Retomando a partir do par {args.start_from} ({len(noticias_sinteticas)} já gerados, {len(erros_detalhados)} erros)")

    # 3. Inicializar gerador
    gerador = GeradorNoticiasSinteticas()

    # 4. Gerar notícias
    log(f"Início da geração TEST SET: {n_pares} pares (start_from={args.start_from})")
    tempo_inicio = time.time()
    erros = 0
    sucessos = 0

    # Barra de progresso
    pbar = tqdm(
        range(args.start_from, n_pares),
        desc="Gerando notícias TEST",
        initial=args.start_from,
        total=n_pares,
        unit="par",
        ncols=100
    )

    for i in pbar:
        noticia  = noticias[i]
        processo = processos[i]

        texto_original    = noticia.get("texto_noticia", "")
        titulo_original   = noticia.get("titulo", "")
        descricao_objeto  = processo.get("descricao_objeto", "")
        modalidade        = processo.get("modalidade", "")
        municipio         = processo.get("municipio", "")
        id_processo       = processo.get("id_processo_licitatorio", "")
        numero_edital     = processo.get("numero_edital", "")

        log(f"[{i+1:03d}/{n_pares}] Gerando par TEST: noticia[{i}] ↔ processo {id_processo} ({municipio})")

        resultado, info_erro = gerador.gerar_noticia_com_retry(
            texto_original=texto_original,
            titulo_original=titulo_original,
            descricao_objeto=descricao_objeto,
            modalidade=modalidade,
            municipio=municipio,
            indice_par=i,
        )

        if not resultado:
            erros += 1
            log(f"  ⚠ ERRO PERSISTENTE ao gerar par {i}. Mantendo originais.")
            titulo_sintetico = titulo_original
            texto_sintetico = texto_original
            
            # Registrar erro detalhado
            if info_erro:
                info_erro["id_processo_licitatorio"] = id_processo
                info_erro["numero_edital"] = numero_edital
                erros_detalhados.append(info_erro)
        else:
            sucessos += 1
            titulo_sintetico = resultado.get("titulo", titulo_original)
            texto_sintetico = resultado.get("texto", texto_original)
            
            # Se título vazio, usar original
            if not titulo_sintetico.strip():
                titulo_sintetico = titulo_original
            
            # Imprimir notícia gerada completa
            print("\n" + "━" * 80)
            print(f"NOTÍCIA SINTÉTICA GERADA #{i+1}/{n_pares}")
            print("━" * 80)
            print(f"ID Processo: {id_processo}")
            print(f"Município: {municipio}")
            print(f"Modalidade: {modalidade}")
            print(f"Número Edital: {numero_edital}")
            print()
            print("Título Sintético:")
            print(f"  {titulo_sintetico}")
            print()
            print("Texto Sintético (COMPLETO):")
            print(f"  {texto_sintetico}")
            print("━" * 80 + "\n")
        
        # Atualizar barra de progresso
        pbar.set_postfix({
            'OK': sucessos,
            'Erros': erros,
            'Município': municipio[:15] if municipio else 'N/A'
        })

        # Salvar notícia sintética
        noticias_sinteticas.append({
            "indice_par":                i,
            "titulo_original":           titulo_original,
            "texto_original":            texto_original,
            "titulo_sintetico":          titulo_sintetico,
            "texto_sintetico":           texto_sintetico,
            "id_processo_licitatorio":   id_processo,
            "municipio_injetado":        municipio,
            "modalidade_injetada":       modalidade,
            "descricao_objeto_injetada": descricao_objeto,
        })

        # Salvar mapeamento
        mapping.append({
            "indice_par":               i,
            "id_processo_licitatorio":  id_processo,
            "numero_edital":            numero_edital,
            "municipio":                municipio,
            "modalidade":               modalidade,
            "descricao_objeto":         descricao_objeto,
            "titulo_noticia_original":  titulo_original,
        })

        # Salvar progresso a cada 10 registros (para poder retomar)
        if (i + 1) % 10 == 0 or i == n_pares - 1:
            salvar_csv(noticias_sinteticas, OUTPUT_CSV)
            salvar_mapping(mapping, MAPPING_JSON)
            
            # Salvar erros se houver
            if erros_detalhados:
                with open(ERROS_JSON, "w", encoding="utf-8") as f:
                    json.dump(erros_detalhados, f, ensure_ascii=False, indent=2)
            
            # Atualizar status
            atualizar_status(i + 1, n_pares, sucessos, erros, tempo_inicio)
            
            elapsed = time.time() - tempo_inicio
            avg = elapsed / (i - args.start_from + 1)
            remaining = avg * (n_pares - i - 1)
            log(f"  ✓ Progresso salvo ({i+1}/{n_pares}) | "
                f"Tempo: {elapsed/60:.1f}min | "
                f"Restante estimado: {remaining/60:.1f}min | "
                f"Erros: {len(erros_detalhados)}")

    # Fechar barra de progresso
    pbar.close()

    # 5. Salvar final
    salvar_csv(noticias_sinteticas, OUTPUT_CSV)
    salvar_mapping(mapping, MAPPING_JSON)
    
    # Salvar erros finais
    if erros_detalhados:
        with open(ERROS_JSON, "w", encoding="utf-8") as f:
            json.dump(erros_detalhados, f, ensure_ascii=False, indent=2)
    
    # Atualizar status final
    tempo_total = time.time() - tempo_inicio
    atualizar_status(n_pares, n_pares, sucessos, erros, tempo_inicio)

    print(f"\n{'='*65}")
    print(f" CONCLUÍDO! (TEST SET)")
    print(f"{'='*65}")
    print(f"  Total gerado:  {len(noticias_sinteticas)} notícias sintéticas")
    print(f"  Sucessos:      {len(noticias_sinteticas) - erros}")
    print(f"  Erros:         {erros}")
    if erros_detalhados:
        print(f"  Arquivo erros: {ERROS_JSON}")
        print(f"                 ({len(erros_detalhados)} pares com falha)")
    print(f"  Tempo total:   {tempo_total/60:.1f} min")
    print(f"  CSV saída:     {OUTPUT_CSV}")
    print(f"  Mapeamento:    {MAPPING_JSON}")
    print(f"{'='*65}")
    
    if erros_detalhados:
        print(f"\n⚠️  ATENÇÃO: {len(erros_detalhados)} pares falharam após {MAX_RETRIES} tentativas.")
        print(f"   Verifique {ERROS_JSON} para detalhes.")
        print(f"   Para reprocessar apenas os erros, use o arquivo de erros como referência.")


if __name__ == "__main__":
    main()
