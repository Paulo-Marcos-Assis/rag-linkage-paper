import os
import json
import time
import signal
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ===========================
# CONFIGURAÇÕES (Carregadas de Var. de Ambiente ou Padrão)
# ===========================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b") # "gpt-oss:20b", "qwen3:8b", "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b", 
LLM_TEMPERATURE = 0
LLM_TIMEOUT_SECONDS = 900  # hard limit via signal.alarm (reasoning model pode ser lento)

# ===========================
# FUNÇÕES AUXILIARES
# ===========================
def normalize_edital(edital_str: Optional[str]) -> Optional[str]:
    """Remove zeros à esquerda do número do edital (ex: 005/2023 -> 5/2023)."""
    if edital_str is None:
        return None
    editorial = str(edital_str).strip()
    parts = editorial.split('/')
    if len(parts) == 2:
        num_norm = parts[0].lstrip('0')
        if not num_norm:
            num_norm = '0'
        return f"{num_norm}/{parts[1]}"
    else:
        return editorial

# ===========================
# CLASSE PRINCIPAL
# ===========================
class FeatureExtractor:
    def __init__(self):
        print(f"Feature Extractor configurado para usar Ollama em {OLLAMA_HOST} (modelo: {SELECTED_MODEL})")
        self.llm = None
        self._llm_initialized = False
    
    def _ensure_llm(self):
        """Lazy initialization do LLM - só conecta quando realmente precisar"""
        if not self._llm_initialized:
            print(f"Conectando ao Ollama em {OLLAMA_HOST}...")
            try:
                self.llm = ChatOllama(
                    model=SELECTED_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=LLM_TEMPERATURE,
                    # timeout=None: sem timeout httpx no streaming;
                    # o hard limit é o signal.alarm(LLM_TIMEOUT_SECONDS) no extract()
                )
                self._llm_initialized = True
                print("Conexão com Ollama estabelecida com sucesso!")
            except Exception as e:
                print(f"ERRO ao conectar ao Ollama: {e}")
                self.llm = None
                self._llm_initialized = True

    def extract(self, text: str, return_metrics: bool = False) -> Dict:
        """
        Recebe o texto bruto da notícia e retorna o dicionário extraído.
        
        Args:
            text: Texto da notícia
            return_metrics: Se True, retorna também métricas de tempo
        
        Returns:
            Se return_metrics=False: {"municipio": [], "modalidade": [], "edital": [], "objeto": [], "ano_inicio": "", "ano_fim": ""}
            Se return_metrics=True: {
                "atributos": {"municipio": [], "modalidade": [], "edital": [], "objeto": [], "ano_inicio": "", "ano_fim": ""},
                "metricas": {"tempo_inferencia_s": float, "tamanho_prompt_chars": int, ...}
            }
        """
        default_return = {"municipio": [], "modalidade": [], "edital": [], "objeto": [], "ano_inicio": "", "ano_fim": ""}
        
        # Garante que o LLM está inicializado
        self._ensure_llm()
        
        if not self.llm:
            print("Erro: LLM não inicializado.")
            if return_metrics:
                return {"atributos": default_return, "metricas": {"erro": "LLM não inicializado"}}
            return default_return
        
        if not text or not isinstance(text, str):
            if return_metrics:
                return {"atributos": default_return, "metricas": {"erro": "Texto inválido"}}
            return default_return

        
        prompt_content = f"""
Você é um especialista em análise de notícias sobre licitações públicas.
Sua tarefa é identificar e extrair, de forma precisa e sem inferências, atributos específicos presentes **explicitamente** no texto.

Essas informações serão utilizadas posteriormente para cruzamento com bases públicas de licitações. Portanto, siga rigorosamente as regras de extração e normalização.

----------------------------------------------------------------------
INSTRUÇÕES GERAIS:
- **EXTRAIA INFORMAÇÕES DO TEXTO DA NOTÍCIA COMO UMA CÓPIA EXATA, NÃO INVENTE, APENAS "COPIE" O QUE ESTÁ ESCRITO**
- Leia o todo o texto da notícia.
- Extraia SOMENTE informações que apareçam de forma explícita.
- Não faça inferências, não complete ausências e não reformule conteúdos.
- Para cada atributo, retorne uma lista (array). 
- Use [] quando não houver ocorrências.
- Remova duplicatas.
- A resposta deve ser APENAS um JSON válido.

----------------------------------------------------------------------
1) município:
- **O objetivo é encontrar o município relacionado à licitação**
- **Identifique e extraia o(s) município(s) que o texto da notícia aborda**.

Considerações:
- Se o texto mencionar vários municípios, mas um deles for claramente o principal, aquele que publicou o edital de licitação, então retorne apenas o município principal. 
- Se o texto mencionar diversos municípios e todos tiverem a mesma importância, retorne a lista com todos os municípios
- Se no texto não houver menção de nenhum município, não retorne nada (vazio que será preenchido por [])
- Se houver menção de municípios que não são do estado de Santa Catarina, não retorne nada (vazio que será preenchido por [])
- É possível que a licitação seja expedida pelo Governo do Estado de Santa Catarina, nesse caso retorne "Estado de Santa Catarina"

- Normalizações:
  - Remova expressões como "cidade de" ou "município de".
  - Mantenha apenas o nome principal.
  - Exemplo: "cidade de Florianópolis" → "Florianópolis"
  
### Exemplo few-shot:

Notícia 1:
"a polícia civil de santa catarina, por meio da 5ª delegacia especializada no combate à corrupção (5ª decor/chapecó), deflagrou uma operação na data de hoje, 02/05/2024, que resultou no cumprimento de nove mandados de busca e apreensão e três mandados de prisão nos municípios de quilombo/sc, são lourenço do oeste/sc e pato branco/pr.

a ação é um desdobramento da investigação de supostas fraudes em licitações no setor de obras do município de quilombo/sc, além de outras infrações penais correlatas, como formação de organização criminosa, lavagem de dinheiro e advocacia administrativa.
"
Note que: pelo eixo da fraude ser o municíio de quilombo/sc "*supostas fraudes em licitações no setor de obras do município de quilombo/sc*",a saída esperada nesse caso é:

Saída esperada:
{{
  "municipio_ente": "quilombo"
}}

Notícia 2:
"Operação mira 18 prefeituras de Santa Catarina por suspeita de fraude (...) a CNN apurou que 18 prefeituras do estado são alvos de busca e apreensão. São elas: São Miguel do Oeste, Guaraciaba, São José do Cedro, Bom Jesus do Oeste, Princesa, Bandeirantes, Flor do Sertão, São João do Oeste, Santa Helena, Sul Brasil, Descanso, Riqueza, Mondaí, Cordilheira Alta, Jardinópolis, Rio Fortuna, Águas Mornas e Antônio Carlos."

Saída esperada:
{{
  "municipio_ente": "São Miguel do Oeste, Guaraciaba, São José do Cedro, Bom Jesus do Oeste, Princesa, Bandeirantes, Flor do Sertão, São João do Oeste, Santa Helena, Sul Brasil, Descanso, Riqueza, Mondaí, Cordilheira Alta, Jardinópolis, Rio Fortuna, Águas Mornas e Antônio Carlos"
}}

----------------------------------------------------------------------
2) modalidade:
Extraia a modalidade de licitação **somente se claramente identificada como modalidade**.

Modalidades válidas (exemplos):
- pregão presencial
- pregão eletrônico 
- concorrência
- concorrência presencial
- concorrência eletrônica
- convite
- tomada de preços
- dispensa de licitação
- inexigibilidade de licitação
- leilão
- concurso (apenas quando modalidade de licitação, não para concurso público)
- regime diferenciado de contratação (RDC)

REGRAS DE NORMALIZAÇÃO:
- Converter plurais para singular (ex: "pregões" → "pregão").
- "dispensa" (singular ou plural) → normalizar para "dispensa de licitação".
- "concorrência pública" → normalizar para apenas "concorrência".
- "concorrência" → só extraia quando for CERTEZA que refere-se à modalidade.

O QUE IGNORAR (NÃO EXTRAIR EM NENHUMA HIPÓTESE):
- “registro de preço” ou “sistema de registro de preço”.
- “pregão público” (termo genérico).
- “concorrência” usada no sentido de competição (ex: “frustrar a concorrência”).
- “concurso” usado para seleção de pessoal (ex: "concurso público para cargos").

----------------------------------------------------------------------
3) edital:
- Extraia TODOS os números de editais mencionados no texto, incluindo editais de diferentes modalidades (pregão, dispensa, inexigibilidade, tomada de preços, etc.).
- Se houver múltiplos editais no texto, extraia todos eles.
- Normalizar para formato "NUMERO/ANO", removendo prefixos como siglas de órgãos.
- Exemplos de normalização:
  - "Edital nº 123/2023" → "123/2023"
  - "pregão número 24 de 2022" → "24/2022"
  - "Dispensa de Licitação nº 283/SMLCP/SULIC/2023" → "283/2023"
  - "Pregão Eletrônico nº 058/SMLCP/SULIC/2024" → "58/2024"
  - "Tomada de Preços n. 03/2023" → "3/2023"
- IMPORTANTE: Retorne TODOS os editais encontrados, mesmo que sejam de contexto histórico ou secundário.
- Se um edital for mencionado múltiplas vezes, inclua apenas uma vez (sem duplicatas).

----------------------------------------------------------------------
4) objeto:

**OBJETIVO:** Extrair a descrição COMPLETA do objeto da licitação, usando as PALAVRAS DA NOTÍCIA.

**CONTEXTO IMPORTANTE:**
As notícias descrevem licitações em linguagem jornalística — o texto da notícia pode não ser idêntico ao texto formal do edital, mas contém as mesmas informações. Sua tarefa é extrair o máximo de informação sobre o objeto que aparece na notícia a respeito de um determinado objeto (o que está sendo licitado), mesmo que seja uma descrição jornalística, paráfrase ou resumo.

**REGRAS CRÍTICAS:**
- Use as PALAVRAS DA NOTÍCIA, não invente ou substitua termos
- Se o objeto for descrito em vários trechos da notícia, combine-os em um único texto coerente
- Preserve obrigatoriamente:
  → A FINALIDADE da contratação (o que está sendo licitado)
  → O TIPO de objeto (serviço / obra / aquisição de material)
  → O CONTEXTO funcional (secretaria, órgão, local, beneficiário) quando mencionado
  → Especificações técnicas relevantes (marcas, modelos, quantidades) quando presentes
- Inclua o termo "registro de preços" quando mencionado
- NÃO inclua informações que não descrevem o objeto (ex: nome de investigados, datas do processo, valor pago)

**O QUE NÃO FAZER:**
- ❌ Reduzir a uma palavra ou frase genérica: "aquisição de materiais" em vez da descrição completa
- ❌ Inventar termos que não estão na notícia
- ❌ Trocar o texto da notícia pelo que seria o texto formal do edital
- ❌ Incluir contexto investigativo/jurídico que não descreve o objeto (ex: "conforme apurou a delegacia")

### EXEMPLOS FEW-SHOT:

**Exemplo 1 — A notícia descreve com linguagem jornalística:**
Notícia: "A Prefeitura de Campos Novos publicou pregão eletrônico para adquirir concreto asfáltico quente, material que será usado nos serviços de tapa-buraco nas ruas da cidade."

✅ CORRETO (usa as palavras da notícia, completo):
{{
  "objeto": ["adquirir concreto asfáltico quente para os serviços de tapa-buraco nas ruas da cidade"]
}}

❌ ERRADO (inventou texto do edital formal):
{{
  "objeto": ["REGISTRO DE PREÇO PARA AQUISIÇÃO DE CONCRETO ASFÁLTICO USINADO A QUENTE QUE SERÁ UTILIZADO PARA TAPA BURACOS EM VIAS PÚBLICAS"]
}}

**Exemplo 2 — A notícia descreve com palavras próximas ao edital:**
Notícia: "A dispensa de licitação tinha como objetivo a contratação de empresa para aquisição de peças para a realização da manutenção dos implementos agrícolas da Secretaria de Agricultura."

✅ CORRETO:
{{
  "objeto": ["contratação de empresa para aquisição de peças para a realização da manutenção dos implementos agrícolas da Secretaria de Agricultura"]
}}

❌ ERRADO (resumido):
{{
  "objeto": ["compra de peças de reposição para maquinário agrícola"]
}}

**Exemplo 3 — Objeto com especificações técnicas detalhadas:**
Notícia: "O processo licitatório tinha como objeto a contratação de empresa, mediante dispensa de licitação, para prestação de serviço de seguro total dos equipamentos da usina de asfalto, por item, quais sejam: Usina de Asfalto Móvel 20-40 Ton/h, ano de fabricação 2018 Tanque bipartido - Diesel - Pesagem Ind. 380V; Vibroacabadora de Asfalto, marca CIBER, modelo AF 4000, sobre esteiras."

✅ CORRETO (incluir todos os detalhes técnicos):
{{
  "objeto": ["contratação de empresa para prestação de serviço de seguro total dos equipamentos da usina de asfalto, por item: Usina de Asfalto Móvel 20-40 Ton/h, ano de fabricação 2018 Tanque bipartido - Diesel - Pesagem Ind. 380V; Vibroacabadora de Asfalto, marca CIBER, modelo AF 4000, sobre esteiras"]
}}

❌ ERRADO (perdeu os detalhes técnicos):
{{
  "objeto": ["serviço de seguro dos equipamentos da usina de asfalto"]
}}

**Exemplo 4 — Registro de preços com beneficiários:**
Notícia: "O pregão tinha como objetivo registrar preços para o fornecimento parcelado de alimentos não perecíveis e itens correlatos para uso dos órgãos consorciados ao Consórcio Interfederativo Santa Catarina – CINCATARINA."

✅ CORRETO:
{{
  "objeto": ["registro de preços para fornecimento parcelado de alimentos não perecíveis e itens correlatos para uso dos órgãos consorciados ao Consórcio Interfederativo Santa Catarina – CINCATARINA"]
}}

❌ ERRADO (perdeu beneficiário):
{{
  "objeto": ["fornecimento de alimentos não perecíveis"]
}}

----------------------------------------------------------------------
5) ano_inicio e ano_fim:
- **Identifique o ANO ou PERÍODO em que a licitação/fraude ocorreu**
- Se houver apenas UM ano: retorne o mesmo em ano_inicio e ano_fim
- Se houver PERÍODO (ex: "entre 2021 e 2024"): retorne o ano mais antigo em ano_inicio e o mais recente em ano_fim
- Se houver MÚLTIPLOS anos isolados: retorne o mais antigo e o mais recente
- **NÃO INTERPOLE** - retorne apenas início e fim
- Se não houver data, deixe ambos vazios

Exemplos:
- "edital 42/2013" → ano_inicio: "2013", ano_fim: "2013"
- "fraudes entre 2020 e 2023" → ano_inicio: "2020", ano_fim: "2023"
- "licitações de 2018, 2020 e 2022" → ano_inicio: "2018", ano_fim: "2022"


----------------------------------------------------------------------
FORMATO DE RESPOSTA:
Retorne APENAS um JSON válido, no formato:

{{
  "municipio": [],
  "modalidade": [],
  "edital": [],
  "objeto": [],
  "ano_inicio": "",
  "ano_fim": ""
}}

**IMPORTANTE - ORDEM DE CORRESPONDÊNCIA:**
Quando houver múltiplos editais/modalidades/objetos, mantenha a ORDEM DE CORRESPONDÊNCIA entre eles:
- O primeiro item em "modalidade" deve corresponder ao primeiro item em "edital" e "objeto"
- O segundo item em "modalidade" deve corresponder ao segundo item em "edital" e "objeto"
- E assim sucessivamente

Exemplo:
Se a notícia menciona:
1. "Dispensa de Licitação nº 283/2023 para contratação de serviços de TI"
2. "Pregão Eletrônico nº 58/2024 para compra de software"

A saída deve ser:
{{
  "modalidade": ["dispensa de licitação", "pregão eletrônico"],
  "edital": ["283/2023", "58/2024"],
  "objeto": ["contratação de serviços de TI", "compra de software"]
}}

Se um atributo não tiver correspondente, use string vazia "" na posição correspondente para manter o alinhamento.

*POR FIM: **EXTRAIA INFORMAÇÕES DO TEXTO DA NOTÍCIA COMO UMA CÓPIA EXATA, NÃO INVENTE, APENAS "COPIE" O QUE ESTÁ ESCRITO**
Texto da notícia:
\"\"\"{text}\"\"\"

Responda APENAS com o JSON válido, sem texto adicional.
"""
        # ==============================================================================

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"LLM timeout após {LLM_TIMEOUT_SECONDS}s")

        signal.signal(signal.SIGALRM, _timeout_handler)

        try:
            start_time = time.time()
            tamanho_prompt = len(prompt_content)
            tamanho_texto = len(text)

            if not return_metrics:
                print(f"[DEBUG] Enviando prompt ao LLM ({tamanho_prompt} chars)...", flush=True)

            signal.alarm(LLM_TIMEOUT_SECONDS)
            try:
                response = self.llm.invoke([HumanMessage(content=prompt_content)])
            finally:
                signal.alarm(0)

            tempo_inferencia = time.time() - start_time
            tamanho_resposta = len(response.content)

            if not return_metrics:
                print(f"[DEBUG] Resposta recebida em {tempo_inferencia:.1f}s", flush=True)

            result = response.content.strip()
            atributos = self._parse_json_response(result, default_return)

            if return_metrics:
                return {
                    "atributos": atributos,
                    "metricas": {
                        "tempo_inferencia_s": round(tempo_inferencia, 2),
                        "tamanho_texto_chars": tamanho_texto,
                        "tamanho_prompt_chars": tamanho_prompt,
                        "tamanho_resposta_chars": tamanho_resposta,
                        "modelo": SELECTED_MODEL,
                        "temperatura": LLM_TEMPERATURE,
                    }
                }
            else:
                return atributos

        except Exception as e:
            signal.alarm(0)
            print(f"[Erro na Extração] {type(e).__name__}: {e}")
            if return_metrics:
                return {
                    "atributos": default_return,
                    "metricas": {"erro": str(e)}
                }
            return default_return

    def _parse_json_response(self, result_str: str, default_return: Dict) -> Dict:
        """
        Limpa e valida o JSON retornado pelo modelo.
        """
        # Limpeza de blocos de código Markdown (comum em LLMs)
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        
        result_str = result_str.strip()

        try:
            data = json.loads(result_str)
            
            # Garante a estrutura de saída
            out = {}
            
            # Processa campos de lista (municipio, modalidade, edital, objeto)
            for key in ["municipio", "modalidade", "edital", "objeto"]:
                value = data.get(key, [])
                
                # Força formato de lista de strings
                if isinstance(value, str):
                    value = [value] if value.strip() else []
                elif isinstance(value, (int, float)):
                    value = [str(value)]
                elif not isinstance(value, list):
                    value = []

                # Remove duplicatas e strings vazias
                clean_list = []
                seen = set()
                for item in value:
                    s = str(item).strip()
                    # Remove aspas duplas ou simples no início e fim
                    s = s.strip('"').strip("'").strip()
                    if s and s not in seen:
                        clean_list.append(s)
                        seen.add(s)
                
                out[key] = clean_list

            # Aplica normalização específica de editais
            out['edital'] = [normalize_edital(e) for e in out.get('edital', []) if e]
            
            # Processa campos de string (ano_inicio, ano_fim)
            for key in ["ano_inicio", "ano_fim"]:
                value = data.get(key, "")
                
                # Força formato de string
                if isinstance(value, (int, float)):
                    value = str(value)
                elif isinstance(value, list):
                    # Se vier como lista, pega o primeiro elemento
                    value = str(value[0]) if value else ""
                elif not isinstance(value, str):
                    value = ""
                
                # Remove aspas duplas ou simples no início e fim
                value = str(value).strip().strip('"').strip("'").strip()
                
                out[key] = value

            return out

        except json.JSONDecodeError:
            print(f"Falha ao decodificar JSON. Início da resposta: {result_str[:50]}...")
            return default_return
