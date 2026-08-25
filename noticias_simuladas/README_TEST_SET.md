# Test Set (30%) - Notícias Sintéticas para Experimento de Linkage

**Data de criação:** 28 de maio de 2026  
**Objetivo:** Criar conjunto de teste (30%) para avaliação do pipeline de linkage RAG

---

## 📊 Estrutura do Split 70/30

| Conjunto | Tamanho | Percentual | Status |
|----------|---------|------------|--------|
| **Training Set** | 500 notícias | 70% | ✅ Existente |
| **Test Set** | 214 notícias | 30% | 🔄 Em geração |
| **Total** | 714 notícias | 100% | - |

**Proporção:** 500/214 = 2.33 (correto para split 70/30)

---

## 📁 Arquivos Gerados

### **Test Set (30%)**

```
/home/paulo/projects/old_main/PAULO/datasets/500_noticias_simuladas/
├── 214_noticias_test.csv                    # Notícias reais selecionadas
├── 214_processos_test.csv                   # Processos licitatórios selecionados
├── 214_noticias_sinteticas_test.csv         # Notícias sintéticas (output)
├── mapeamento_pares_test.json               # Mapeamento para avaliação
├── gerar_noticias_sinteticas_test.py        # Script de geração
├── geracao_log_test.txt                     # Log de execução
├── status_execucao_test.json                # Status em tempo real
└── erros_geracao_test.json                  # Erros (se houver)
```

### **Training Set (70%) - Existente**

```
/home/paulo/projects/old_main/PAULO/datasets/500_noticias_simuladas/
├── 500_noticias_aleatorias.csv
├── 500_processos_aleatorios.csv
├── 500_noticias_sinteticas.csv
├── mapeamento_pares.json
└── gerar_noticias_sinteticas.py
```

---

## 🔄 Processo de Criação

### **Etapa 1: Amostragem de Notícias ✅**

- **Origem:** `442_noticias_nao_usadas.csv` (448 notícias disponíveis)
- **Método:** Amostragem aleatória com seed=42
- **Resultado:** `214_noticias_test.csv` (214 notícias)
- **Colunas:** `[titulo, texto_noticia]`

### **Etapa 2: Amostragem de Processos ✅**

- **Origem:** `325K_processo_licitatorio_with_relations_v2.csv` (325,781 processos)
- **Filtro:** Excluir 500 processos usados no training set
- **Processos disponíveis:** 325,281 (após exclusão)
- **Método:** Amostragem aleatória com seed=42
- **Resultado:** `214_processos_test.csv` (214 processos)
- **Colunas:** `[id_processo_licitatorio, numero_edital, modalidade, unidade_gestora, ente, municipio, uf, descricao_objeto]`
- **Verificação:** ✅ Nenhum overlap com training set (0 IDs em comum)

### **Etapa 3: Geração de Notícias Sintéticas 🔄**

- **Script:** `gerar_noticias_sinteticas_test.py`
- **LLM:** gpt-oss:20b via Ollama (configure OLLAMA_HOST env var, default: http://localhost:11434)
- **Temperatura:** 0.3
- **Processo:** Reescrever cada notícia injetando atributos do processo correspondente
- **Pareamento:** 1:1 (noticia[i] + processo[i] → sintetica[i])
- **Status:** Em execução (iniciado às 14:50, 28/05/2026)
- **Teste:** ✅ 5 pares gerados com sucesso (6.1 min, 100% sucesso)

---

## ⚠️ Garantias de Qualidade

### **1. Sem Vazamento de Dados (Data Leakage)**

✅ **Notícias:**
- Test set: `442_noticias_nao_usadas.csv` (notícias NÃO usadas no training)
- Training set: `500_noticias_aleatorias.csv`
- **Overlap:** 0 notícias em comum

✅ **Processos Licitatórios:**
- Test set: 214 processos de `325K_processo_licitatorio_with_relations_v2.csv`
- Training set: 500 processos de `500_processos_aleatorios.csv`
- **Overlap:** 0 processos em comum (verificado por `id_processo_licitatorio`)

### **2. Reprodutibilidade**

- **Seed fixo:** 42 (usado em todas as amostragens)
- **Versionamento:** Arquivos de entrada documentados
- **Logs:** Todos os passos registrados

### **3. Consistência de Estrutura**

- Mesma estrutura de colunas do training set
- Pareamento 1:1 entre notícias e processos
- Mesmo formato de saída (CSV + JSON)

---

## 📈 Estimativas de Tempo

### **Teste com 5 pares:**
- Tempo total: 6.1 minutos
- Tempo médio por par: ~73 segundos
- Taxa de sucesso: 100%

### **Estimativa para 214 pares:**
- Tempo estimado: ~4.3 horas (214 × 73s ÷ 60)
- Com overhead: ~5 horas

---

## 🎯 Uso do Test Set

### **Para Avaliação do Pipeline RAG:**

```python
import pandas as pd
import json

# Carregar test set
df_test = pd.read_csv('214_noticias_sinteticas_test.csv')

# Carregar mapeamento (ground truth)
with open('mapeamento_pares_test.json', 'r') as f:
    ground_truth = json.load(f)

# Para cada notícia sintética:
for i, row in df_test.iterrows():
    noticia_sintetica = row['texto_sintetico']
    id_gold = row['id_processo_licitatorio']
    
    # Executar pipeline RAG
    id_recuperado = pipeline_rag(noticia_sintetica)
    
    # Avaliar
    acerto = (id_recuperado == id_gold)
```

### **Métricas Esperadas:**

- **Recall@1:** % de notícias onde o processo correto está em 1º lugar
- **Recall@k:** % de notícias onde o processo correto está no top-k
- **MRR (Mean Reciprocal Rank):** Média do inverso da posição do processo correto
- **Match Rate:** % de notícias onde o processo correto foi recuperado

---

## 📝 Comandos Úteis

### **Monitorar execução em tempo real:**

```bash
# Ver status
cat status_execucao_test.json | jq

# Ver log
tail -f geracao_log_test.txt

# Verificar progresso
wc -l 214_noticias_sinteticas_test.csv
```

### **Retomar execução (se interrompida):**

```bash
# Retomar a partir do último par salvo
python3 gerar_noticias_sinteticas_test.py --start-from <N>
```

### **Teste rápido:**

```bash
# Gerar apenas 10 pares para teste
python3 gerar_noticias_sinteticas_test.py --limit 10
```

---

## ✅ Checklist de Validação

- [x] Notícias do test set são diferentes do training set
- [x] Processos do test set são diferentes do training set
- [x] Seed fixo para reprodutibilidade (seed=42)
- [x] Estrutura de arquivos consistente com training set
- [x] Script de geração adaptado e testado
- [ ] Geração completa de 214 pares (em andamento)
- [ ] Verificação final de qualidade
- [ ] Documentação de resultados

---

## 🔗 Referências

- **Training set:** `/home/paulo/projects/old_main/PAULO/datasets/500_noticias_simuladas/`
- **Fonte de notícias:** `/home/paulo/projects/old_main/PAULO/datasets/983_test_set/Clean_to_json/442_noticias_nao_usadas.csv`
- **Fonte de processos:** `/home/paulo/projects/old_main/PAULO/datasets/LICITAÇÕES/325K_processo_licitatorio_with_relations_v2.csv`
- **Script original:** `gerar_noticias_sinteticas.py`
- **Script test set:** `gerar_noticias_sinteticas_test.py`
