# Video Captioning — TCC

Pipeline para geração e avaliação automática de legendas de vídeo usando modelos de IA multimodal no dataset ActivityNet Entities.

Dois modelos de geração (GPT-4.1 e LLaMA-4) são comparados entre si e contra o baseline SkimCap, avaliados por métricas automáticas e por um LLM avaliador (ACCR).

## Estrutura do Projeto

```
tcc/
├── pipeline.py              # Orquestrador principal (geração + ACCR + auto-métricas)
├── src/
│   ├── agent/
│   │   ├── config.py        # Configurações, credenciais e nomes dos modelos
│   │   ├── main.py          # Agente de geração (download → frames → legenda → JSON)
│   │   ├── token_manager.py # Rotação automática de tokens GitHub Models
│   │   └── prompts/
│   │       └── caption.txt  # Prompt de geração de legendas
│   └── evaluation/
│       ├── llm_eval.py      # Avaliação ACCR com LLM como avaliador
│       ├── auto_metrics.py  # Métricas automáticas (BLEU-4, METEOR, ROUGE-L, CIDEr-D, R@4)
│       └── prompts/
│           └── accr.txt     # Prompt do avaliador ACCR
├── data/
│   ├── ground_truth/        # Anotações do ActivityNet Entities
│   │   ├── anet_entities_test_1.json
│   │   └── anet_entities_test_2.json
│   ├── baselines/           # Predições do SkimCap para comparação
│   │   └── greedy_pred_test.json
│   ├── videos_com_urls.json       # Mapeamento video_id → URL do YouTube
│   └── videos_disponiveis.json    # IDs dos vídeos acessíveis no YouTube
├── scripts/
│   └── get_videos.py        # Verifica disponibilidade dos vídeos no YouTube
├── output/                  # Tudo gerado em runtime (ignorado pelo git)
│   ├── predictions/         # JSONs gerados pelos modelos
│   │   ├── predictions_gpt.json     # GPT-4.1
│   │   └── predictions_llama.json   # LLaMA-4
│   ├── metrics/
│   │   ├── auto/            # Métricas automáticas por modelo e GT
│   │   └── accr/            # Relatórios ACCR por modelo e GT
│   ├── videos/              # Cache de vídeos baixados (temporário)
│   └── frames/              # Cache de frames extraídos (temporário)
├── .env.example             # Template de variáveis de ambiente
└── requirements.txt
```

## Configuração

```bash
# 1. Clonar e entrar no diretório
git clone <repo> && cd tcc

# 2. Criar e ativar virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar credenciais
cp .env.example .env
# Edite .env e preencha GITHUB_TOKEN (obrigatório), GITHUB_TOKEN_2 e GITHUB_TOKEN_3 (opcionais)
```

Tokens do GitHub Models: <https://github.com/marketplace/models>  
Limite por token: 50 requisições/dia · 10 req/min.

## Uso

### Pipeline completo (recomendado)

```bash
python pipeline.py --nome1 GPT-4.1 --nome2 LLaMA
```

Executa em sequência: geração → ACCR → métricas automáticas.

**Opções úteis:**

| Flag | Descrição |
|---|---|
| `--limit N` | Processa apenas N vídeos |
| `--skip-gen` | Pula geração (usa JSONs existentes em `output/predictions/`) |
| `--skip-accr` | Pula avaliação ACCR (economiza tokens) |
| `--skip-auto` | Pula métricas automáticas |
| `--retry-nulls` | Re-processa apenas segmentos com `caption: null` no arquivo existente |
| `--no-skimcap` | Exclui o baseline SkimCap da avaliação |
| `--workers N` | Segmentos processados em paralelo (padrão: 2) |
| `--pred1 ARQUIVO` | Nome do JSON do modelo 1 (padrão: `predictions_gpt.json`) |
| `--pred2 ARQUIVO` | Nome do JSON do modelo 2 |

```bash
# Testar com 3 vídeos, sem ACCR (economiza tokens)
python pipeline.py --nome1 GPT-4.1 --nome2 LLaMA --limit 3 --skip-accr

# Re-processar apenas captions que ficaram null
python pipeline.py --nome1 GPT-4.1 --nome2 LLaMA --retry-nulls --skip-accr --skip-auto

# Apenas ACCR (geração já feita)
python pipeline.py --nome1 GPT-4.1 --nome2 LLaMA --skip-gen --skip-auto

# Apenas métricas automáticas (geração já feita)
python pipeline.py --nome1 GPT-4.1 --nome2 LLaMA --skip-gen --skip-accr
```

### Execução individual

```bash
# Apenas geração
python src/agent/main.py --output predictions/predictions_gpt.json --workers 2

# Apenas ACCR
python src/evaluation/llm_eval.py \
  --predictions output/predictions/predictions_gpt.json \
  --predictions2 output/predictions/predictions_llama.json \
  --gt data/ground_truth/anet_entities_test_1.json \
      data/ground_truth/anet_entities_test_2.json \
  --skimcap data/baselines/greedy_pred_test.json \
  --modelo-nome GPT-4.1 --modelo2-nome LLaMA

# Apenas métricas automáticas
python src/evaluation/auto_metrics.py \
  --predictions output/predictions/predictions_gpt.json \
  --predictions2 output/predictions/predictions_llama.json \
  --references data/ground_truth/anet_entities_test_1.json \
               data/ground_truth/anet_entities_test_2.json \
  --skimcap data/baselines/greedy_pred_test.json \
  --modelo-nome GPT-4.1 --modelo2-nome LLaMA
```

## Modelos

| Papel | Provider | Modelo |
|---|---|---|
| Geração (modelo 1) | `github_gpt41` | `openai/gpt-4.1` |
| Geração (modelo 2) | `github_llama` | `meta/Llama-4-Maverick-17B-128E-Instruct-FP8` |
| Avaliador ACCR | `github_phi` | `microsoft/Phi-4-multimodal-instruct` |

> O Phi-4-multimodal é usado apenas como avaliador de texto (ACCR). Ele não suporta geração a partir de frames via API GitHub Models.

## Métricas de Avaliação

**Automáticas** (calculadas contra os dois arquivos de ground truth, somente nos vídeos gerados):

| Métrica | Descrição |
|---|---|
| CIDEr-D | Alinhamento semântico via TF-IDF |
| BLEU-4 | Precisão de 4-gramas |
| ROUGE-L | Cobertura via subsequência mais longa |
| METEOR | Alinhamento com suporte a sinônimos (via nltk) |
| R@4 | Repetição de 4-gramas entre segmentos (↓ melhor) |

**LLM-based (ACCR)**: Accuracy, Completeness, Conciseness, Relevance — escala 0–100 por dimensão, avaliado pelo Phi-4.

## Pré-requisitos

- Python 3.12+
- FFmpeg no PATH (`brew install ffmpeg` · `sudo apt install ffmpeg`)
- Token do [GitHub Models](https://github.com/marketplace/models)
