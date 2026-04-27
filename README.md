# Video Captioning — TCC

Pipeline para geração e avaliação automática de legendas de vídeo usando modelos de IA multimodal no dataset ActivityNet.

## Estrutura do Projeto

```
tcc/
├── src/
│   ├── agent/          # Pipeline de geração de legendas
│   │   ├── config.py   # Configurações e tokens da API
│   │   └── main.py     # Agente principal (download → frames → legenda → JSON)
│   └── evaluation/     # Pipeline de avaliação
│       ├── llm_eval.py     # Avaliação ACCR com LLM como avaliador
│       └── auto_metrics.py # Métricas automáticas (BLEU, METEOR, ROUGE-L, CIDEr)
├── data/
│   ├── ground_truth/   # Anotações do ActivityNet (anet_entities_test_1/2.json)
│   ├── baselines/      # Predições do modelo SkimCap para comparação
│   ├── videos_com_urls.json      # Mapeamento video_id → URL do YouTube
│   └── videos_disponiveis.json   # Lista dos 10 vídeos disponíveis para teste
├── scripts/
│   └── get_videos.py   # Filtra vídeos disponíveis no YouTube (50–120s)
├── output/             # Saídas geradas (gitignored)
│   └── predictions.json
├── .env.example        # Template de variáveis de ambiente
└── requirements.txt
```

## Fluxo de Uso

### 1. Configuração

```bash
cp .env.example .env
# Preencha GITHUB_TOKEN com seu token do GitHub Models
pip install -r requirements.txt
```

### 2. Geração de Legendas

```bash
python src/agent/main.py
```

Lê `data/videos_disponiveis.json`, baixa os vídeos do YouTube, extrai frames e gera legendas usando GitHub Models. Salva em `output/predictions.json`.

### 3. Avaliação

```bash
# Avaliação ACCR (LLM como avaliador)
python src/evaluation/llm_eval.py

# Métricas automáticas (BLEU, METEOR, ROUGE-L, CIDEr)
python src/evaluation/auto_metrics.py
```

### 4. Descoberta de Vídeos (opcional)

```bash
python scripts/get_videos.py
```

Verifica disponibilidade no YouTube e atualiza `data/videos_disponiveis.json`.

## Modelos Suportados

Configurados em `src/agent/config.py` via `PROVIDER`:

| Provider | Modelo |
|---|---|
| `github_gpt4o` | openai/gpt-4.1 |
| `github_llama` | meta/Llama-4-Maverick-17B-128E-Instruct-FP8 |
| `github_phi` | microsoft/Phi-4-multimodal-instruct |

## Métricas de Avaliação

- **ACCR** (LLM-based): Accuracy, Completeness, Conciseness, Relevance (0–100)
- **Automáticas**: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr-D, R@4

## Pré-requisitos

- Python 3.8+
- FFmpeg instalado e no PATH
- Token do [GitHub Models](https://github.com/marketplace/models)
