# Backend de Video Captioning com GPT-4 Vision

**Motor do sistema** que faz todo o processamento de vídeos: baixa, segmenta, extrai frames e gera legendas usando IA multimodal.

## 🎯 Responsabilidades do Backend

Este backend é o **motor do sistema** - ele faz a "mão de obra" toda:

### 1️⃣ Ingestão
- ✅ Recebe URL do YouTube ou vídeo local
- ✅ Valida formato e entrada
- ✅🏗️ Arquitetura dos Módulos

```
┌─────────────────────────────────────────────────────┐
│                 VideoCaptioningAgent                │
│            (Orquestrador - main.py)                 │
└──────┬──────────────────────────────────────────────┘
       │
       ├─► [1] Módulo de Ingestão
       │       └─ baixar_video()
       │         • Valida URL
       │         • Baixa com yt-dlp
       │         • Trata erros de download
       │
       ├─► [2] Módulo de Processamento
       │       └─ extrair_frames()
       │         • Segmenta vídeo
       │         • Extrai 3 frames/segmento
       │         • Usa ffmpeg
       │
       ├─► [3] Módulo de Captioning
       │       └─ gerar_legenda()
       │         • Converte frames → base64
       │         • Monta prompt
       │         • Chama GPT-4 Vision API
       │         • Valida resposta
       │
       └─► [4] Módulo de Persistência
               └─ salvar_resultados()
                 • Estrutura JSON
                 • Salva metadados
                 • Organiza output/
```

Cada módulo tem **uma responsabilidade clara** e pode ser testado/modificado independentemente.cesso

### 2️⃣ Processamento de Vídeo  
- ✅ Baixa vídeo com `yt-dlp`
- ✅ Corta em segmentos temporais
- ✅ Extrai 3 frames por segmento (início, meio, fim)
- ✅ Gerencia arquivos temporários

### 3️⃣ Captioning (IA Generativa)
- ✅ Prepara frames para o GPT-4 Vision
- ✅ Monta prompt com critérios de qualidade
- ✅ Envia requisição para OpenAI API
- ✅ Recebe e valida legendas geradas
- ✅ Trata erros de API (rate limit, quota)

### 4️⃣ Persistência
- ✅ Salva legendas em JSON estruturado
- ✅ Mantém metadados (timestamps, frames, video_id)
- ✅ Organiza output para avaliação posterior

### ❌ O que o Backend NÃO fazx

- ❌ Avaliação de qualidade (isso é feito em `avaliar_legendas.py`)
- ❌ Cálculo de métricas (BLEU, METEOR, etc)
- ❌ Comparação com Ground Truth
- ❌ Interface visual

**Separação clara**: Backend gera, avaliação compara.

## 📋 Pré-requisitos

### 1. Python 3.8+

### 2. FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Baixe do site oficial: https://ffmpeg.org/download.html

### 3. Dependências Python

```bash
pip install -r requirements.txt
```

### 4. API Key da OpenAI

1. Crie uma conta em: https://platform.openai.com/
2. Gere uma API key em: https://platform.openai.com/api-keys
3. Copie o arquivo `.env.example` para `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edite o `.env` e coloque sua API key:
   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   ```

## 🚀 Como usar

### Modo Simples (1 vídeo)

Edite o arquivo `main.py`, procure a função `exemplo_uso_simples()` e modifique:

```python
# URL do vídeo
url = "https://www.youtube.com/watch?v=SUA_URL_AQUI"

# Define os segmentos (início, fim) em segundos
segmentos = [
    (0, 5),      # Segmento 1: 0 a 5 segundos
    (5, 10),     # Segmento 2: 5 a 10 segundos
    (10, 15)     # Segmento 3: 10 a 15 segundos
]
```

Depois, descomente a linha no final do arquivo:

```python
if __name__ == "__main__":
    exemplo_uso_simples()  # ← Descomente esta linha
```

Execute:

```bash
python main.py
```

### Modo Avançado (múltiplos vídeos)

Para processar vários vídeos de uma vez, use a função `exemplo_uso_avancado()`:

```python
videos = [
    {
        "id": "video_001",
        "url": "https://www.youtube.com/watch?v=XXXXXXXXXXX",
        "segments": [(0, 5), (5, 10), (10, 15)]
    },
    {
        "id": "video_002",
        "url": "https://www.youtube.com/watch?v=YYYYYYYYYYY",
        "segments": [(0, 8), (8, 16)]
    }
]
```

### Integração com ActivityNet

Para usar com seus dados do ActivityNet:

```python
import json

# Carrega as anotações do ActivityNet
with open("../descricoes/anet_entities_test_1.json", "r") as f:
    dados = json.load(f)

agente = VideoCaptioningAgent()

# Para cada vídeo no dataset
for video_id, info in dados.items():
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Extrai os timestamps dos segmentos
    segmentos = info.get("timestamps", [])
    
    # Processa
    resultado = agente.processar_video(url, segmentos, video_id=video_id)
    
    if resultado:
        agente.salvar_resultados(resultado, f"{video_id}_captions.json")
```

## 📁 Estrutura do projeto

```
agente/
├── main.py                      # Script principal com VideoCaptioningAgent
├── config.py                    # Configurações (modelo, prompt, pastas)
├── processar_activitynet.py    # Integração com ActivityNet dataset
├── validar_ambiente.py          # Validação do ambiente
├── requirements.txt             # Dependências Python
├── .env                         # Suas credenciais (NÃO commitar!)
├── .env.example                 # Exemplo de .env
│
├── videos/                      # Vídeos baixados (temporário)
├── frames/                      # Frames extraídos (temporário)
│   └── seg_0/
│       ├── start.jpg
│       ├── mid.jpg
│       └── end.jpg
│
└── output/                      # Legendas geradas (JSON)
    └── captions_20260331_143022.json
```

## 📊 Formato da saída

O arquivo JSON gerado tem esta estrutura:

```json
{
  "video_id": "exemplo",
  "url": "https://www.youtube.com/watch?v=...",
  "timestamp": "2026-03-31T14:30:22.123456",
  "num_segments": 3,
  "segments": [
    {
      "segment_id": 0,
      "timestamps": [0, 5],
      Tratamento de Erros

O backend trata diversos cenários de falha:

### Erros de Download
- ✅ Vídeo do YouTube indisponível
- ✅ Link quebrado ou região bloqueada
- ✅ Vídeo privado ou removido
- ✅ Timeout de download

### Erros de Processamento
- ✅ Formato de vídeo não suportado
- ✅ Timestamp fora do range do vídeo
- ✅ Frame corrompido ou não extraído
- ✅ Falta de espaço em disco

### Erros de API
- ✅ API key inválida ou expirada
- ✅ Rate limit excedido (429)
- ✅ Quota de créditos esgotada
- ✅ Timeout de requisição
- ✅ Resposta vazia ou malformada

**Todos os erros são logados** e o processamento continua para os próximos vídeos quando possível.oa está caminhando em um parque...",
      "frames": [
        "frames/seg_0/start.jpg",
        "frames/seg_0/mid.jpg",
        "frames/seg_0/end.jpg"
      ]
    }
  ]
}
```

## ⚙️ Configurações

Edite `config.py` para ajustar:

- **Modelo**: `GPT_MODEL = "gpt-4o-mini"` (mais barato) ou `"gpt-4o"` (mais preciso)
- **Prompt**: Customize o `CAPTION_PROMPT` para mudar o estilo das legendas
- **Pastas**: Mude onde os arquivos são salvos

## 💰 Custos

O GPT-4 Vision cobra por imagem processada:

- **gpt-4o-mini**: ~$0.00015 por imagem (mais barato)
- **gpt-4o**: ~$0.0015 por imagem (mais preciso)

Para 1 segmento (3 frames):
- gpt-4o-mini: ~$0.00045
- gpt-4o: ~$0.0045

Para 100 segmentos:
- gpt-4o-mini: ~$0.045 (4.5 centavos)
- gpt-4o: ~$0.45 (45 centavos)

## 🔧 Solução de problemas

### Erro: "ffmpeg not found"

Instale o FFmpeg conforme instruções acima.

### Erro: "OpenAI API key not found"

Verifique se:
1. O arquivo `.env` existe
2. A variável `OPENAI_API_KEY` está definida
3. A API key está correta

### Vídeo não baixa

Alguns vídeos do YouTube podem estar indisponíveis. Verifique:
1. A URL está correta
2. O vídeo não está privado ou bloqueado
3. Você tem conexão com a internet

### Frames não são extraídos

Verifique se:
1. FFmpeg está instalado corretamente: `ffmpeg -version`
2. Os timestamps estão dentro da duração do vídeo
3. Você tem permissões de escrita na pasta

## 🎓 Para sgera legendas que podem ser usadas para:

1. **Gerar captions candidatos** para comparar com:
   - Ground Truth (anotações humanas)
   - SkimCap (modelo baseline)

2. **Experimentos**:
   - Variar número de frames (2, 3, 5?)
   - Testar diferentes modelos (gpt-4o vs gpt-4o-mini)
   - Comparar com/sem prompt engineering
   - Ajustar o prompt com diferentes critérios

3. **Análise qualitativa**: Ver onde o GPT acerta/erra

**Nota**: A avaliação com métricas (BLEU, METEOR, CIDEr, ACCR) deve ser feita em um script separado, usando os JSONs gerados pelo agente

4. **Análise qualitativa**: Ver onde o GPT acerta/erra
ossíveis melhorias

- [ ] Implementar detecção automática de cenas (em vez de segmentos fixos)
- [ ] Adicionar suporte para vídeos locais (não só YouTube)
- [ ] Variar número de frames por segmento (configurável)
- [ ] Adicionar múltiplos estratégias de sampling de frames
- [ ] Fazer batch processing paralelo (múltiplos vídeos simultaneamente)
- [ ] Salvar frames com metadados para análise posterioróprio script
- [ ] Fazer batch processing mais eficiente

## 📄 Licença

Projeto acadêmico - TCC

## 🤝 Contribuindo

Fork, melhore, faça pull request!
