"""
Configurações do agente de captioning
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ─── Provider ────────────────────────────────────────────────────
# Opções: "github_gpt41" | "github_llama" | "github_phi"
PROVIDER           = "github_gpt41"   # primeiro modelo de geração
PROVIDER_2         = "github_llama"   # segundo modelo de geração (comparação)
EVALUATOR_PROVIDER = "github_gpt41"   # modelo de avaliação ACCR (llm_eval.py)


# ─── GitHub Models (único token para todos os modelos) ────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
GITHUB_TOKEN_2  = os.getenv("GITHUB_TOKEN_2", "")  # token alternativo (dobra a cota)
GITHUB_TOKEN_3  = os.getenv("GITHUB_TOKEN_3", "")  # terceiro token
GITHUB_TOKEN_4  = os.getenv("GITHUB_TOKEN_4", "")  # quarto token
GITHUB_TOKEN_5  = os.getenv("GITHUB_TOKEN_5", "")  # quinto token
GITHUB_TOKEN_6  = os.getenv("GITHUB_TOKEN_6", "")  # sexto token
GITHUB_TOKEN_7  = os.getenv("GITHUB_TOKEN_7", "")  # sétimo token (350 req/dia total)
GITHUB_ENDPOINT = "https://models.github.ai/inference"

GITHUB_GPT41    = "openai/gpt-4.1"
GITHUB_LLAMA    = "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
GITHUB_PHI      = "microsoft/Phi-4-multimodal-instruct"

# Pastas do projeto
VIDEOS_DIR = "output/videos"
FRAMES_DIR = "output/frames"
OUTPUT_DIR = "output"

# Timeout em segundos para chamadas à API
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# Prompt para o modelo — carregado do arquivo externo para não misturar conteúdo com config
_PROMPT_FILE = Path(__file__).parent / "prompts" / "caption.txt"
CAPTION_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8")
