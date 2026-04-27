"""
Configurações do agente de captioning
"""
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ─── Provider ────────────────────────────────────────────────────
# Opções: "github_gpt4o" | "github_llama" | "github_phi"
PROVIDER           = "github_llama"  # usado pelo main.py (geração)
EVALUATOR_PROVIDER = "github_gpt4o"  # usado pelo avaliar_legendas.py (avaliação)


# ─── GitHub Models (único token para todos os modelos) ────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
GITHUB_TOKEN_2  = os.getenv("GITHUB_TOKEN_2", "")  # token alternativo (dobra a cota)
GITHUB_TOKEN_3  = os.getenv("GITHUB_TOKEN_3", "")  # terceiro token (150 req/dia total)
GITHUB_ENDPOINT = "https://models.github.ai/inference"

GITHUB_GPT4O    = "openai/gpt-4.1"
GITHUB_LLAMA    = "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
GITHUB_PHI      = "microsoft/Phi-4-multimodal-instruct"

# Pastas do projeto
VIDEOS_DIR = "videos"
FRAMES_DIR = "frames"
OUTPUT_DIR = "output"

# Prompt para o modelo
CAPTION_PROMPT = """You will see a sequence of frames from the same video segment, from start to end.

Generate a single caption in English that describes what is happening in this segment.

IMPORTANT: Your caption will be automatically evaluated by the following metrics and MUST be satisfactory in all dimensions:
• CIDEr-D: Semantic alignment via TF-IDF → Accuracy + Relevance
• BLEU-4: Precision of 4-word sequences → Accuracy
• ROUGE-L: Coverage via longest common subsequence → Completeness
• METEOR: Alignment with synonym support → Accuracy + Completeness
• R@4: 4-gram repetition between segments → Conciseness

Your caption MUST satisfy the ACCR framework:
- Accuracy: factual and precise descriptions
- Completeness: covers all relevant actions, objects, and context
- Conciseness: clear and efficient (2-3 sentences maximum)
- Relevance: describes the main content of the segment
- Coherence: makes sense across all 3 frames

RULES:
- Be factual and objective
- DO NOT invent details that are not visible
- Describe actions, objects, and context
- Use at most 2-3 sentences
- DO NOT use markdown formatting

Caption:"""
