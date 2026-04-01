"""
Configurações do agente de captioning
"""
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Provedor de IA: "openai" ou "gemini"
PROVIDER = os.getenv("PROVIDER", "openai")

# API Key da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Key do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Pastas do projeto
VIDEOS_DIR = "videos"
FRAMES_DIR = "frames"
OUTPUT_DIR = "output"

# Modelo OpenAI
GPT_MODEL = "gpt-4o-mini"

# Modelo Gemini
GEMINI_MODEL = "gemini-pro-vision"

# Prompt para o modelo
CAPTION_PROMPT = """You will see 3 frames from the same video segment: start, middle, and end.

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
