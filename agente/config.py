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
GEMINI_MODEL = "gemini-1.5-flash"

# Prompt para o modelo
CAPTION_PROMPT = """Você verá 3 frames de um mesmo segmento de vídeo: início, meio e fim.

Gere uma única legenda em português que descreva o que está acontecendo nesse segmento.

CRITÉRIOS DE QUALIDADE (sua legenda será avaliada por):
- Relevância: descreve o conteúdo principal do segmento
- Coerência: a descrição faz sentido entre os 3 frames
- Concisão: informação clara em 2-3 frases
- Completude: cobre ações, objetos e contexto relevantes

REGRAS:
- Seja factual e objetivo
- NÃO invente detalhes que não estão visíveis
- Descreva ações, objetos e contexto
- Use 2-3 frases no máximo
- NÃO use formatação markdown

Legenda:"""
