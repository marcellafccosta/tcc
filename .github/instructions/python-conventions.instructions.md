---
description: "Use when writing or editing Python code in this project. Covers language conventions, file structure, naming, configuration, and docstring patterns."
applyTo: "**/*.py"
---
# Python Conventions

## Language
- Write all **docstrings and inline comments in Portuguese**
- Class names and third-party identifiers remain in English (`VideoCaptioningAgent`, `ChatCompletionsClient`)
- Method and function names use Portuguese for domain logic (`baixar_video`, `extrair_frames`, `gerar_legenda`, `carregar_json`)
- Internal/private attributes may be English (`retry_total`, `endpoint`)

## File Header
Every module starts with a triple-quoted docstring describing its purpose and architecture:

```python
"""
Nome do módulo

Breve descrição do que este módulo faz.

ARQUITETURA:
Módulo 1: Ingestão       → baixar_video()
Módulo 2: Processamento  → extrair_frames()

FLUXO:
Entrada → Passo 1 → Passo 2 → Saída
"""
```

## Section Separators
Use `# ─────────────────────────────────────────────────────────────` (Unicode box-drawing dashes) to separate logical sections inside a file. Add a label comment on the next line:

```python
# ─────────────────────────────────────────────────────────────
# Módulo 1: Ingestão
# ─────────────────────────────────────────────────────────────
```

## Configuration
- All constants and credentials live in `src/agent/config.py`, loaded via `python-dotenv`
- **Never hardcode tokens, API keys, or endpoints** — always read from `os.getenv()`
- Import config as `import config` (not from individual variables)

## Type Hints
Use type hints for method return types and non-obvious parameters:

```python
def _model_name(self) -> str: ...
def _github_disponivel(self) -> bool: ...
def _rotulos_temporais(self, n: int) -> list: ...
```

## Paths
- Prefer `pathlib.Path` over `os.path` for new path constructions
- Use `os.makedirs(path, exist_ok=True)` when creating directories

## Error Handling
- Catch `azure.core.exceptions.HttpResponseError` for Azure AI calls
- Use `try/except` with a descriptive Portuguese `print` on failure; return `None` on recoverable errors
- Never swallow exceptions silently

## Progress Output
Use `print()` with emoji prefix for user-facing progress messages:
- `✅` success
- `⚠️` warning / rate limit
- `❌` error
- `📹`, `🎞️`, `🖼️` for media operations
