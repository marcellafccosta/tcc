---
description: "Use when writing code that calls Azure AI Inference, GitHub Models, or manages API tokens and rate limits. Covers client setup, token rotation, model selection, and request patterns."
---
# Azure AI Inference & GitHub Models

## Client Setup
Always instantiate `ChatCompletionsClient` with `retry_total=0` so rate-limit errors surface immediately instead of being silently retried:

```python
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

client = ChatCompletionsClient(
    endpoint=config.GITHUB_ENDPOINT,
    credential=AzureKeyCredential(token),
    retry_total=0,
)
```

## Token Rotation
When multiple API tokens are available (`GITHUB_TOKEN`, `GITHUB_TOKEN_2`, `GITHUB_TOKEN_3`):
- Store clients in a list `_github_clients` and call counts in `_github_calls`
- Keep an active index `_token_idx`
- Rotate to the next token when the current one reaches `_github_daily_limit`
- Always check availability via a `_github_disponivel()` guard before making a call

## Model Selection
Model names are constants in `config.py`. Never hardcode model strings inline:

```python
# config.py
GITHUB_GPT41 = "openai/gpt-4.1"
GITHUB_LLAMA = "meta/Llama-4-Maverick-17B-128E-Instruct-FP8"
GITHUB_PHI   = "microsoft/Phi-4-multimodal-instruct"
```

Select via a dict lookup, falling back to GPT-4o:
```python
model = {
    "github_gpt41":  config.GITHUB_GPT41,
    "github_llama":  config.GITHUB_LLAMA,
    "github_phi":    config.GITHUB_PHI,
}.get(self.provider, config.GITHUB_GPT41)
```

## Message Format for Vision
Pass images as `ImageContentItem` with `ImageUrl` (base64 data URL for local files):

```python
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem, UserMessage

UserMessage(content=[
    TextContentItem(text="...prompt..."),
    ImageContentItem(image_url=ImageUrl(url="data:image/jpeg;base64,...")),
])
```

## Error Handling
Wrap every API call:

```python
from azure.core.exceptions import HttpResponseError

try:
    response = client.complete(...)
except HttpResponseError as e:
    print(f"  ❌ Erro na API: {e.status_code} — {e.message}")
    return None
```

## Rate Limit Awareness
- `PROVIDER` in `config.py` controls which model is used for **generation** (`main.py`)
- `EVALUATOR_PROVIDER` controls which model is used for **evaluation** (`llm_eval.py`)
- Do not share the same token pool between generation and evaluation
