# Relatório Comparativo Final

> Gerado em: 02/06/2026 11:05  
> Modelos: **GPT-4.1** × **LLaMA** × **SkimCap**  
> Ground truths: `anet_entities_test_1` e `anet_entities_test_2`

---

## Resumo por Modelo

> Valores médios entre os dois arquivos de ground truth.

### Métricas Automáticas — Média

| Modelo | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |
|---|---:|---:|---:|---:|---:|
| **GPT-4.1** | 0.018 | 0.026 | 0.247 | 0.263 | 0.003 |
| **LLaMA** | 0.112 | 0.024 | 0.224 | 0.253 | 0.008 |
| **SkimCap** | 0.376 | 0.091 | 0.318 | 0.304 | 0.050 |

### ACCR — Média (escala 0–100)

| Modelo | Accuracy | Completeness | Conciseness | Relevance | Média Geral |
|---|---:|---:|---:|---:|---:|
| **GPT-4.1** | 49.8 | 44.8 | 79.5 | 55.6 | 57.4 |
| **LLaMA** | 43.1 | 38.8 | 78.4 | 48.5 | 52.2 |
| **SkimCap** | 27.2 | 23.5 | 67.7 | 32.3 | 37.7 |

---

## Detalhamento por Ground Truth

### Tabela 1 — Métricas Automáticas por GT

| Modelo / GT | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 (anet_entities_test_1) | 0.030 | 0.026 | 0.247 | 0.259 | 0.003 |
| GPT-4.1 (anet_entities_test_2) | 0.006 | 0.025 | 0.248 | 0.267 | 0.003 |
| | | | | | |
| LLaMA (anet_entities_test_1) | 0.173 | 0.022 | 0.218 | 0.255 | 0.008 |
| LLaMA (anet_entities_test_2) | 0.051 | 0.026 | 0.231 | 0.252 | 0.008 |
| | | | | | |
| SkimCap (anet_entities_test_1) | 0.409 | 0.122 | 0.353 | 0.352 | 0.050 |
| SkimCap (anet_entities_test_2) | 0.343 | 0.061 | 0.283 | 0.255 | 0.050 |

### Tabela 2 — ACCR por GT (escala 0–100)

| Modelo / GT | Accuracy | Completeness | Conciseness | Relevance | Média |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 (anet_entities_test_1) | 49.2 | 44.8 | 78.9 | 55.4 | 57.1 |
| GPT-4.1 (anet_entities_test_2) | 50.4 | 44.7 | 80.2 | 55.8 | 57.8 |
| | | | | | |
| LLaMA (anet_entities_test_1) | 42.8 | 37.1 | 78.4 | 48.0 | 51.6 |
| LLaMA (anet_entities_test_2) | 43.3 | 40.6 | 78.5 | 49.0 | 52.9 |
| | | | | | |
| SkimCap (anet_entities_test_1) | 28.5 | 24.5 | 68.5 | 35.0 | 39.1 |
| SkimCap (anet_entities_test_2) | 26.0 | 22.5 | 66.8 | 29.7 | 36.2 |

### Tabela 3 — Ranking por Posição (🥇 = melhor, R@4 ↓ = menor é melhor)

| Modelo / GT | CIDEr | BLEU-4 | ROUGE-L | METEOR | R@4 | Accuracy | Completeness | Conciseness | Relevance | ACCR | Rank Médio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| GPT-4.1 (anet_entities_test_1) | 5 | 4 | 4 | 🥉 | 🥇 | 🥈 | 🥇 | 🥈 | 🥈 | 🥈 | 2.60 |
| GPT-4.1 (anet_entities_test_2) | 6 | 5 | 🥉 | 🥈 | 🥈 | 🥇 | 🥈 | 🥇 | 🥇 | 🥇 | 2.40 |
| |  |  |  |  |  |  |  |  |  |  | |
| LLaMA (anet_entities_test_1) | 🥉 | 6 | 6 | 5 | 🥉 | 4 | 4 | 4 | 4 | 4 | 4.30 |
| LLaMA (anet_entities_test_2) | 4 | 🥉 | 5 | 6 | 4 | 🥉 | 🥉 | 🥉 | 🥉 | 🥉 | 3.70 |
| |  |  |  |  |  |  |  |  |  |  | |
| SkimCap (anet_entities_test_1) | 🥇 | 🥇 | 🥇 | 🥇 | 5 | 5 | 5 | 5 | 5 | 5 | 3.40 |
| SkimCap (anet_entities_test_2) | 🥈 | 🥈 | 🥈 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 4.60 |

---

## Conclusão

### Melhores por Métrica Automática

| Métrica | Melhor Modelo | Valor |
|---|---|---:|
| CIDEr-D | SkimCap (anet_entities_test_1) | 0.409 |
| BLEU-4 | SkimCap (anet_entities_test_1) | 0.122 |
| ROUGE-L | SkimCap (anet_entities_test_1) | 0.353 |
| METEOR | SkimCap (anet_entities_test_1) | 0.352 |
| R@4 ↓ (menor = melhor) | GPT-4.1 (anet_entities_test_1) | 0.003 |

### Melhores por ACCR

| Dimensão | Melhor Modelo | Score |
|---|---|---:|
| Accuracy | GPT-4.1 (anet_entities_test_2) | 50.4/100 |
| Completeness | GPT-4.1 (anet_entities_test_1) | 44.8/100 |
| Conciseness | GPT-4.1 (anet_entities_test_2) | 80.2/100 |
| Relevance | GPT-4.1 (anet_entities_test_2) | 55.8/100 |
| ACCR Média Geral | GPT-4.1 (anet_entities_test_2) | 57.8/100 |

### Análise

Nas métricas automáticas (CIDEr-D, BLEU-4, ROUGE-L, METEOR), **LLaMA** obteve médias superiores a **GPT-4.1** (CIDEr médio: 0.112 vs 0.018).

O baseline SkimCap dominou as métricas automáticas (CIDEr médio: 0.376 vs 0.112). Isso é esperado: o SkimCap foi treinado diretamente no dataset ActivityNet Entities, favorecendo sobreposição lexical com o ground truth.

Na avaliação ACCR (LLM como juiz), **GPT-4.1** obteve média de 57.4/100 contra 52.2/100 de **LLaMA**.

Diferentemente das métricas automáticas, na avaliação ACCR os modelos gerativos (GPT-4.1: 57.4/100) superaram o SkimCap (37.7/100). Isso evidencia a dissociação entre sobreposição lexical e qualidade semântica: modelos generativos produzem legendas semanticamente melhores, mas com vocabulário diferente do ground truth, penalizando-os nas métricas automáticas.

Há uma divergência entre os critérios: **LLaMA** liderou as métricas automáticas enquanto **GPT-4.1** foi superior na avaliação ACCR. A escolha do melhor modelo depende da prioridade — fidelidade lexical ao corpus ou qualidade semântica percebida.
