# Relatório Comparativo Final

> Gerado em: 28/04/2026 10:28  
> Modelos: **GPT-4.1** × **LLaMA-4** × **SkimCap**  
> Ground truths: `anet_entities_test_1` e `anet_entities_test_2`

---

## Resumo por Modelo

> Valores médios entre os dois arquivos de ground truth.

### Métricas Automáticas — Média

| Modelo | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |
|---|---:|---:|---:|---:|---:|
| **GPT-4.1** | 0.002 | 0.021 | 0.194 | 0.227 | 0.050 |
| **LLaMA-4** | 0.030 | 0.030 | 0.195 | 0.265 | 0.065 |
| **SkimCap** | 0.376 | 0.091 | 0.318 | 0.304 | 0.050 |

### ACCR — Média (escala 0–100)

| Modelo | Accuracy | Completeness | Conciseness | Relevance | Média Geral |
|---|---:|---:|---:|---:|---:|
| **GPT-4.1** | 57.4 | 52.9 | 76.4 | 61.4 | 62.0 |
| **LLaMA-4** | 50.5 | 47.4 | 75.2 | 55.5 | 57.1 |
| **SkimCap** | 27.0 | 23.9 | 69.1 | 32.2 | 38.1 |

---

## Detalhamento por Ground Truth

### Tabela 1 — Métricas Automáticas por GT

| Modelo / GT | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 (anet_entities_test_1) | 0.001 | 0.015 | 0.197 | 0.237 | 0.050 |
| GPT-4.1 (anet_entities_test_2) | 0.004 | 0.026 | 0.192 | 0.216 | 0.050 |
| | | | | | |
| LLaMA-4 (anet_entities_test_1) | 0.002 | 0.027 | 0.199 | 0.283 | 0.065 |
| LLaMA-4 (anet_entities_test_2) | 0.058 | 0.033 | 0.190 | 0.248 | 0.065 |
| | | | | | |
| SkimCap (anet_entities_test_1) | 0.409 | 0.122 | 0.353 | 0.352 | 0.050 |
| SkimCap (anet_entities_test_2) | 0.343 | 0.061 | 0.283 | 0.255 | 0.050 |

### Tabela 2 — ACCR por GT (escala 0–100)

| Modelo / GT | Accuracy | Completeness | Conciseness | Relevance | Média |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 (anet_entities_test_1) | 57.9 | 52.9 | 78.3 | 62.3 | 62.8 |
| GPT-4.1 (anet_entities_test_2) | 57.0 | 52.8 | 74.6 | 60.6 | 61.2 |
| | | | | | |
| LLaMA-4 (anet_entities_test_1) | 53.3 | 50.1 | 76.7 | 58.4 | 59.6 |
| LLaMA-4 (anet_entities_test_2) | 47.6 | 44.7 | 73.7 | 52.6 | 54.6 |
| | | | | | |
| SkimCap (anet_entities_test_1) | 27.5 | 25.0 | 68.2 | 32.8 | 38.4 |
| SkimCap (anet_entities_test_2) | 26.5 | 22.8 | 70.0 | 31.7 | 37.8 |

### Tabela 3 — Ranking por Posição (🥇 = melhor, R@4 ↓ = menor é melhor)

| Modelo / GT | CIDEr | BLEU-4 | ROUGE-L | METEOR | R@4 | Accuracy | Completeness | Conciseness | Relevance | ACCR | Rank Médio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| GPT-4.1 (anet_entities_test_1) | 6 | 6 | 4 | 5 | 🥉 | 🥇 | 🥇 | 🥇 | 🥇 | 🥇 | 2.90 |
| GPT-4.1 (anet_entities_test_2) | 4 | 5 | 5 | 6 | 4 | 🥈 | 🥈 | 🥉 | 🥈 | 🥈 | 3.50 |
| |  |  |  |  |  |  |  |  |  |  | |
| LLaMA-4 (anet_entities_test_1) | 5 | 4 | 🥉 | 🥈 | 5 | 🥉 | 🥉 | 🥈 | 🥉 | 🥉 | 3.30 |
| LLaMA-4 (anet_entities_test_2) | 🥉 | 🥉 | 6 | 4 | 6 | 4 | 4 | 4 | 4 | 4 | 4.20 |
| |  |  |  |  |  |  |  |  |  |  | |
| SkimCap (anet_entities_test_1) | 🥇 | 🥇 | 🥇 | 🥇 | 🥇 | 5 | 5 | 6 | 5 | 5 | 3.10 |
| SkimCap (anet_entities_test_2) | 🥈 | 🥈 | 🥈 | 🥉 | 🥈 | 6 | 6 | 5 | 6 | 6 | 4.00 |

---

## Conclusão

### Melhores por Métrica Automática

| Métrica | Melhor Modelo | Valor |
|---|---|---:|
| CIDEr-D | SkimCap (anet_entities_test_1) | 0.409 |
| BLEU-4 | SkimCap (anet_entities_test_1) | 0.122 |
| ROUGE-L | SkimCap (anet_entities_test_1) | 0.353 |
| METEOR | SkimCap (anet_entities_test_1) | 0.352 |
| R@4 ↓ (menor = melhor) | SkimCap (anet_entities_test_1) | 0.050 |

### Melhores por ACCR

| Dimensão | Melhor Modelo | Score |
|---|---|---:|
| Accuracy | GPT-4.1 (anet_entities_test_1) | 57.9/100 |
| Completeness | GPT-4.1 (anet_entities_test_1) | 52.9/100 |
| Conciseness | GPT-4.1 (anet_entities_test_1) | 78.3/100 |
| Relevance | GPT-4.1 (anet_entities_test_1) | 62.3/100 |
| ACCR Média Geral | GPT-4.1 (anet_entities_test_1) | 62.8/100 |

### Análise

Nas métricas automáticas (CIDEr-D, BLEU-4, ROUGE-L, METEOR), **LLaMA-4** obteve médias superiores a **GPT-4.1** (CIDEr médio: 0.030 vs 0.002).

O baseline SkimCap dominou as métricas automáticas (CIDEr médio: 0.376 vs 0.030). Isso é esperado: o SkimCap foi treinado diretamente no dataset ActivityNet Entities, favorecendo sobreposição lexical com o ground truth.

Na avaliação ACCR (LLM como avaliador), **GPT-4.1** obteve média de 62.0/100 contra 57.1/100 de **LLaMA-4**.

Diferentemente das métricas automáticas, na avaliação ACCR os modelos gerativos (GPT-4.1: 62.0/100) superaram o SkimCap (38.1/100). Isso evidencia a dissociação entre sobreposição lexical e qualidade semântica: modelos generativos produzem legendas semanticamente melhores, mas com vocabulário diferente do ground truth, penalizando-os nas métricas automáticas.

Há uma divergência entre os critérios: **LLaMA-4** liderou as métricas automáticas enquanto **GPT-4.1** foi superior na avaliação ACCR. A escolha do melhor modelo depende da prioridade — fidelidade lexical ao corpus ou qualidade semântica percebida.
