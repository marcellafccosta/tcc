---
description: "Use when working on evaluation, metrics, scoring, or ACCR assessment. Covers the evaluation pipeline, JSON formats, automatic metrics (CIDEr-D, BLEU-4, ROUGE-L, METEOR, R@4), and LLM-based ACCR scoring."
---
# Evaluation Pipeline

## Two Evaluation Modes
This project has **two independent evaluation scripts** — never mix them:

| Script | Purpose | Metrics |
|--------|---------|---------|
| `src/evaluation/auto_metrics.py` | Automatic, n-gram based | CIDEr-D, BLEU-4, ROUGE-L, METEOR, R@4 |
| `src/evaluation/llm_eval.py` | LLM-as-judge | ACCR (Accuracy, Completeness, Conciseness, Relevance) |

## JSON Formats

### Predictions (`output/predictions.json`)
```json
{
  "video_id": [
    {
      "timestamp": [start, end],
      "caption": "Generated caption text."
    }
  ]
}
```

### Ground Truth (`data/ground_truth/anet_entities_test_1.json`)
Uses ActivityNet Entities format (Ranjay Krishna, 2017). The evaluator merges `test_1` and `test_2` files.

## Automatic Metrics
- Prefer `pycocoevalcap` scorers when available (same as original ANETcaptions)
- Fall back to `nltk` implementations when `pycocoevalcap` is not installed
- Always check `_COCO_OK` / `_NLTK_OK` flags before calling scorers
- R@4 (4-gram repetition) — **lower is better**; all other metrics **higher is better**

## ACCR Scoring
The ACCR prompt template uses Greek-letter delimiters to parse scores:
- `α{score}α` → Accuracy
- `β{score}β` → Completeness
- `ψ{score}ψ` → Conciseness
- `δ{score}δ` → Relevance

Parse with regex; score range is 0–100 integer per dimension. Never change the delimiter characters without updating the parser.

## Default Paths
Use `pathlib.Path(__file__).parent` as the anchor for default paths so scripts work regardless of the working directory:

```python
_DIR         = Path(__file__).parent
DEFAULT_PRED = str(_DIR / ".." / ".." / "output" / "predictions.json")
DEFAULT_GT_1 = str(_DIR / ".." / ".." / "data" / "ground_truth" / "anet_entities_test_1.json")
```

## Baselines
Baseline predictions live in `data/baselines/`. Reference them when comparing agent output (greedy decode baseline: `greedy_pred_test.json`, `greedy_pred_val.json`).
