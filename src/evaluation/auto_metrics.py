#!/usr/bin/env python3
"""
avaliacao_automatica.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Avaliação automática das legendas geradas pelo agente.
Adaptado do ANETcaptions evaluator (Ranjay Krishna, 2017).

Métricas:
• CIDEr-D  – alinhamento semântico via TF-IDF
• BLEU-4   – precisão de n-gramas (1→4)
• ROUGE-L  – cobertura via subsequência mais longa
• METEOR   – alinhamento com suporte a sinônimos
• R@4      – repetição de 4-gramas entre segmentos (↓ melhor)

Formato de entrada:
predictions  →  output/predictions_gpt.json   (formato do projeto)
ground truth →  anet_entities_test_1.json + anet_entities_test_2.json

Uso:
python avaliacao_automatica.py
python avaliacao_automatica.py -p output/predictions.json -v
python avaliacao_automatica.py --help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import re
import tempfile
import numpy as np
from collections import Counter
from pathlib import Path

# ─── Caminhos padrão ─────────────────────────────────────────
_DIR         = Path(__file__).parent
PROJECT_ROOT = _DIR.parent.parent

# Ground truths
DEFAULT_GT = [
    PROJECT_ROOT / "data" / "ground_truth" / "anet_entities_test_1.json",
    PROJECT_ROOT / "data" / "ground_truth" / "anet_entities_test_2.json",
]

# Predições
DEFAULT_PRED_GPT = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "predictions_gpt.json"
)

DEFAULT_PRED_LLAMA = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "predictions_llama.json"
)

DEFAULT_SKIMCAP = (
    PROJECT_ROOT
    / "data"
    / "baselines"
    / "greedy_pred_test.json"
)

# Saída
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "metrics"
    / "auto"
)
# ─── Tenta importar pycocoevalcap (pip ou pasta local) ────────
_COCO_OK = False

# 1) Adiciona pasta local ao path, se existir
for _cand in [
  _DIR / "coco-caption",
  _DIR / ".." / "coco-caption",
  _DIR / "outros" / "coco-caption",
]:
  if (_cand / "pycocoevalcap").exists():
      sys.path.insert(0, str(_cand.resolve()))
      break

# 2) Tenta import (funciona tanto via pip quanto via pasta local)
try:
  from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer   # noqa: F401
  from pycocoevalcap.bleu.bleu   import Bleu
  from pycocoevalcap.rouge.rouge import Rouge
  from pycocoevalcap.cider.cider import Cider
  _COCO_OK = True
except Exception as e:
  print(f"⚠️  pycocoevalcap indisponível: {e}")

# ─── nltk (METEOR sem Java) ───────────────────────────────────
_NLTK_OK = False
try:
  import nltk
  from nltk.translate.bleu_score   import corpus_bleu, SmoothingFunction
  from nltk.translate.meteor_score import meteor_score as _meteor_fn
  try:
      nltk.download("wordnet",  quiet=True)
      nltk.download("omw-1.4", quiet=True)
  except Exception as e:
      print(f"⚠️  nltk download falhou (possível erro de rede): {e}. METEOR pode estar indisponível.")
  _NLTK_OK = True
except ImportError:
  pass

# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS  (iguais ao original ANETcaptions)
# ══════════════════════════════════════════════════════════════

def parse_sent(text: str) -> list:
    """Tokenização idêntica ao SkimCap/ANETcaptions: remove tudo que não é letra."""
    res = re.sub('[^a-zA-Z]', ' ', text)
    return res.strip().lower().split()

def to_paragraph(sentences) -> str:
  """Junta lista de sentenças em parágrafo único."""
  if isinstance(sentences, str):
      return sentences
  return " ".join(s.strip() for s in sentences if s and s.strip())

# ══════════════════════════════════════════════════════════════
# R@4 – REPETIÇÃO DE 4-GRAMAS ENTRE SEGMENTOS
# ══════════════════════════════════════════════════════════════

def _ngrams(tokens: list, n: int) -> list:
  return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]

def compute_r4(captions: list) -> float:
    """
    R@4 (Repetition-4) — réplica EXATA do SkimCap / densevid_eval.
        R@4 = sum(max(count(g)-1, 0)) / sum(count(g))   sobre 4-gramas
    Acumula os 4-gramas de TODOS os segmentos do vídeo nos mesmos
    contadores (repetição global no parágrafo). ↓ melhor.
    Tokenização idêntica ao SkimCap: corta no primeiro ' .',
    troca ',' por espaço, colapsa espaços, split por espaço.
    """
    fourgrams: dict = {}

    for sent in captions:
        if not sent:
            continue
        # Tokenização do SkimCap
        s = sent.split(' .')[0]
        s = s.replace(',', ' ')
        while '  ' in s:
            s = s.replace('  ', ' ')
        words = s.split(' ')

        # Acumula 4-gramas COM multiplicidade
        for i in range(len(words) - 3):
            g = '%s_%s_%s_%s' % (words[i], words[i+1], words[i+2], words[i+3])
            fourgrams[g] = fourgrams.get(g, 0) + 1

    total = float(sum(fourgrams.values()))
    if total == 0:
        return 0.0
    repeated = float(sum(max(c - 1, 0) for c in fourgrams.values()))
    return repeated / total

# ══════════════════════════════════════════════════════════════
# IMPLEMENTAÇÕES FALLBACK (sem pycocoevalcap)
# ══════════════════════════════════════════════════════════════

def _lcs(x: list, y: list) -> int:
  """Comprimento da subsequência comum mais longa."""
  m, n = len(x), len(y)
  dp = [[0] * (n + 1) for _ in range(m + 1)]
  for i in range(1, m + 1):
      for j in range(1, n + 1):
          dp[i][j] = (
              dp[i - 1][j - 1] + 1
              if x[i - 1] == y[j - 1]
              else max(dp[i - 1][j], dp[i][j - 1])
          )
  return dp[m][n]

def _rouge_l_single(ref: str, hyp: str) -> float:
  r, h = parse_sent(ref), parse_sent(hyp)
  if not r or not h:
      return 0.0
  lcs = _lcs(r, h)
  p   = lcs / len(h)
  rc  = lcs / len(r)
  return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0

def _fallback_rouge_l(gts: dict, res: dict) -> tuple[float, dict]:
  """Retorna (média, scores_por_item) para ROUGE-L."""
  per_item = {}
  for k in res:
      hyp = res[k][0] if res[k] else ""
      refs = gts.get(k, [""])
      per_item[k] = max(_rouge_l_single(ref, hyp) for ref in refs)

  media = float(np.mean(list(per_item.values()))) if per_item else 0.0
  return media, per_item


def _fallback_bleu(gts: dict, res: dict) -> tuple[float, dict]:
  """
  Retorna (BLEU-4 de corpus, BLEU-4 por item) via nltk.

  Observação: o score agregado continua sendo corpus_bleu, como antes.
  O score por item usa sentence_bleu apenas para permitir a análise
  estatística em nível de vídeo quando pycocoevalcap não está disponível.
  """
  if not _NLTK_OK:
      return -1.0, {}

  from nltk.translate.bleu_score import sentence_bleu

  refs_list, hyps_list = [], []
  per_item = {}
  smooth = SmoothingFunction().method1

  for k in res:
      refs = [parse_sent(r) for r in gts.get(k, [""])]
      hyp = parse_sent(res[k][0] if res[k] else "")
      refs_list.append(refs)
      hyps_list.append(hyp)
      per_item[k] = sentence_bleu(
          refs,
          hyp,
          weights=(0.25, 0.25, 0.25, 0.25),
          smoothing_function=smooth,
      )

  corpus = corpus_bleu(
      refs_list,
      hyps_list,
      weights=(0.25, 0.25, 0.25, 0.25),
      smoothing_function=smooth,
  )
  return float(corpus), per_item


def _fallback_meteor(gts: dict, res: dict) -> tuple[float, dict]:
  """Retorna (média, scores_por_item) para METEOR."""
  if not _NLTK_OK:
      return -1.0, {}

  per_item = {}
  for k in res:
      per_item[k] = _meteor_fn(
          [parse_sent(r) for r in gts.get(k, [""])],
          parse_sent(res[k][0] if res[k] else "")
      )

  media = float(np.mean(list(per_item.values()))) if per_item else 0.0
  return media, per_item

# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════

def load_predictions(filepath: str) -> dict:
  """
  Carrega predictions.json (formato do projeto).

  Retorna
  -------
  raw  : {video_id: [caption_seg0, caption_seg1, ...]}
  para : {video_id: "parágrafo concatenado"}
  """
  with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)

  raw, para = {}, {}
  for video in data.get("videos", []):
      vid  = video["video_id"]
      caps = [
          seg["caption"]
          for seg in video.get("segments", [])
          if seg.get("caption")
      ]
      raw[vid]  = caps
      p = to_paragraph(caps)
      para[vid] = p.replace("..", ".").replace(".", " .")
  return raw, para

def load_ground_truth(filepath: str) -> dict:
  """
  Carrega anet_entities_test_*.json (formato do projeto).

  Retorna
  -------
  {video_id: "parágrafo concatenado"}  ← igual ao formato original
  """
  with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)

  result = {}
  for vid_id, info in data.items():
      key  = vid_id if vid_id.startswith("v_") else f"v_{vid_id}"
      para = to_paragraph(info.get("sentences", []))
      para = para.replace("..", ".").replace(".", " .")
      result[key] = para
  return result

# ══════════════════════════════════════════════════════════════
# AVALIADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════

class AvaliacaoAutomatica:
  """
  Avaliador adaptado do ANETcaptions para o formato do projeto.

  Mantém a mesma lógica do original:
    1. Converte GT e predições para formato parágrafo
    2. Avalia com BLEU-4, METEOR, ROUGE-L, CIDEr-D
    3. Acrescenta R@4 (repetição entre segmentos)
  """

  def __init__(self, gt_files: list, pred_file: str, verbose: bool = False,
               restrict_ids: set | None = None):
      self.verbose = verbose

      # Predições: raw (por segmento) e parágrafo
      self.predictions_raw, self.prediction = load_predictions(pred_file)

      # Ground truths: um dict por arquivo (multi-referência)
      self.ground_truths = [load_ground_truth(f) for f in gt_files]

      # IDs a avaliar: preditos ∩ GT (∩ restrict_ids se fornecido)
      self._restrict_ids = restrict_ids

      # Scores individuais (0–100) por vídeo, preenchidos em evaluate().
      self.per_video_scores: dict[str, dict[str, float]] = {}

      if self.verbose:
          n_gt = len(set().union(*[set(g.keys()) for g in self.ground_truths]))
          print(f"\n  📂 Arquivos GT      : {len(gt_files)}")
          print(f"  🎬 Vídeos no GT     : {n_gt}")
          print(f"  📝 Vídeos preditos  : {len(self.prediction)}")

  # ─── Utilitários internos ─────────────────────────────────

  def _get_gt_vid_ids(self) -> list:
      ids: set = set()
      for gt in self.ground_truths:
          ids |= set(gt.keys())
      # Restringe aos vídeos que têm predição gerada
      ids &= set(self.prediction.keys())
      # Restringe ao conjunto de referência (para comparação justa entre modelos)
      if self._restrict_ids:
          ids &= self._restrict_ids
      return sorted(ids)

  def _build_gts_res(self):
      """
      Constrói gts e res no formato {idx: [strings]}
      igual ao ANETcaptions original.
      """
      gt_vid_ids = self._get_gt_vid_ids()
      vid2idx    = {k: i for i, k in enumerate(gt_vid_ids)}

      # gts: multi-referência – um parágrafo por arquivo GT
      gts = {vid2idx[k]: [] for k in gt_vid_ids}
      for gt in self.ground_truths:
          for k in gt_vid_ids:
              if k in gt:
                  gts[vid2idx[k]].append(
                      " ".join(parse_sent(gt[k]))
                  )

      # res: predição como parágrafo único
      res = {
          vid2idx[k]: (
              [" ".join(parse_sent(self.prediction[k]))]
              if k in self.prediction and self.prediction[k]
              else [""]
          )
          for k in gt_vid_ids
      }
      return gt_vid_ids, gts, res

  # ─── Avaliação ────────────────────────────────────────────

  def evaluate(self) -> dict:
      self.scores = self._evaluate_para()
      return self.scores

  def _evaluate_para(self) -> dict:
      """
      Réplica do evaluate_para() do ANETcaptions original,
      com adição de R@4 e preservação dos scores por vídeo.

      Importante:
      - self.scores contém os scores AGREGADOS do corpus (0–100),
        mantendo o mesmo formato usado anteriormente.
      - self.per_video_scores contém os scores individuais de cada vídeo
        (0–100), usados na análise estatística em nível de parágrafo.
      """
      gt_vid_ids, gts, res = self._build_gts_res()
      output: dict[str, float] = {}
      per_video: dict[str, dict[str, float]] = {vid: {} for vid in gt_vid_ids}

      # ── pycocoevalcap: BLEU-4, ROUGE-L, CIDEr ──────────────
      if _COCO_OK:
          scorers = [
              (Bleu(4), "Bleu_4"),
              (Rouge(), "ROUGE_L"),
              (Cider(), "CIDEr"),
          ]

          for scorer, method in scorers:
              if self.verbose:
                  print(f"  🔍 Computing {method}...")

              score, item_scores = scorer.compute_score(gts, res)

              if method == "Bleu_4":
                  # Bleu retorna:
                  #   score       = [BLEU-1, BLEU-2, BLEU-3, BLEU-4] do corpus
                  #   item_scores = [lista B1, lista B2, lista B3, lista B4]
                  output[method] = float(score[3])
                  values = item_scores[3]
              else:
                  output[method] = float(score)
                  values = item_scores

              for vid, value in zip(gt_vid_ids, values):
                  per_video[vid][method] = float(value)

      # ── Fallback sem pycocoevalcap ─────────────────────────
      else:
          if self.verbose:
              print("  ⚠️  Modo fallback (sem pycocoevalcap)")

          if _NLTK_OK:
              if self.verbose:
                  print("  🔍 Computing BLEU-4 (nltk)...")
              bleu, bleu_items = _fallback_bleu(gts, res)
              if bleu >= 0:
                  output["Bleu_4"] = bleu
                  for idx, value in bleu_items.items():
                      per_video[gt_vid_ids[idx]]["Bleu_4"] = float(value)
          else:
              print("  💡 Instale nltk: pip install nltk")

          if self.verbose:
              print("  🔍 Computing ROUGE-L (built-in)...")
          rouge, rouge_items = _fallback_rouge_l(gts, res)
          output["ROUGE_L"] = rouge
          for idx, value in rouge_items.items():
              per_video[gt_vid_ids[idx]]["ROUGE_L"] = float(value)

      # ── METEOR via nltk (sem Java) ─────────────────────────
      if _NLTK_OK:
          if self.verbose:
              print("  🔍 Computing METEOR (nltk)...")
          meteor, meteor_items = _fallback_meteor(gts, res)
          if meteor >= 0:
              output["METEOR"] = meteor
              for idx, value in meteor_items.items():
                  per_video[gt_vid_ids[idx]]["METEOR"] = float(value)

      # ── R@4 (sempre disponível) ────────────────────────────
      if self.verbose:
          print("  🔍 Computing R@4...")

      r4_scores = [compute_r4(self.predictions_raw.get(v, [])) for v in gt_vid_ids]
      output["R@4"] = float(np.mean(r4_scores)) if r4_scores else 0.0

      for vid, value in zip(gt_vid_ids, r4_scores):
          per_video[vid]["R@4"] = float(value)

      # ── Converte corpus e scores individuais para 0–100 ────
      output = {
          key: round(float(value) * 100, 4)
          for key, value in output.items()
      }

      self.per_video_scores = {
          vid: {
              key: round(float(value) * 100, 4)
              for key, value in metrics.items()
          }
          for vid, metrics in per_video.items()
      }

      n_scored = len(gt_vid_ids)
      print(f"\n  📊 Vídeos avaliados : {n_scored} / {len(gt_vid_ids)}")
      return output

  # ─── Saída ────────────────────────────────────────────────

  def print_results(self):
      SEP = "─" * 48
      print(f"\n{SEP}")
      print(f"  {'MÉTRICA':<22} {'SCORE (0–100)':>14}")
      print(SEP)

      # Ordem de exibição preferencial
      order = [
          ("CIDEr",   "CIDEr-D"),
          ("Bleu_4",  "BLEU-4"),
          ("ROUGE_L", "ROUGE-L"),
          ("METEOR",  "METEOR"),
          ("R@4",     "R@4  (↓ melhor)"),
      ]
      shown = set()
      for key, label in order:
          if key in self.scores:
              print(f"  {label:<22} {self.scores[key]:>14.2f}")
              shown.add(key)

      for key, v in self.scores.items():
          if key not in shown:
              print(f"  {key:<22} {v:>14.2f}")

      print(SEP + "\n")

  def print_per_video(self):
      """Exibe todas as métricas individuais no modo verbose."""
      print("  Métricas por vídeo (0–100):")
      order = ["Bleu_4", "ROUGE_L", "CIDEr", "METEOR", "R@4"]

      for vid in sorted(self.per_video_scores):
          metrics = self.per_video_scores[vid]
          parts = []
          for metric in order:
              if metric in metrics:
                  parts.append(f"{metric}={metrics[metric]:.2f}")
          print(f"    {vid}: " + "  ".join(parts))
      print()

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def _converter_skimcap_para_tmp(skimcap_path: str) -> str:
    """
    Converte greedy_pred_test.json (formato SkimCap/ANETcaptions) para
    o formato predictions.json do projeto e salva em arquivo temporário.

    Formato SkimCap de entrada:
      {"results": {video_id: [{"sentence": ..., "timestamp": [s, e]}, ...]}}

    Retorna o caminho do arquivo temporário (deve ser deletado após uso).
    """
    with open(skimcap_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    videos = []
    for vid, segs in results.items():
        segments = [
            {
                "segment_id": i,
                "timestamps": seg.get("timestamp", [0, 0]),
                "caption": seg.get("sentence", ""),
            }
            for i, seg in enumerate(segs)
        ]
        videos.append({"video_id": vid, "segments": segments})

    predictions = {"videos": videos}

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    return tmp_path


def _imprimir_tabela_comparacao(resultados: dict) -> None:
    """
    Imprime tabela resumo comparando métricas automáticas de vários modelos.

    resultados: {label: scores_dict}
    """
    METRICAS = [
        ("CIDEr",   "CIDEr-D"),
        ("Bleu_4",  "BLEU-4"),
        ("ROUGE_L", "ROUGE-L"),
        ("METEOR",  "METEOR"),
        ("R@4",     "R@4 (↓ melhor)"),
    ]
    modelos = list(resultados.keys())
    col = 12
    L = 24 + col * len(modelos)

    print("\n" + "═" * L)
    print("  COMPARAÇÃO — MÉTRICAS AUTOMÁTICAS")
    print("═" * L)

    header = f"  {'MÉTRICA':<22}"
    for m in modelos:
        header += f" {m[:col]:>{col}}"
    print(header)
    print("─" * L)

    mostradas: set = set()
    for key, label in METRICAS:
        if any(key in resultados[m] for m in modelos):
            linha = f"  {label:<22}"
            for m in modelos:
                v = resultados[m].get(key)
                linha += f" {v:>{col}.2f}" if v is not None else f" {'—':>{col}}"
            print(linha)
            mostradas.add(key)

    extras = {k for s in resultados.values() for k in s} - mostradas
    for key in sorted(extras):
        linha = f"  {key:<22}"
        for m in modelos:
            v = resultados[m].get(key)
            linha += f" {v:>{col}.2f}" if v is not None else f" {'—':>{col}}"
        print(linha)

    print("═" * L + "\n")


def _avaliar_e_salvar(
    gt_files: list,
    pred_file: str,
    output_path: str,
    verbose: bool,
    label: str = "",
    restrict_ids: set | None = None,
) -> tuple[dict, dict]:
    """
    Avalia um arquivo de predições.

    Mantém o JSON agregado existente e cria adicionalmente um JSON
    com métricas por vídeo para o mesmo modelo.
    """
    tag = f" [{label}]" if label else ""
    print(f"\n{'─'*55}")
    print(f"  Predictions{tag}: {pred_file}")
    print(f"{'─'*55}")

    evaluator = AvaliacaoAutomatica(
        gt_files=gt_files,
        pred_file=pred_file,
        verbose=verbose,
        restrict_ids=restrict_ids,
    )
    evaluator.evaluate()
    evaluator.print_results()

    if verbose:
        evaluator.print_per_video()

    # 1) Arquivo agregado — MESMO formato anterior.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(evaluator.scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(out)
    print(f"✅ Métricas agregadas salvas em: {out}")

    # 2) Arquivo individual desse modelo.
    # Ex.: metricas_GPT-4.1_multiref.json
    #   -> metricas_GPT-4.1_por_video_multiref.json
    if out.stem.endswith("_multiref"):
        base_stem = out.stem[:-len("_multiref")]
        per_video_out = out.with_name(f"{base_stem}_por_video_multiref.json")
    else:
        per_video_out = out.with_name(f"{out.stem}_por_video.json")
    per_video_payload = {
        "modelo": label or Path(pred_file).stem,
        "escala": "0-100",
        "metricas_por_video": evaluator.per_video_scores,
    }
    tmp_per = per_video_out.with_suffix(".json.tmp")
    tmp_per.write_text(
        json.dumps(per_video_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_per.replace(per_video_out)
    print(f"✅ Métricas por vídeo salvas em: {per_video_out}")

    return evaluator.scores, evaluator.per_video_scores


def _salvar_metricas_por_video_consolidadas(
    out_dir: Path,
    per_video_por_modelo: dict,
    sufixo: str,
) -> None:
    """
    Gera duas saídas consolidadas para análise estatística:

    1. metricas_por_video_multiref.json
    2. metricas_por_video_multiref.csv

    CSV: uma linha por vídeo × modelo (esperado: 10 × 3 = 30 linhas).
    """
    metric_order = ["Bleu_4", "ROUGE_L", "CIDEr", "METEOR", "R@4"]

    # Reorganiza de {modelo: {video: métricas}} para
    # {video: {modelo: métricas}}.
    consolidated: dict = {}
    for model, videos in per_video_por_modelo.items():
        for video_id, metrics in videos.items():
            consolidated.setdefault(video_id, {})[model] = metrics

    json_path = out_dir / f"metricas_por_video_{sufixo}.json"
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_json.replace(json_path)

    csv_path = out_dir / f"metricas_por_video_{sufixo}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "model", *metric_order],
        )
        writer.writeheader()

        for video_id in sorted(consolidated):
            for model in per_video_por_modelo:
                metrics = consolidated[video_id].get(model)
                if metrics is None:
                    continue

                row = {
                    "video_id": video_id,
                    "model": model,
                }
                for metric in metric_order:
                    row[metric] = metrics.get(metric, "")
                writer.writerow(row)

    total_rows = sum(len(models) for models in consolidated.values())
    print(f"\n💾 Consolidado por vídeo: {json_path}")
    print(f"💾 CSV estatístico       : {csv_path} ({total_rows} linhas)")


def main(args):
    mode = "pycocoevalcap ✓" if _COCO_OK else "fallback (nltk + built-in)"

    label1       = getattr(args, "modelo_nome",  "") or Path(args.predictions).stem
    predictions2 = getattr(args, "predictions2", None)
    label2       = (getattr(args, "modelo2_nome", "") or Path(predictions2).stem) if predictions2 else None
    skimcap_path = getattr(args, "skimcap", None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("  AVALIAÇÃO AUTOMÁTICA DE LEGENDAS DE VÍDEO")
    print("=" * 55)
    print(f"  Ground truth: {len(args.ground_truth)} arquivo(s)  [multi-referência]")
    print(f"  Modo        : {mode}")
    print("=" * 55)

    # Converte SkimCap uma vez (reutilizado na avaliação)
    skimcap_tmp: str | None = None
    if skimcap_path and Path(skimcap_path).exists():
        skimcap_tmp = _converter_skimcap_para_tmp(skimcap_path)

    try:
        sufixo = "multiref"

        print(f"\n{'═'*55}")
        print(f"  GTs (multi-ref): {', '.join(Path(g).name for g in args.ground_truth)}")
        print(f"{'═'*55}")

        todos_scores: dict = {}
        per_video_por_modelo: dict = {}

        # Modelo 1
        out1 = str(out_dir / f"metricas_{label1}_{sufixo}.json")
        scores1, per_video1 = _avaliar_e_salvar(
            gt_files=args.ground_truth,
            pred_file=args.predictions,
            output_path=out1,
            verbose=args.verbose,
            label=label1,
        )
        todos_scores[label1] = scores1
        per_video_por_modelo[label1] = per_video1

        # IDs avaliados no modelo 1 — base para comparação justa entre modelos.
        _ref_ids: set | None = None
        try:
            _raw, _para = load_predictions(args.predictions)
            _gt_union: set = set()
            for _g in args.ground_truth:
                _gt_union |= set(load_ground_truth(_g).keys())
            _ref_ids = set(_para.keys()) & _gt_union
        except Exception:
            pass

        # Modelo 2 (opcional)
        if predictions2 and Path(predictions2).exists():
            out2 = str(out_dir / f"metricas_{label2}_{sufixo}.json")
            scores2, per_video2 = _avaliar_e_salvar(
                gt_files=args.ground_truth,
                pred_file=predictions2,
                output_path=out2,
                verbose=args.verbose,
                label=label2,
                restrict_ids=_ref_ids,
            )
            todos_scores[label2] = scores2
            per_video_por_modelo[label2] = per_video2

        # SkimCap (opcional)
        if skimcap_tmp:
            out_sc = str(out_dir / f"metricas_SkimCap_{sufixo}.json")
            scores_sc, per_video_sc = _avaliar_e_salvar(
                gt_files=args.ground_truth,
                pred_file=skimcap_tmp,
                output_path=out_sc,
                verbose=args.verbose,
                label="SkimCap",
                restrict_ids=_ref_ids,
            )
            todos_scores["SkimCap"] = scores_sc
            per_video_por_modelo["SkimCap"] = per_video_sc

        # Tabela comparativa agregada final — comportamento anterior.
        if len(todos_scores) > 1:
            _imprimir_tabela_comparacao(todos_scores)
        else:
            print()

        # NOVO: resultado por vídeo/modelo para análise estatística.
        _salvar_metricas_por_video_consolidadas(
            out_dir=out_dir,
            per_video_por_modelo=per_video_por_modelo,
            sufixo=sufixo,
        )

    finally:
        if skimcap_tmp:
            Path(skimcap_tmp).unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avaliação automática de legendas – formato do projeto."
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        nargs="+",
        default=[str(p) for p in DEFAULT_GT],
        help="Arquivos ground truth"
    )

    parser.add_argument(
        "-p",
        "--predictions",
        default=str(DEFAULT_PRED_GPT),
        help="Predições do GPT-4.1"
    )

    parser.add_argument(
        "-p2",
        "--predictions2",
        default=str(DEFAULT_PRED_LLAMA),
        help="Predições do Llama 4"
    )

    parser.add_argument(
        "-s",
        "--skimcap",
        default=str(DEFAULT_SKIMCAP),
        help="Predições do SkimCap"
    )

    parser.add_argument(
        "--modelo-nome",
        default="GPT-4.1"
    )

    parser.add_argument(
        "--modelo2-nome",
        default="Llama 4"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório de saída"
    )
    
    parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Exibe detalhes das métricas por vídeo"
    )
    
    args = parser.parse_args()
    main(args)
