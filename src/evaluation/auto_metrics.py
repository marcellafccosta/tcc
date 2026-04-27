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
predictions  →  output/predictions.json   (formato do projeto)
ground truth →  anet_entities_test_1.json + anet_entities_test_2.json

Uso:
python avaliacao_automatica.py
python avaliacao_automatica.py -p output/predictions.json -v
python avaliacao_automatica.py --help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter
from pathlib import Path

# ─── Caminhos padrão ─────────────────────────────────────────
_DIR         = Path(__file__).parent
DEFAULT_PRED = str(_DIR / ".." / ".." / "output" / "predictions.json")
DEFAULT_GT_1 = str(_DIR / ".." / ".." / "data" / "ground_truth" / "anet_entities_test_1.json")
DEFAULT_GT_2 = str(_DIR / ".." / ".." / "data" / "ground_truth" / "anet_entities_test_2.json")
DEFAULT_OUT  = str(_DIR / ".." / ".." / "output" / "metricas.json")

# ─── Tenta importar pycocoevalcap (mesmos scorers do original) ─
_COCO_OK = False
for _cand in [
  _DIR / "coco-caption",
  _DIR / ".." / "coco-caption",
  _DIR / "outros" / "coco-caption",
]:
  if (_cand / "pycocoevalcap").exists():
      sys.path.insert(0, str(_cand.resolve()))
      _COCO_OK = True
      break

if _COCO_OK:
  try:
      from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer   # noqa: F401
      from pycocoevalcap.bleu.bleu   import Bleu
      from pycocoevalcap.meteor.meteor import Meteor
      from pycocoevalcap.rouge.rouge import Rouge
      from pycocoevalcap.cider.cider import Cider
  except Exception as e:
      print(f"⚠️  pycocoevalcap encontrado mas com erro: {e}")
      _COCO_OK = False

# ─── Fallback: nltk ───────────────────────────────────────────
_NLTK_OK = False
if not _COCO_OK:
  try:
      import nltk
      from nltk.translate.bleu_score  import corpus_bleu, SmoothingFunction
      from nltk.translate.meteor_score import meteor_score as _meteor_fn
      nltk.download("wordnet",  quiet=True)
      nltk.download("omw-1.4", quiet=True)
      _NLTK_OK = True
  except ImportError:
      pass

# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS  (iguais ao original ANETcaptions)
# ══════════════════════════════════════════════════════════════

def parse_sent(text: str) -> list:
  """Tokenização simples: lowercase + split (igual ao original)."""
  return text.strip().lower().split()

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
  R@4 = proporção de 4-gramas (distintos) que aparece em
        mais de um segmento do mesmo vídeo.

  R@4 = 0  →  nenhum 4-grama repetido entre segmentos  (ideal)
  R@4 = 1  →  todos os 4-gramas repetidos  (muito repetitivo)

  Mede concisão: legendas de segmentos distintos devem
  descrever conteúdos diferentes.
  """
  captions = [c for c in captions if c and c.strip()]
  if len(captions) <= 1:
      return 0.0

  ngrams_per_seg = [set(_ngrams(parse_sent(c), 4)) for c in captions]

  # Quantos segmentos distintos contêm cada 4-grama
  counts: Counter = Counter()
  for ng_set in ngrams_per_seg:
      for ng in ng_set:
          counts[ng] += 1

  total    = len(counts)
  repeated = sum(1 for v in counts.values() if v > 1)
  return repeated / total if total > 0 else 0.0

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

def _fallback_rouge_l(gts: dict, res: dict) -> float:
  scores = [
      max(_rouge_l_single(ref, res[k][0] if res[k] else "")
          for ref in gts.get(k, [""]))
      for k in res
  ]
  return float(np.mean(scores)) if scores else 0.0

def _fallback_bleu(gts: dict, res: dict) -> dict:
  """Retorna Bleu_1…Bleu_4 via nltk."""
  if not _NLTK_OK:
      return {}
  refs_list, hyps_list = [], []
  for k in res:
      refs_list.append([parse_sent(r) for r in gts.get(k, [""])])
      hyps_list.append(parse_sent(res[k][0] if res[k] else ""))
  smooth = SmoothingFunction().method1
  out = {}
  for n, w in [
      ("Bleu_1", (1, 0, 0, 0)),
      ("Bleu_2", (0.5, 0.5, 0, 0)),
      ("Bleu_3", (1/3, 1/3, 1/3, 0)),
      ("Bleu_4", (0.25, 0.25, 0.25, 0.25)),
  ]:
      out[n] = corpus_bleu(refs_list, hyps_list,
                           weights=w,
                           smoothing_function=smooth)
  return out

def _fallback_meteor(gts: dict, res: dict) -> float:
  if not _NLTK_OK:
      return -1.0
  scores = [
      _meteor_fn(
          [parse_sent(r) for r in gts.get(k, [""])],
          parse_sent(res[k][0] if res[k] else "")
      )
      for k in res
  ]
  return float(np.mean(scores)) if scores else 0.0

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
      para[vid] = to_paragraph(caps)
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

  def __init__(self, gt_files: list, pred_file: str, verbose: bool = False):
      self.verbose = verbose

      # Predições: raw (por segmento) e parágrafo
      self.predictions_raw, self.prediction = load_predictions(pred_file)

      # Ground truths: um dict por arquivo (multi-referência)
      self.ground_truths = [load_ground_truth(f) for f in gt_files]

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
      com adição de R@4.
      """
      gt_vid_ids, gts, res = self._build_gts_res()
      output = {}

      # ── pycocoevalcap (scorers idênticos ao original) ─────
      if _COCO_OK:
          scorers = [
              (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
              (Meteor(), "METEOR"),
              (Rouge(),  "ROUGE_L"),
              (Cider(),  "CIDEr"),
          ]
          for scorer, method in scorers:
              label = method[-1] if isinstance(method, list) else method
              if self.verbose:
                  print(f"  🔍 Computing {label}...")
              score, _ = scorer.compute_score(gts, res)
              if isinstance(method, list):
                  for i, m in enumerate(method):
                      output[m] = score[i]
              else:
                  output[method] = score

      # ── Fallback: nltk + ROUGE-L built-in ─────────────────
      else:
          if self.verbose:
              print("  ⚠️  Modo fallback (sem pycocoevalcap)")

          if _NLTK_OK:
              if self.verbose: print("  🔍 Computing BLEU 1-4 (nltk)...")
              output.update(_fallback_bleu(gts, res))

              if self.verbose: print("  🔍 Computing METEOR (nltk)...")
              v = _fallback_meteor(gts, res)
              if v >= 0:
                  output["METEOR"] = v
          else:
              print("  💡 Instale nltk: pip install nltk")

          if self.verbose: print("  🔍 Computing ROUGE-L (built-in)...")
          output["ROUGE_L"] = _fallback_rouge_l(gts, res)

      # ── R@4 (sempre disponível) ───────────────────────────
      if self.verbose:
          print("  🔍 Computing R@4...")
      r4_scores = [
          compute_r4(self.predictions_raw.get(v, []))
          for v in gt_vid_ids
      ]
      output["R@4"] = float(np.mean(r4_scores))

      n_scored = sum(1 for v in gt_vid_ids if v in self.prediction)
      print(f"\n  📊 Vídeos avaliados : {n_scored} / {len(gt_vid_ids)}")
      return output

  # ─── Saída ────────────────────────────────────────────────

  def print_results(self):
      SEP = "─" * 48
      print(f"\n{SEP}")
      print(f"  {'MÉTRICA':<14} {'SCORE':>8}   {'× 100':>8}")
      print(SEP)

      # Ordem de exibição preferencial
      order = [
          ("CIDEr",   "CIDEr-D"),
          ("Bleu_4",  "BLEU-4"),
          ("ROUGE_L", "ROUGE-L"),
          ("METEOR",  "METEOR"),
          ("R@4",     "R@4  (↓ melhor)"),
          ("Bleu_1",  "BLEU-1"),
          ("Bleu_2",  "BLEU-2"),
          ("Bleu_3",  "BLEU-3"),
      ]
      shown = set()
      for key, label in order:
          if key in self.scores:
              v = self.scores[key]
              print(f"  {label:<22} {v:>8.4f}   {100*v:>7.2f}%")
              shown.add(key)

      for key, v in self.scores.items():
          if key not in shown:
              print(f"  {key:<22} {v:>8.4f}   {100*v:>7.2f}%")

      print(SEP + "\n")

  def print_per_video(self):
      """Detalhe por vídeo no modo verbose."""
      print("  R@4 por vídeo:")
      for vid in sorted(self.predictions_raw):
          score  = compute_r4(self.predictions_raw[vid])
          n_segs = len([c for c in self.predictions_raw[vid] if c])
          in_gt  = any(vid in gt for gt in self.ground_truths)
          marker = "✓" if in_gt else "✗ (fora do GT)"
          print(f"    {vid}  segs={n_segs}  R@4={score:.3f}  {marker}")
      print()

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main(args):
  mode = "pycocoevalcap ✓" if _COCO_OK else "fallback (nltk + built-in)"
  print("\n" + "=" * 55)
  print("  AVALIAÇÃO AUTOMÁTICA DE LEGENDAS DE VÍDEO")
  print("=" * 55)
  print(f"  Predictions : {args.predictions}")
  print(f"  Ground truth: {len(args.references)} arquivo(s)")
  print(f"  Modo        : {mode}")
  print("=" * 55)

  evaluator = AvaliacaoAutomatica(
      gt_files  = args.references,
      pred_file = args.predictions,
      verbose   = args.verbose,
  )
  evaluator.evaluate()
  evaluator.print_results()

  if args.verbose:
      evaluator.print_per_video()

  # Salva métricas em JSON
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "w", encoding="utf-8") as f:
      json.dump(evaluator.scores, f, indent=2, ensure_ascii=False)
  print(f"✅ Métricas salvas em: {out}\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Avaliação automática de legendas – formato do projeto."
  )
  parser.add_argument(
      "-p", "--predictions",
      type=str, default=DEFAULT_PRED,
      help=f"predictions.json do projeto  (padrão: {DEFAULT_PRED})",
  )
  parser.add_argument(
      "-r", "--references",
      type=str, nargs="+",
      default=[DEFAULT_GT_1, DEFAULT_GT_2],
      help="Arquivo(s) de ground truth (podem ser múltiplos)",
  )
  parser.add_argument(
      "-o", "--output",
      type=str, default=DEFAULT_OUT,
      help=f"Saída JSON com as métricas  (padrão: {DEFAULT_OUT})",
  )
  parser.add_argument(
      "-v", "--verbose",
      action="store_true",
      help="Exibe detalhes por scorer e por vídeo",
  )
  args = parser.parse_args()
  main(args)