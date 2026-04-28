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
import tempfile
import numpy as np
from collections import Counter
from pathlib import Path

# ─── Caminhos padrão ─────────────────────────────────────────
_DIR         = Path(__file__).parent
DEFAULT_PRED = str(_DIR / ".." / ".." / "output" / "predictions" / "predictions_gpt.json")
DEFAULT_GT_1 = str(_DIR / ".." / ".." / "data" / "ground_truth" / "anet_entities_test_1.json")
DEFAULT_GT_2 = str(_DIR / ".." / ".." / "data" / "ground_truth" / "anet_entities_test_2.json")
DEFAULT_OUT  = str(_DIR / ".." / ".." / "output" / "metrics" / "auto" / "metricas.json")

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
  """Retorna só Bleu_4 via nltk."""
  if not _NLTK_OK:
      return {}
  refs_list, hyps_list = [], []
  for k in res:
      refs_list.append([parse_sent(r) for r in gts.get(k, [""])])
      hyps_list.append(parse_sent(res[k][0] if res[k] else ""))
  smooth = SmoothingFunction().method1
  return {
      "Bleu_4": corpus_bleu(refs_list, hyps_list,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smooth),
  }

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

  def __init__(self, gt_files: list, pred_file: str, verbose: bool = False,
               restrict_ids: set | None = None):
      self.verbose = verbose

      # Predições: raw (por segmento) e parágrafo
      self.predictions_raw, self.prediction = load_predictions(pred_file)

      # Ground truths: um dict por arquivo (multi-referência)
      self.ground_truths = [load_ground_truth(f) for f in gt_files]

      # IDs a avaliar: preditos ∩ GT (∩ restrict_ids se fornecido)
      self._restrict_ids = restrict_ids

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
      com adição de R@4.
      """
      gt_vid_ids, gts, res = self._build_gts_res()
      output = {}

      # ── pycocoevalcap: BLEU-4, ROUGE-L, CIDEr (sem METEOR — requer Java) ─
      if _COCO_OK:
          scorers = [
              (Bleu(4),  "Bleu_4"),
              (Rouge(),  "ROUGE_L"),
              (Cider(),  "CIDEr"),
          ]
          for scorer, method in scorers:
              if self.verbose:
                  print(f"  🔍 Computing {method}...")
              score, _ = scorer.compute_score(gts, res)
              # Bleu scorer retorna lista [b1,b2,b3,b4] — pegar só BLEU-4
              output[method] = score[3] if method == "Bleu_4" else score
      else:
          if self.verbose:
              print("  ⚠️  Modo fallback (sem pycocoevalcap)")
          if _NLTK_OK:
              if self.verbose: print("  🔍 Computing BLEU-4 (nltk)...")
              output.update(_fallback_bleu(gts, res))
          else:
              print("  💡 Instale nltk: pip install nltk")
          if self.verbose: print("  🔍 Computing ROUGE-L (built-in)...")
          output["ROUGE_L"] = _fallback_rouge_l(gts, res)

      # ── METEOR via nltk (sem Java) ────────────────────────
      if _NLTK_OK:
          if self.verbose: print("  🔍 Computing METEOR (nltk)...")
          v = _fallback_meteor(gts, res)
          if v >= 0:
              output["METEOR"] = v

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
                linha += f" {v * 100:>{col}.2f}%" if v is not None else f" {'—':>{col}}"
            print(linha)
            mostradas.add(key)

    extras = {k for s in resultados.values() for k in s} - mostradas
    for key in sorted(extras):
        linha = f"  {key:<22}"
        for m in modelos:
            v = resultados[m].get(key)
            linha += f" {v * 100:>{col}.2f}%" if v is not None else f" {'—':>{col}}"
        print(linha)

    print("═" * L + "\n")


def _avaliar_e_salvar(
    gt_files: list,
    pred_file: str,
    output_path: str,
    verbose: bool,
    label: str = "",
    restrict_ids: set | None = None,
) -> dict:
    """Avalia um arquivo de predições e salva o JSON de métricas."""
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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(evaluator.scores, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(f"✅ Métricas salvas em: {out}")
    return evaluator.scores


def main(args):
    mode = "pycocoevalcap ✓" if _COCO_OK else "fallback (nltk + built-in)"

    label1       = getattr(args, "modelo_nome",  "") or Path(args.predictions).stem
    predictions2 = getattr(args, "predictions2", None)
    label2       = (getattr(args, "modelo2_nome", "") or Path(predictions2).stem) if predictions2 else None
    skimcap_path = getattr(args, "skimcap", None)
    out_dir      = Path(args.output)

    print("\n" + "=" * 55)
    print("  AVALIAÇÃO AUTOMÁTICA DE LEGENDAS DE VÍDEO")
    print("=" * 55)
    print(f"  Ground truth: {len(args.references)} arquivo(s)")
    print(f"  Modo        : {mode}")
    print("=" * 55)

    # Converte SkimCap uma vez (reutilizado em todos os GTs)
    skimcap_tmp: str | None = None
    if skimcap_path and Path(skimcap_path).exists():
        skimcap_tmp = _converter_skimcap_para_tmp(skimcap_path)

    try:
        # ── Loop por GT (igual ao llm_eval.py) ───────────────────
        for gt_file in args.references:
            sufixo = Path(gt_file).stem  # ex: anet_entities_test_1

            print(f"\n{'═'*55}")
            print(f"  GT: {Path(gt_file).name}")
            print(f"{'═'*55}")

            todos_scores: dict = {}

            # Modelo 1
            out1 = str(out_dir / f"metricas_{label1}_{sufixo}.json")
            scores1 = _avaliar_e_salvar(
                gt_files=[gt_file],
                pred_file=args.predictions,
                output_path=out1,
                verbose=args.verbose,
                label=label1,
            )
            todos_scores[label1] = scores1

            # IDs avaliados no modelo 1 — base para comparação justa
            _ref_ids: set | None = None
            try:
                _raw, _para = load_predictions(args.predictions)
                _gt = load_ground_truth(gt_file)
                _ref_ids = set(_para.keys()) & set(_gt.keys())
            except Exception:
                pass

            # Modelo 2 (opcional)
            if predictions2 and Path(predictions2).exists():
                out2 = str(out_dir / f"metricas_{label2}_{sufixo}.json")
                todos_scores[label2] = _avaliar_e_salvar(
                    gt_files=[gt_file],
                    pred_file=predictions2,
                    output_path=out2,
                    verbose=args.verbose,
                    label=label2,
                    restrict_ids=_ref_ids,
                )

            # SkimCap (opcional)
            if skimcap_tmp:
                out_sc = str(out_dir / f"metricas_skimcap_{sufixo}.json")
                todos_scores["SkimCap"] = _avaliar_e_salvar(
                    gt_files=[gt_file],
                    pred_file=skimcap_tmp,
                    output_path=out_sc,
                    verbose=args.verbose,
                    label="SkimCap",
                    restrict_ids=_ref_ids,
                )

            # Tabela comparativa para este GT
            if len(todos_scores) > 1:
                _imprimir_tabela_comparacao(todos_scores)
            else:
                print()

    finally:
        if skimcap_tmp:
            Path(skimcap_tmp).unlink(missing_ok=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Avaliação automática de legendas – formato do projeto."
  )
  parser.add_argument(
      "-p", "--predictions",
      type=str, default=DEFAULT_PRED,
      help=f"predictions.json do modelo principal  (padrão: {DEFAULT_PRED})",
  )
  parser.add_argument(
      "-p2", "--predictions2",
      type=str, default=None,
      help="predictions.json do segundo modelo (opcional)",
  )
  parser.add_argument(
      "-s", "--skimcap",
      type=str, default=None,
      help="greedy_pred_test.json da baseline SkimCap (opcional)",
  )
  parser.add_argument(
      "--modelo-nome", dest="modelo_nome", type=str, default="",
      help="Rótulo do modelo principal na tabela comparativa",
  )
  parser.add_argument(
      "--modelo2-nome", dest="modelo2_nome", type=str, default="",
      help="Rótulo do segundo modelo na tabela comparativa",
  )
  parser.add_argument(
      "-r", "--references",
      type=str, nargs="+",
      default=[DEFAULT_GT_1, DEFAULT_GT_2],
      help="Arquivo(s) de ground truth (podem ser múltiplos)",
  )
  parser.add_argument(
      "-o", "--output",
      type=str, default=str(Path(DEFAULT_OUT).parent),
      help=f"Pasta de saída para os JSONs de métricas (padrão: output/)",
  )
  parser.add_argument(
      "-v", "--verbose",
      action="store_true",
      help="Exibe detalhes por scorer e por vídeo",
  )
  args = parser.parse_args()
  main(args)