"""
Script de avaliação de legendas geradas — ACCR com LLM como avaliador
Compara as legendas do agente GPT-4 com Ground Truth e SkimCap

Métricas ACCR (0–100):
α  Accuracy     — descrição factual e precisa
β  Completeness — cobre ações, objetos e contexto relevantes
ψ  Conciseness  — clara e eficiente (2-3 frases máximo)
δ  Relevance    — descreve o conteúdo principal do segmento

Este script é SEPARADO do agente de geração.
Ele lê os JSONs gerados pelo agente e chama um LLM para avaliar.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional

# Garante que config seja encontrado independente de onde o script é executado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
# Garante que auto_metrics seja encontrado (mesmo diretório)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

import config
from token_manager import GerenciadorTokens

# ─────────────────────────────────────────────────────────────────
# Prompt ACCR
# ─────────────────────────────────────────────────────────────────

ACCR_PROMPT_TEMPLATE: str = (
    Path(__file__).parent / "prompts" / "accr.txt"
).read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────
# Carregamento de arquivos
# ─────────────────────────────────────────────────────────────────

def carregar_json(arquivo: str) -> dict:
  with open(arquivo, "r", encoding="utf-8") as f:
      return json.load(f)

# ─────────────────────────────────────────────────────────────────
# Métricas Automáticas (BLEU, METEOR, ROUGE-L, CIDEr)
# ─────────────────────────────────────────────────────────────────

def converter_predictions_para_anet(predictions: dict) -> dict:
  """
  Converte predictions.json → formato ANETcaptions.
  {"results": {vid: [{"sentence": "..."}]}}
  """
  results = {}
  for video in predictions.get("videos", []):
      vid = video["video_id"]
      sentences = [
          {"sentence": seg["caption"]}
          for seg in video.get("segments", [])
          if seg.get("caption")
      ]
      if sentences:
          results[vid] = sentences
  return {"results": results}


def converter_gt_para_anet(gt_data: dict) -> dict:
  """
  Converte anet_entities_test_X.json → formato ANETcaptions.
  {vid: "sentence1 sentence2 ..."}
  """
  return {
      vid: " ".join(info.get("sentences", []))
      for vid, info in gt_data.items()
      if info.get("sentences")
  }


def avaliar_metricas_automatizadas(
  predictions_file: str,
  arquivos_gt: List[str],
  arquivo_saida: str,
) -> dict:
  """
  Calcula BLEU-4, METEOR, ROUGE-L, CIDEr e R@4 usando AvaliacaoAutomatica.
  """
  try:
      from auto_metrics import AvaliacaoAutomatica
  except ImportError as e:
      print(f"  ✗ auto_metrics não encontrado: {e}")
      return {}

  gt_validos = [f for f in arquivos_gt if os.path.exists(f)]
  if not gt_validos:
      print("  ✗ Nenhum arquivo GT válido encontrado")
      return {}

  try:
      evaluator = AvaliacaoAutomatica(
          gt_files=gt_validos,
          pred_file=predictions_file,
          verbose=True,
      )
      scores = evaluator.evaluate()

      Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
      tmp = Path(arquivo_saida).with_suffix(".json.tmp")
      tmp.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
      tmp.replace(arquivo_saida)

      evaluator.print_results()
      print(f"\n✓ Métricas salvas em: {arquivo_saida}")
      return scores

  except Exception as e:
      print(f"  ✗ Erro ao calcular métricas: {e}")
      return {}

      for p in gt_paths:
          try:
              os.unlink(p)
          except OSError:
              pass


# ─────────────────────────────────────────────────────────────────

class AvaliadorACCR:
  """
  Usa um LLM como juiz para calcular métricas ACCR.
  Suporta GitHub Models: github_gpt41 | github_llama | github_phi
  """

  # Regex primários (marcadores gregos)
  PADROES = {
      "accuracy":     re.compile(r"α(\d{1,3})α"),
      "completeness": re.compile(r"β(\d{1,3})β"),
      "conciseness":  re.compile(r"ψ(\d{1,3})ψ"),
      "relevance":    re.compile(r"δ(\d{1,3})δ"),
  }

  # Regex de fallback — captura padrões como "Accuracy: 40" ou "accuracy score is 40"
  PADROES_FALLBACK = {
      "accuracy":     re.compile(r"accuracy(?:\s+score)?(?:\s+is)?[:\s]+([0-9]{1,3})", re.I),
      "completeness": re.compile(r"completeness(?:\s+score)?(?:\s+is)?[:\s]+([0-9]{1,3})", re.I),
      "conciseness":  re.compile(r"conciseness(?:\s+score)?(?:\s+is)?[:\s]+([0-9]{1,3})", re.I),
      "relevance":    re.compile(r"relevance(?:\s+score)?(?:\s+is)?[:\s]+([0-9]{1,3})", re.I),
  }

  # GPT-4.1: 10 req/min → 1 req a cada 6.5s
  _DELAY_PADRAO = 6.5
  _DAILY_LIMIT  = 50

  def __init__(self, provider: str = None, delay: float = None):
      """
      Args:
          provider: github_gpt41 | github_llama | github_phi
                    (padrão: config.EVALUATOR_PROVIDER)
          delay: segundos entre chamadas (padrão: 6.5s para respeitar 10 req/min)
      """
      self.provider = provider or config.EVALUATOR_PROVIDER
      self.delay = delay if delay is not None else self._DELAY_PADRAO

      tokens = [
          config.GITHUB_TOKEN,
          getattr(config, "GITHUB_TOKEN_2", ""),
          getattr(config, "GITHUB_TOKEN_3", ""),
          getattr(config, "GITHUB_TOKEN_4", ""),
          getattr(config, "GITHUB_TOKEN_5", ""),
          getattr(config, "GITHUB_TOKEN_6", ""),
          getattr(config, "GITHUB_TOKEN_7", ""),
      ]
      self._tokens = GerenciadorTokens(
          tokens=tokens,
          endpoint=config.GITHUB_ENDPOINT,
          limite_diario=self._DAILY_LIMIT,
          timeout=config.LLM_TIMEOUT,
      )

  def _model_name(self) -> str:
      return {
          "github_gpt41": config.GITHUB_GPT41,
          "github_llama": config.GITHUB_LLAMA,
          "github_phi":   config.GITHUB_PHI,
      }.get(self.provider, config.GITHUB_GPT41)

  # ── Chamada ao LLM ────────────────────────────────────────────

  def _chamar_llm(self, prompt: str) -> Optional[str]:
      """Envia prompt ao LLM e retorna o texto da resposta."""
      max_tentativas = 3
      for tentativa in range(max_tentativas):
          try:
              cliente = self._tokens.cliente_atual()
              response = cliente.complete(
                  messages=[UserMessage(content=prompt)],
                  model=self._model_name(),
                  max_tokens=2048,
                  temperature=0.0,
              )
              chamadas = self._tokens.registrar_chamada()
              print(f"    [Token {self._tokens.idx + 1}: {chamadas}/{self._tokens.limite_diario} req]")
              return response.choices[0].message.content

          except RuntimeError:
              # Todos os tokens esgotados — lançado pelo GerenciadorTokens
              print("  ⚠️  Todos os tokens esgotados")
              return None
          except Exception as e:
              msg = str(e).lower()
              is_rate_limit = (
                  '429' in msg
                  or ('json is invalid' in msg and 'too many requests' in msg)
              )
              is_timeout = 'read timed out' in msg or 'timed out' in msg or 'timeout' in msg
              is_gateway = ('bad gateway' in msg or 'service unavailable' in msg
                           or 'internal server error' in msg or 'unexpected eof' in msg
                           or '502' in msg or '503' in msg or '500' in msg)
              if is_rate_limit:
                  ainda_ha = self._tokens.marcar_esgotado()
                  if not ainda_ha:
                      return None
                  # Continua o loop com o próximo token
              elif is_timeout or is_gateway:
                  espera = 20 * (tentativa + 1)
                  motivo = "Timeout" if is_timeout else "Bad Gateway"
                  print(f"    ⚠️  {motivo} (tentativa {tentativa + 1}/{max_tentativas}) — aguardando {espera}s")
                  time.sleep(espera)
                  # Continua o loop para nova tentativa
              else:
                  print(f"    ✗ Erro LLM ({self.provider}): {e}")
                  return None
      print(f"    ✗ Esgotadas {max_tentativas} tentativas")
      return None

  # ── Parsing da resposta ───────────────────────────────────────

  def _extrair_scores(self, texto: str) -> Dict[str, Optional[int]]:
      """
      Extrai os 4 scores da resposta do LLM.
      Tenta primeiro os marcadores gregos; se falhar usa padrões de texto.
      Retorna None em cada dimensão que não for encontrada.
      """
      scores = {}
      for dimensao in self.PADROES:
          m = self.PADROES[dimensao].search(texto)
          if m:
              scores[dimensao] = int(m.group(1))
          else:
              # Fallback: "Accuracy: 40" ou "Accuracy score is 40"
              m2 = self.PADROES_FALLBACK[dimensao].search(texto)
              scores[dimensao] = int(m2.group(1)) if m2 else None
      return scores

  # ── Avaliação de um segmento ──────────────────────────────────

  def avaliar_segmento(
      self,
      caption: str,
      referencias: List[str]
  ) -> Dict[str, any]:
      """
      Avalia uma legenda contra as referências.

      Returns:
          {
              "scores": {"accuracy": int, "completeness": int,
                         "conciseness": int, "relevance": int},
              "media": float,
              "resposta_llm": str,
              "valido": bool
          }
      """
      scores_nulos = {
          "accuracy": None,
          "completeness": None,
          "conciseness": None,
          "relevance": None
      }

      if not caption:
          return {"scores": scores_nulos, "media": 0.0,
                  "resposta_llm": "", "valido": False}

      # Formata referências como lista numerada
      refs_texto = "\n".join(
          f"{i+1}. {r}" for i, r in enumerate(referencias) if r
      )

      prompt = ACCR_PROMPT_TEMPLATE.format(
          reference=refs_texto,
          caption=caption
      )

      resposta = self._chamar_llm(prompt)
      time.sleep(self.delay)  # aguarda APÓS a chamada para não atrasar a primeira

      if not resposta:
          return {"scores": scores_nulos, "media": 0.0,
                  "resposta_llm": "", "valido": False}

      scores = self._extrair_scores(resposta)

      # Debug: avisa se o parsing falhou
      valores_encontrados = sum(1 for v in scores.values() if v is not None)
      if valores_encontrados < 4:
          print(f"      ⚠ Parsing incompleto ({valores_encontrados}/4 scores). Resposta LLM:")
          print(f"        {resposta[:300]}")

      # Calcula média apenas com scores válidos
      valores = [v for v in scores.values() if v is not None]
      media = sum(valores) / len(valores) if valores else 0.0

      valido = all(v is not None for v in scores.values())

      return {
          "scores": scores,
          "media": round(media, 2),
          "resposta_llm": resposta,
          "valido": valido
      }

# ─────────────────────────────────────────────────────────────────
# Preparação dos dados
# ─────────────────────────────────────────────────────────────────

def preparar_para_comparacao(
  resultados_agente: dict,
  ground_truth_data: dict,
  skimcap_data: dict = None,
  modelo_nome: str = "modelo",
  resultados_agente2: dict = None,
  modelo2_nome: str = None
) -> dict:
  """
  Organiza dados de todos os modelos por video_id.

  Returns:
      {
          video_id: {
              modelo_nome:  [captions],
              "ground_truth": [refs],
              "SkimCap":    [captions] | None,
              modelo2_nome: [captions] | None,
              "timestamps": [[start, end], ...]
          }
      }
  """
  dados = {}

  agente2_map = {}
  if resultados_agente2 and modelo2_nome:
      for video in resultados_agente2.get("videos", []):
          agente2_map[video["video_id"]] = [
              seg["caption"] for seg in video.get("segments", [])
          ]

  for video in resultados_agente.get("videos", []):
      vid = video["video_id"]

      captions = [seg["caption"] for seg in video.get("segments", [])]
      timestamps = [seg["timestamps"] for seg in video.get("segments", [])]

      gt = ground_truth_data.get(vid, {})
      gt_captions = gt.get("sentences", [])

      skimcap_captions = None
      _sk_results = (skimcap_data or {}).get("results", {})
      if vid in _sk_results:
          skimcap_captions = [item.get("sentence", "") for item in _sk_results[vid]]

      entry = {
          modelo_nome:    captions,
          "ground_truth": gt_captions,
          "SkimCap":      skimcap_captions,
          "timestamps":   timestamps
      }

      if modelo2_nome and vid in agente2_map:
          entry[modelo2_nome] = agente2_map[vid]

      dados[vid] = entry

  return dados

# ─────────────────────────────────────────────────────────────────
# Avaliação por vídeo
# ─────────────────────────────────────────────────────────────────

def avaliar_video(
  video_id: str,
  dados: dict,
  modelos: List[str],
  avaliador: AvaliadorACCR
) -> List[dict]:
  """
  Avalia todos os segmentos de um vídeo para cada modelo.

  Returns:
      Lista de dicts por segmento com scores ACCR de cada modelo.
  """
  gt_captions = dados["ground_truth"]
  timestamps = dados.get("timestamps", [])
  n_segmentos = len(dados.get(modelos[0], []))

  print(f"\n{'='*60}")
  print(f"Vídeo: {video_id}  |  Segmentos: {n_segmentos}")
  print(f"{'='*60}")

  resultados_segmentos = []

  for i in range(n_segmentos):
      referencias = [gt_captions[i]] if i < len(gt_captions) else gt_captions or [""]
      timestamp = timestamps[i] if i < len(timestamps) else [0, 0]

      print(f"\n  Segmento {i} [{timestamp[0]:.1f}s – {timestamp[1]:.1f}s]")
      if referencias and referencias[0]:
          print(f"    GT:  {referencias[0][:90]}...")

      resultado_seg = {
          "segment":    i,
          "timestamps": timestamp,
          "ground_truth": referencias,
          "avaliacoes": {}
      }

      for modelo in modelos:
          captions = dados.get(modelo) or []
          caption = captions[i] if i < len(captions) else None

          print(f"    [{modelo}] {(caption or '[sem legenda]')[:80]}...")

          avaliacao = avaliador.avaliar_segmento(caption or "", referencias)
          resultado_seg["avaliacoes"][modelo] = {
              "caption": caption,
              **avaliacao
          }

          s = avaliacao["scores"]
          valido = "✓" if avaliacao["valido"] else "⚠"
          print(
              f"      {valido} "
              f"Acc:{s['accuracy']}  "
              f"Comp:{s['completeness']}  "
              f"Conc:{s['conciseness']}  "
              f"Rel:{s['relevance']}  "
              f"→ Média:{avaliacao['media']:.1f}"
          )

      resultados_segmentos.append(resultado_seg)

  return resultados_segmentos

# ─────────────────────────────────────────────────────────────────
# Relatório consolidado
# ─────────────────────────────────────────────────────────────────

DIMENSOES = ["accuracy", "completeness", "conciseness", "relevance"]

def _agregar_scores(avaliacoes: List[dict]) -> dict:
  """Calcula média, mín e máx para cada dimensão e a média geral."""
  por_dim = {d: [] for d in DIMENSOES}
  medias  = []

  for av in avaliacoes:
      for d in DIMENSOES:
          v = av["scores"].get(d)
          if v is not None:
              por_dim[d].append(v)
      if av["media"] > 0:
          medias.append(av["media"])

  resultado = {}
  for d in DIMENSOES:
      vals = por_dim[d]
      resultado[d] = {
          "media": round(sum(vals) / len(vals), 2) if vals else 0.0,
          "min":   min(vals) if vals else 0,
          "max":   max(vals) if vals else 0,
      }

  resultado["media_geral"] = round(sum(medias) / len(medias), 2) if medias else 0.0
  return resultado

def _imprimir_tabela_resumo(
  modelos: List[str],
  metricas_por_modelo: dict,
  todos_resultados: dict,
) -> None:
  """Imprime tabela resumo de scores ACCR e breakdown por vídeo no terminal."""
  L = 72

  # ── Tabela geral (média + intervalo min–max) ──────────────────
  print("\n" + "─" * L)
  print(f"{'MODELO':<20} {'Accuracy':>10} {'Completeness':>13} {'Conciseness':>12} {'Relevance':>10} {'Média':>7}")
  print(f"{'':20} {'[min–max]':>10} {'[min–max]':>13} {'[min–max]':>12} {'[min–max]':>10}")
  print("─" * L)

  for modelo in modelos:
      ag = metricas_por_modelo[modelo]

      def _faixa(dim: str) -> str:
          return f"{ag[dim]['min']}–{ag[dim]['max']}"

      print(
          f"{modelo:<20} "
          f"{ag['accuracy']['media']:>10.1f} "
          f"{ag['completeness']['media']:>13.1f} "
          f"{ag['conciseness']['media']:>12.1f} "
          f"{ag['relevance']['media']:>10.1f} "
          f"{ag['media_geral']:>7.1f}"
      )
      print(
          f"{'':20} "
          f"{_faixa('accuracy'):>10} "
          f"{_faixa('completeness'):>13} "
          f"{_faixa('conciseness'):>12} "
          f"{_faixa('relevance'):>10}"
      )

  # ── Tabela por vídeo ─────────────────────────────────────────
  print("\n" + "─" * L)
  print("SCORES POR VÍDEO (média ACCR por modelo)")
  print("─" * L)

  header = f"{'VIDEO_ID':<35}"
  for m in modelos:
      header += f" {m[:12]:>12}"
  print(header)
  print("─" * L)

  for video_id, segmentos in todos_resultados.items():
      linha = f"{video_id:<35}"
      for modelo in modelos:
          vals = [
              seg["avaliacoes"][modelo]["media"]
              for seg in segmentos
              if modelo in seg.get("avaliacoes", {})
              and seg["avaliacoes"][modelo].get("media", 0) > 0
          ]
          media_vid = round(sum(vals) / len(vals), 1) if vals else 0.0
          linha += f" {media_vid:>12.1f}"
      print(linha)

  print("─" * L)


def gerar_relatorio(
  todos_resultados: dict,
  arquivo_saida: str,
  modelos: List[str]
) -> dict:
  """Gera JSON de relatório e imprime resumo no terminal."""

  num_videos    = len(todos_resultados)
  num_segmentos = sum(len(segs) for segs in todos_resultados.values())

  # Coleta todas as avaliações por modelo
  avaliacoes_por_modelo = {m: [] for m in modelos}

  for segmentos in todos_resultados.values():
      for seg in segmentos:
          for modelo in modelos:
              av = seg["avaliacoes"].get(modelo)
              if av:
                  avaliacoes_por_modelo[modelo].append(av)

  metricas_por_modelo = {
      m: _agregar_scores(avaliacoes_por_modelo[m])
      for m in modelos
  }

  relatorio = {
      "dataset":              "ActivityNet",
      "modelos_avaliados":    modelos,
      "num_videos":           num_videos,
      "num_segmentos":        num_segmentos,
      "dimensoes_avaliadas":  DIMENSOES,
      "metricas_por_modelo":  metricas_por_modelo,
      "resultados_por_video": todos_resultados
  }

  # Escrita atômica: evita JSON corrompido se interrompido
  tmp = Path(arquivo_saida).with_suffix(".json.tmp")
  tmp.write_text(
      json.dumps(relatorio, ensure_ascii=False, indent=2),
      encoding="utf-8",
  )
  tmp.replace(arquivo_saida)

  # ── Impressão do resumo ───────────────────────────────────────
  print("\n" + "="*60)
  print("RELATÓRIO ACCR — AVALIAÇÃO COM LLM COMO AVALIADOR")
  print("="*60)
  print(f"Dataset     : ActivityNet")
  print(f"Vídeos      : {num_videos}")
  print(f"Segmentos   : {num_segmentos}")
  print(f"Dimensões   : Accuracy · Completeness · Conciseness · Relevance")

  _imprimir_tabela_resumo(modelos, metricas_por_modelo, todos_resultados)

  print(f"\n✓ Relatório salvo em: {arquivo_saida}")
  return relatorio

# ─────────────────────────────────────────────────────────────────
# Main interativo
# ─────────────────────────────────────────────────────────────────

def main(args=None):
  print("=" * 60)
  print("AVALIAÇÃO ACCR — LLM COMO AVALIADOR")
  print("Accuracy · Completeness · Conciseness · Relevance (0–100)")
  print("=" * 60)

  # ── Caminhos e opções — CLI tem prioridade sobre defaults ──────
  BASE = os.path.dirname(__file__)

  ARQUIVO_AGENTE = (
      args.predictions if args and args.predictions
      else os.path.join(BASE, "../../output/predictions/predictions_gpt.json")
  )
  ARQUIVOS_GT = (
      args.gt if args and args.gt
      else [
          os.path.join(BASE, "../../data/ground_truth/anet_entities_test_1.json"),
          os.path.join(BASE, "../../data/ground_truth/anet_entities_test_2.json"),
      ]
  )
  ARQUIVO_SKIMCAP = (
      args.skimcap if args and args.skimcap
      else os.path.join(BASE, "../../data/baselines/greedy_pred_test.json")
  )
  MODELO_NOME    = args.modelo_nome if args and hasattr(args, "modelo_nome") else "Modelo1"
  INCLUIR_SKIMCAP = bool(args and args.skimcap)
  OUTPUT_DIR     = args.output_dir if args and hasattr(args, "output_dir") else os.path.join(BASE, "../../output/metrics/accr")
  ARQUIVO_AGENTE2 = (
      args.predictions2
      if args and hasattr(args, "predictions2") and args.predictions2
      else None
  )
  MODELO2_NOME = (
      args.modelo2_nome
      if args and hasattr(args, "modelo2_nome") and args.modelo2_nome and ARQUIVO_AGENTE2
      else None
  )

  # 1. Primeiro modelo
  print("\n1. Carregando resultados do primeiro modelo...")
  if not os.path.exists(ARQUIVO_AGENTE):
      print(f"   ✗ Arquivo não encontrado: {ARQUIVO_AGENTE}")
      return
  resultados_agente = carregar_json(ARQUIVO_AGENTE)
  print(f"   ✓ {len(resultados_agente.get('videos', []))} vídeos carregados ({MODELO_NOME})")

  # 2. Segundo modelo (opcional)
  resultados_agente2 = None
  if MODELO2_NOME and ARQUIVO_AGENTE2 and os.path.exists(ARQUIVO_AGENTE2):
      resultados_agente2 = carregar_json(ARQUIVO_AGENTE2)
      print(f"\n2. ✓ Segundo modelo carregado ({MODELO2_NOME})")

  # 3. Ground Truth — avalia separadamente para cada arquivo GT
  print("\n3. Carregando Ground Truths...")
  ground_truths = {}
  for arquivo_gt in ARQUIVOS_GT:
      if not os.path.exists(arquivo_gt):
          print(f"   ✗ Arquivo não encontrado: {arquivo_gt}")
          return
      ground_truths[os.path.basename(arquivo_gt)] = carregar_json(arquivo_gt)
      print(f"   ✓ {os.path.basename(arquivo_gt)} carregado")

  # 4. SkimCap
  skimcap = None
  incluir_skimcap = False
  if INCLUIR_SKIMCAP:
      if os.path.exists(ARQUIVO_SKIMCAP):
          skimcap = carregar_json(ARQUIVO_SKIMCAP)
          incluir_skimcap = True
          print("\n4. ✓ SkimCap carregado")
      else:
          print(f"\n4. ⚠ SkimCap não encontrado: {ARQUIVO_SKIMCAP}")
  else:
      print("\n4. SkimCap desativado (INCLUIR_SKIMCAP = False)")

  # 5. Avaliador
  print(f"\n5. LLM Avaliador: {config.EVALUATOR_PROVIDER}")
  avaliador = AvaliadorACCR()
  print(f"   ✓ Avaliador ACCR pronto ({avaliador._model_name()}, delay={avaliador.delay}s)")

  modelos = [MODELO_NOME]
  if MODELO2_NOME:
      modelos.append(MODELO2_NOME)
  if incluir_skimcap:
      modelos.append("SkimCap")  # chave usada em preparar_para_comparacao

  # 6. Avaliação separada por GT
  for nome_gt, ground_truth in ground_truths.items():
      sufixo = os.path.splitext(nome_gt)[0]  # ex: anet_entities_test_1

      print(f"\n{'='*60}")
      print(f"GT: {nome_gt}")
      print(f"{'='*60}")

      print("\n6. Preparando dados...")
      dados = preparar_para_comparacao(
          resultados_agente, ground_truth, skimcap,
          modelo_nome=MODELO_NOME,
          resultados_agente2=resultados_agente2,
          modelo2_nome=MODELO2_NOME
      )
      print(f"   ✓ {len(dados)} vídeos preparados")

      nome_saida      = os.path.join(OUTPUT_DIR, f"accr_predictions_{sufixo}.json")
      nome_checkpoint = os.path.join(OUTPUT_DIR, f"accr_checkpoint_{sufixo}.json")

      # Retoma checkpoint anterior se existir
      todos_resultados = {}
      if os.path.exists(nome_checkpoint):
          with open(nome_checkpoint, "r", encoding="utf-8") as f:
              todos_resultados = json.load(f)
          print(f"\n7. ♻ Checkpoint encontrado: {len(todos_resultados)} vídeo(s) já avaliado(s). Retomando...")
      else:
          print("\n7. Avaliando vídeos com ACCR...")

      for video_id, dados_video in dados.items():
          if video_id in todos_resultados:
              print(f"   ⏭ {video_id} já avaliado (checkpoint). Pulando.")
              continue
          resultados_video = avaliar_video(video_id, dados_video, modelos, avaliador)
          todos_resultados[video_id] = resultados_video
          # Escrita atômica do checkpoint após cada vídeo
          tmp_ck = Path(nome_checkpoint).with_suffix(".json.tmp")
          tmp_ck.write_text(
              json.dumps(todos_resultados, ensure_ascii=False, indent=2),
              encoding="utf-8",
          )
          tmp_ck.replace(nome_checkpoint)
          print(f"   💾 Checkpoint salvo ({len(todos_resultados)}/{len(dados)} vídeos)")

      print("\n8. Gerando relatório...")
      gerar_relatorio(todos_resultados, nome_saida, modelos)

      # Remove checkpoint após relatório final gerado com sucesso
      if os.path.exists(nome_checkpoint):
          os.remove(nome_checkpoint)
          print(f"   🗑 Checkpoint removido: {nome_checkpoint}")


  print("\n✨ Avaliação concluída!")

if __name__ == "__main__":
  _BASE = Path(__file__).parent

  _parser = argparse.ArgumentParser(
      description="Avaliação ACCR — LLM como avaliador de legendas de vídeo."
  )
  _parser.add_argument(
      "--predictions", "-p", type=str,
      default=str(_BASE / "../../output/predictions/predictions_gpt.json"),
      help="predictions.json do modelo principal (ex: LLaMA)",
  )
  _parser.add_argument(
      "--predictions2", "-p2", type=str, default=None,
      help="predictions.json do segundo modelo (ex: GPT-4.1)",
  )
  _parser.add_argument(
      "--gt", "-g", type=str, nargs="+",
      default=[
          str(_BASE / "../../data/ground_truth/anet_entities_test_1.json"),
          str(_BASE / "../../data/ground_truth/anet_entities_test_2.json"),
      ],
      help="Arquivo(s) de ground truth (pode ser múltiplos)",
  )
  _parser.add_argument(
      "--skimcap", "-s", type=str, default=None,
      help="JSON da baseline SkimCap (opcional)",
  )
  _parser.add_argument(
      "--provider", "-m", type=str, default=None,
      choices=["github_gpt41", "github_llama", "github_phi"],
      help=f"Modelo avaliador (padrão: {config.EVALUATOR_PROVIDER})",
  )
  _parser.add_argument(
      "--modelo-nome", type=str, default="Modelo1",
      help="Rótulo do modelo principal no relatório (padrão: Modelo1)",
  )
  _parser.add_argument(
      "--modelo2-nome", type=str, default="Modelo2",
      help="Rótulo do segundo modelo no relatório (padrão: Modelo2)",
  )
  _parser.add_argument(
      "--output-dir", "-o", type=str,
      default=str(_BASE / "../../output/metrics/accr"),
      help="Pasta de saída para relatórios ACCR",
  )
  _args = _parser.parse_args()

  # Injeta provider no config se fornecido via CLI
  if _args.provider:
      config.EVALUATOR_PROVIDER = _args.provider

  main(_args)