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

import json
import os
import re
import sys
import tempfile
import time
from typing import List, Dict, Optional

# Garante que agente_config seja encontrado independente de onde o script é executado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agente"))
# Garante que metricas_automatizadas seja encontrado (mesmo diretório)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

import agente_config as config

# ─────────────────────────────────────────────────────────────────
# Prompt ACCR
# ─────────────────────────────────────────────────────────────────

ACCR_PROMPT_TEMPLATE = """You will be given a caption generated for a short video segment. Your task is to rate the generated caption based on its accuracy in capturing the essential content of the video as described in the reference captions.

Evaluation Criteria:
Score is from 0 to 100 - The generated caption should accurately reflect the content in the reference captions and appropriately describe the key actions or events visible in the video. Annotators should penalize captions that include irrelevant details or omit significant elements indicated in the reference captions and the video.

Evaluation Dimensions:
Accuracy: Does the caption correctly describe the entities and actions shown in the video without errors or hallucinations?
Completeness: Does the caption cover all significant events and aspects of the video, including dynamic actions and possible scene transitions?
Conciseness: Is the caption clear and succinct, avoiding unnecessary details and repetition?
Relevance: Is the caption pertinent to the video content, without including irrelevant information or questions?

Evaluation Steps:
1. Examine the provided reference captions carefully.
 1) Read the full reference captions that describe the overall video content or specific actions.
 2) Review each reference caption thoroughly to understand what aspects of the video they highlight.
2. Read the generated caption.
 1) Carefully read the generated caption that needs to be evaluated.
3. Compare the generated caption with the reference captions and assess how well it captures the essence of the video.
4. Evaluate how accurately and completely the generated caption describes the events and entities shown in the video.
5. Check for the inclusion of irrelevant details or the omission of significant elements.
6. Assign an integer score from 0 to 100 for each dimension.

Reference captions: {reference}
Generated caption: {caption}

Response Format:
You should first give detailed reason for your scores, then end with one sentence per score like this:
..... The Accuracy score is α{{accuracy score}}α.
..... The Completeness score is β{{completeness score}}β.
..... The Conciseness score is ψ{{conciseness score}}ψ.
..... The Relevance score is δ{{relevance score}}δ.

Note: the score must be an integer from 0 to 100 wrapped in the corresponding Greek letter.
Wrap Accuracy score in α
Wrap Completeness score in β
Wrap Conciseness score in ψ
Wrap Relevance score in δ"""

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
  predictions: dict,
  arquivos_gt: List[str],
  arquivo_saida: str,
) -> dict:
  """
  Calcula BLEU-1..4, METEOR, ROUGE-L e CIDEr usando ANETcaptions.
  Converte predictions e GTs para o formato esperado e salva os resultados.
  """
  try:
      from metricas_automatizadas import ANETcaptions
  except ImportError as e:
      print(f"  ✗ metricas_automatizadas não encontrado: {e}")
      return {}

  pred_anet = converter_predictions_para_anet(predictions)
  if not pred_anet["results"]:
      print("  ✗ Nenhuma prediction válida para avaliação")
      return {}

  # Grava predictions em arquivo temporário
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".json", delete=False, encoding="utf-8"
  ) as f:
      json.dump(pred_anet, f)
      pred_path = f.name

  # Converte e grava GTs em arquivos temporários
  gt_paths = []
  for arquivo_gt in arquivos_gt:
      if not os.path.exists(arquivo_gt):
          continue
      gt_anet = converter_gt_para_anet(carregar_json(arquivo_gt))
      with tempfile.NamedTemporaryFile(
          mode="w", suffix=".json", delete=False, encoding="utf-8"
      ) as f:
          json.dump(gt_anet, f)
          gt_paths.append(f.name)

  if not gt_paths:
      os.unlink(pred_path)
      print("  ✗ Nenhum arquivo GT válido encontrado")
      return {}

  try:
      evaluator = ANETcaptions(
          ground_truth_filenames=gt_paths,
          prediction_filename=pred_path,
          verbose=True,
          all_scorer=True,
      )
      evaluator.evaluate()
      scores = evaluator.scores

      with open(arquivo_saida, "w", encoding="utf-8") as f:
          json.dump(scores, f, ensure_ascii=False, indent=2)

      print("\n" + "="*60)
      print("MÉTRICAS AUTOMÁTICAS (BLEU · METEOR · ROUGE-L · CIDEr)")
      print("="*60)
      print(f"{'Métrica':<12} {'Score (0–100)':>14}")
      print("-"*28)
      for metric, score in scores.items():
          print(f"{metric:<12} {100 * score:>14.3f}")
      print(f"\n✓ Métricas salvas em: {arquivo_saida}")
      return scores

  except Exception as e:
      print(f"  ✗ Erro ao calcular métricas: {e}")
      return {}
  finally:
      try:
          os.unlink(pred_path)
      except OSError:
          pass
      for p in gt_paths:
          try:
              os.unlink(p)
          except OSError:
              pass


# ─────────────────────────────────────────────────────────────────

class AvaliadorACCR:
  """
  Usa um LLM como juiz para calcular métricas ACCR.
  Suporta GitHub Models: github_gpt4o | github_gemini | github_llama | github_deepseek
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
          provider: github_gpt4o | github_gemini | github_llama | github_deepseek
                    (padrão: config.EVALUATOR_PROVIDER)
          delay: segundos entre chamadas (padrão: 6.5s para respeitar 10 req/min)
      """
      self.provider = provider or config.EVALUATOR_PROVIDER
      self.delay = delay if delay is not None else self._DELAY_PADRAO

      # Suporte a dois tokens para dobrar a cota diária (50+50=100 req/dia)
      self._clients = []
      self._calls   = []
      for token in [config.GITHUB_TOKEN, getattr(config, "GITHUB_TOKEN_2", ""), getattr(config, "GITHUB_TOKEN_3", "")]:
              self._clients.append(ChatCompletionsClient(
                  endpoint=config.GITHUB_ENDPOINT,
                  credential=AzureKeyCredential(token),
                  retry_total=0,
              ))
              self._calls.append(0)
      self._token_idx = 0

      if not self._clients:
          raise RuntimeError("Nenhum GITHUB_TOKEN configurado")

  def _model_name(self) -> str:
      return {
          "github_gpt4o":    config.GITHUB_GPT4O,
          "github_llama":    config.GITHUB_LLAMA,
          "github_deepseek": config.GITHUB_DEEPSEEK,
      }.get(self.provider, config.GITHUB_GPT4O)

  # ── Chamada ao LLM ────────────────────────────────────────────

  def _cliente_atual(self) -> ChatCompletionsClient:
      """Retorna o cliente do token ativo, alternando quando a cota esgota."""
      for _ in range(len(self._clients)):
          if self._calls[self._token_idx] < self._DAILY_LIMIT:
              return self._clients[self._token_idx]
          print(f"  ⚠️  Token {self._token_idx + 1} esgotado — alternando")
          self._token_idx = (self._token_idx + 1) % len(self._clients)
      raise RuntimeError("Todos os tokens GitHub esgotaram a cota diária")

  def _chamar_llm(self, prompt: str) -> Optional[str]:
      """Envia prompt ao LLM e retorna o texto da resposta."""
      max_tentativas = 3
      for tentativa in range(max_tentativas):
          try:
              cliente = self._cliente_atual()
              response = cliente.complete(
                  messages=[UserMessage(content=prompt)],
                  model=self._model_name(),
                  max_tokens=2048,
                  temperature=0.0,
              )
              self._calls[self._token_idx] += 1
              total = self._calls[self._token_idx]
              print(f"    [Token {self._token_idx + 1}: {total}/{self._DAILY_LIMIT} req]")
              return response.choices[0].message.content

          except RuntimeError:
              raise
          except Exception as e:
              msg = str(e)
              # Detecta rate limit: código 429 OU resposta HTML (JSON inválido com "too many requests")
              is_rate_limit = (
                  '429' in msg
                  or ('json is invalid' in msg.lower() and 'too many requests' in msg.lower())
              )
              m = re.search(r'retry in (\d+)', msg)
              wait = int(m.group(1)) + 5 if m else 65
              if is_rate_limit:
                  # Esgota o token atual e alterna para o próximo imediatamente
                  print(f"  ⚠️  Rate limit — marcando token {self._token_idx + 1} como esgotado")
                  self._calls[self._token_idx] = self._DAILY_LIMIT
                  self._token_idx = (self._token_idx + 1) % len(self._clients)
                  # Verifica se ainda há tokens disponíveis
                  if all(c >= self._DAILY_LIMIT for c in self._calls):
                      print("  ⚠️  Todos os tokens esgotados")
                      return None
              else:
                  print(f"    ✗ Erro LLM ({self.provider}): {e}")
                  return None
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
              "skimcap":    [captions] | None,
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
      if skimcap_data and vid in skimcap_data:
          skimcap_captions = skimcap_data[vid].get("results", [])

      entry = {
          modelo_nome:    captions,
          "ground_truth": gt_captions,
          "skimcap":      skimcap_captions,
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
          captions = dados.get(modelo, [])
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

  with open(arquivo_saida, "w", encoding="utf-8") as f:
      json.dump(relatorio, f, ensure_ascii=False, indent=2)

  # ── Impressão do resumo ───────────────────────────────────────
  print("\n" + "="*60)
  print("RELATÓRIO ACCR — AVALIAÇÃO COM LLM COMO AVALIADOR")
  print("="*60)
  print(f"Dataset     : ActivityNet")
  print(f"Vídeos      : {num_videos}")
  print(f"Segmentos   : {num_segmentos}")
  print(f"Dimensões   : Accuracy · Completeness · Conciseness · Relevance")

  print(f"\n{'MODELO':<20} {'Acc':>6} {'Comp':>6} {'Conc':>6} {'Rel':>6} {'Média':>7}")
  print("-" * 55)

  for modelo in modelos:
      ag = metricas_por_modelo[modelo]
      print(
          f"{modelo:<20} "
          f"{ag['accuracy']['media']:>6.1f} "
          f"{ag['completeness']['media']:>6.1f} "
          f"{ag['conciseness']['media']:>6.1f} "
          f"{ag['relevance']['media']:>6.1f} "
          f"{ag['media_geral']:>7.1f}"
      )

  print(f"\n✓ Relatório salvo em: {arquivo_saida}")
  return relatorio

# ─────────────────────────────────────────────────────────────────
# Main interativo
# ─────────────────────────────────────────────────────────────────

def main():
  print("=" * 60)
  print("AVALIAÇÃO ACCR — LLM COMO AVALIADOR")
  print("Accuracy · Completeness · Conciseness · Relevance (0–100)")
  print("=" * 60)

  # ── Caminhos fixos ────────────────────────────────────────────
  BASE = os.path.dirname(__file__)
  ARQUIVO_AGENTE  = os.path.join(BASE, "../agente/output/predictions.json")
  ARQUIVOS_GT     = [
      os.path.join(BASE, "descricoes/descricoes GT/anet_entities_test_1.json"),
      os.path.join(BASE, "descricoes/descricoes GT/anet_entities_test_2.json"),
  ]
  ARQUIVO_SKIMCAP = os.path.join(BASE, "descricoes/descricoes skimcap/greedy_pred_test.json")
  MODELO_NOME     = "GPT-4.1"
  MODELO2_NOME    = None   # None = sem segundo modelo; ex: "LLaMA" para comparar
  ARQUIVO_AGENTE2 = None   # caminho do segundo modelo se MODELO2_NOME for definido
  INCLUIR_SKIMCAP = False  # True para incluir SkimCap na avaliação

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
      modelos.append("skimcap")  # chave usada em preparar_para_comparacao

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

      nome_saida      = os.path.join(BASE, f"accr_predictions_{sufixo}.json")
      nome_checkpoint = os.path.join(BASE, f"accr_checkpoint_{sufixo}.json")

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
          # Salva checkpoint após cada vídeo
          with open(nome_checkpoint, "w", encoding="utf-8") as f:
              json.dump(todos_resultados, f, ensure_ascii=False, indent=2)
          print(f"   💾 Checkpoint salvo ({len(todos_resultados)}/{len(dados)} vídeos)")

      print("\n8. Gerando relatório...")
      gerar_relatorio(todos_resultados, nome_saida, modelos)

      # Remove checkpoint após relatório final gerado com sucesso
      if os.path.exists(nome_checkpoint):
          os.remove(nome_checkpoint)
          print(f"   🗑 Checkpoint removido: {nome_checkpoint}")

  # 9. Métricas automáticas (BLEU, METEOR, ROUGE-L, CIDEr)
  print("\n" + "="*60)
  print("9. Calculando métricas automáticas...")
  nome_metricas = os.path.join(BASE, "metricas_automatizadas_predictions.json")
  avaliar_metricas_automatizadas(resultados_agente, ARQUIVOS_GT, nome_metricas)

  print("\n✨ Avaliação concluída!")

if __name__ == "__main__":
  main()