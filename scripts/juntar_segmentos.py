"""
Concatena os segmentos de cada vídeo em um parágrafo único.
Gera arquivos com a mesma estrutura, mas com um único "segment" por vídeo.

Entrada:
  - output/predictions/predictions_gpt.json
  - output/predictions/predictions_llama.json
  - data/baselines/greedy_pred_test.json (SkimCap)

Saída:
  - output/predictions/predictions_gpt_full.json
  - output/predictions/predictions_llama_full.json
  - data/baselines/greedy_pred_test_full.json
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def juntar_agente(arquivo_entrada: Path, arquivo_saida: Path) -> None:
    """Concatena segmentos de predictions no formato do agente (GPT/LLaMA)."""
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video in data.get("videos", []):
        segments = video.get("segments", [])
        if not segments:
            continue

        caption_completa = " ".join(seg["caption"] for seg in segments if seg.get("caption"))
        ts_inicio = segments[0]["timestamps"][0]
        ts_fim = segments[-1]["timestamps"][1]

        video["segments"] = [{
            "segment_id": 0,
            "timestamps": [ts_inicio, ts_fim],
            "caption": caption_completa,
        }]

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ {arquivo_saida.name}: {len(data.get('videos', []))} vídeos")


def juntar_skimcap(arquivo_entrada: Path, arquivo_saida: Path) -> None:
    """Concatena segmentos do SkimCap (formato results dict)."""
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    for video_id, segments in results.items():
        if not segments:
            continue

        caption_completa = " ".join(seg["sentence"] for seg in segments if seg.get("sentence"))
        ts_inicio = segments[0]["timestamp"][0]
        ts_fim = segments[-1]["timestamp"][1]

        gt_completo = " ".join(seg.get("gt_sentence", "") for seg in segments if seg.get("gt_sentence"))

        results[video_id] = [{
            "sentence": caption_completa,
            "timestamp": [ts_inicio, ts_fim],
            "gt_sentence": gt_completo,
        }]

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ {arquivo_saida.name}: {len(results)} vídeos")


def main():
    print("Juntando segmentos em parágrafos por vídeo...\n")

    # GPT
    juntar_agente(
        BASE / "output/predictions/predictions_gpt.json",
        BASE / "output/predictions/predictions_gpt_full.json",
    )

    # LLaMA
    juntar_agente(
        BASE / "output/predictions/predictions_llama.json",
        BASE / "output/predictions/predictions_llama_full.json",
    )

    # SkimCap
    juntar_skimcap(
        BASE / "data/baselines/greedy_pred_test.json",
        BASE / "data/baselines/greedy_pred_test_full.json",
    )

    print("\nPronto! Use os arquivos *_full.json na avaliação ACCR.")


if __name__ == "__main__":
    main()
