"""
BACKEND de Video Captioning com GPT-4 Vision

Motor do sistema que processa vídeos e gera legendas.

ARQUITETURA:
Módulo 1: Ingestão       → baixar_video()
Módulo 2: Processamento  → extrair_frames()
Módulo 3: Captioning     → gerar_legenda()
Módulo 4: Persistência   → salvar_resultados()

FLUXO:
URL → Baixa → Segmenta → Extrai frames → GPT-4 → Legenda → JSON

NOTA:
Este backend NÃO faz avaliação.
Para avaliar, use avaliar_legendas.py
"""

import os
import json
import base64
import subprocess
import shutil
from datetime import datetime
import sys

if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

import yt_dlp
from openai import OpenAI

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import agente_config as config


class VideoCaptioningAgent:
    """Backend para geração de legendas de vídeos"""

    def __init__(self):
        self.provider = config.PROVIDER

        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai não instalado. Execute: pip install google-genai"
                )
            self.gemini_client = google_genai.Client(api_key=config.GEMINI_API_KEY)
            self.client = None
        else:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.gemini_client = None

        self._criar_diretorios()

    # ─────────────────────────────────────────────────────────────
    # Utilitários internos
    # ─────────────────────────────────────────────────────────────

    def _criar_diretorios(self):
        """Cria as pastas necessárias"""
        for pasta in [config.VIDEOS_DIR, config.FRAMES_DIR, config.OUTPUT_DIR]:
            os.makedirs(pasta, exist_ok=True)

    def _imagem_para_base64(self, caminho):
        """Converte uma imagem local para data URL base64"""
        with open(caminho, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _rotulos_temporais(self, n: int) -> list:
        """
        Gera rótulos descritivos para N frames.
        Ex: n=3 → ['início', 'meio', 'fim']
            n=5 → ['início', '25%', 'meio', '75%', 'fim']
        """
        if n == 1:
            return ["meio"]
        if n == 2:
            return ["início", "fim"]
        if n == 3:
            return ["início", "meio", "fim"]

        rotulos = ["início"]
        for i in range(1, n - 1):
            pct = int(i / (n - 1) * 100)
            rotulos.append(f"{pct}%")
        rotulos.append("fim")
        return rotulos

    # ─────────────────────────────────────────────────────────────
    # Módulo 1: Ingestão
    # ─────────────────────────────────────────────────────────────

    def baixar_video(self, url, video_id=None):
        """
        Baixa vídeo do YouTube.

        Args:
            url: URL do YouTube
            video_id: ID opcional para nomear o arquivo

        Returns:
            Caminho do vídeo baixado, ou None em caso de erro
        """
        try:
            nome_base = video_id if video_id else "%(id)s"
            output_template = os.path.join(config.VIDEOS_DIR, f"{nome_base}.%(ext)s")

            ydl_opts = {
                "outtmpl": output_template,
                "format": "best[ext=mp4]/best",
                "noplaylist": True,
                "quiet": False,
                "no_warnings": False,
                "user_agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "referer": "https://www.youtube.com/",
                "extractor_retries": 3,
                "fragment_retries": 3,
                "skip_unavailable_fragments": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "web"],
                        "player_skip": ["webpage", "config"]
                    }
                }
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)

            print(f"✓ Vídeo baixado: {video_path}")
            return video_path

        except Exception as e:
            print(f"✗ Erro ao baixar vídeo: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Módulo 2: Processamento
    # ─────────────────────────────────────────────────────────────

    def calcular_n_frames(self, t_start: float, t_end: float) -> int:
        """
        1 frame a cada 10 segundos.
        Mínimo: 3  |  Máximo: 8

        Exemplos:
            5s  → 3 frames
            30s → 3 frames
            45s → 4 frames
            90s → 8 frames (cap)
        """
        duracao = t_end - t_start
        n = max(3, int(duracao / 10))
        return min(n, 8)

    def extrair_frames(self, video_path, t_start, t_end, segment_id):
        """
        Extrai N frames espaçados uniformemente no segmento.
        N é proporcional à duração (1 frame/10s, mín 3, máx 8).

        Args:
            video_path: Caminho do vídeo
            t_start: Tempo de início em segundos
            t_end: Tempo de fim em segundos
            segment_id: ID do segmento

        Returns:
            Lista com os caminhos dos frames, ou None em caso de erro
        """
        try:
            segment_dir = os.path.join(config.FRAMES_DIR, f"seg_{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)

            n_frames = self.calcular_n_frames(t_start, t_end)

            # Recuar levemente o t_end para evitar extrair frame além da duração
            t_end_safe = t_end - 0.5

            if n_frames == 1:
                tempos = [(t_start + t_end_safe) / 2]
            else:
                passo = (t_end_safe - t_start) / (n_frames - 1)
                tempos = [t_start + i * passo for i in range(n_frames)]

            caminhos = []
            for i, tempo in enumerate(tempos):
                saida = os.path.join(segment_dir, f"frame_{i:02d}.jpg")

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(tempo),
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    saida,
                ]

                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                caminhos.append(saida)

            print(f"    → {n_frames} frames extraídos "
                  f"({t_end - t_start:.0f}s de duração)")
            return caminhos

        except Exception as e:
            print(f"✗ Erro ao extrair frames do segmento {segment_id}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Módulo 3: Captioning
    # ─────────────────────────────────────────────────────────────

    def gerar_legenda(self, frames):
        """Dispatcher: envia para OpenAI ou Gemini conforme config.PROVIDER"""
        if self.provider == "gemini":
            return self._gerar_legenda_gemini(frames)
        return self._gerar_legenda_openai(frames)

    def _gerar_legenda_openai(self, frames):
        """
        Gera legenda usando GPT Vision com N frames do segmento.

        Args:
            frames: lista com caminhos de imagens

        Returns:
            String com a legenda gerada, ou None em caso de erro
        """
        try:
            if not frames:
                return None

            n = len(frames)
            rotulos = self._rotulos_temporais(n)

            content = [{"type": "text", "text": config.CAPTION_PROMPT}]

            for i, caminho in enumerate(frames):
                content.append({
                    "type": "text",
                    "text": f"[Frame {i + 1}/{n} — {rotulos[i]}]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._imagem_para_base64(caminho),
                        "detail": "low"
                    }
                })

            response = self.client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"✗ Erro ao gerar legenda (OpenAI): {e}")
            return None

    def _gerar_legenda_gemini(self, frames):
        """
        Gera legenda usando Gemini Vision com N frames do segmento.

        Args:
            frames: lista com caminhos de imagens

        Returns:
            String com a legenda gerada, ou None em caso de erro
        """
        try:
            if not frames:
                return None

            n = len(frames)
            rotulos = self._rotulos_temporais(n)

            partes = [config.CAPTION_PROMPT]
            for i, caminho in enumerate(frames):
                partes.append(f"\n[Frame {i + 1}/{n} — {rotulos[i]}]")
                with open(caminho, "rb") as f:
                    dados = f.read()
                partes.append(
                    genai_types.Part.from_bytes(data=dados, mime_type="image/jpeg")
                )

            response = self.gemini_client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=partes,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=300,
                    temperature=0.3
                )
            )

            return response.text.strip()

        except Exception as e:
            print(f"✗ Erro ao gerar legenda (Gemini): {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Módulo 4: Persistência
    # ─────────────────────────────────────────────────────────────

    def processar_segmento(self, video_path, t_start, t_end, segment_id):
        """
        Processa um segmento do vídeo:
        1. Extrai os frames
        2. Gera a legenda

        Returns:
            Dicionário com os resultados do segmento
        """
        print(f"  Segmento {segment_id}: [{t_start:.1f}s - {t_end:.1f}s]")

        frames = self.extrair_frames(video_path, t_start, t_end, segment_id)
        if not frames:
            return {
                "segment_id": segment_id,
                "timestamps": [t_start, t_end],
                "caption": None,
                "frames": None,
                "error": "Falha ao extrair frames"
            }

        legenda = self.gerar_legenda(frames)

        if legenda:
            print(f"  ✓ Legenda: {legenda[:80]}...")
        else:
            print("  ✗ Não foi possível gerar legenda.")

        return {
            "segment_id": segment_id,
            "timestamps": [t_start, t_end],
            "caption": legenda,
            "frames": frames
        }

    def processar_video(self, url, segmentos, video_id=None):
        """
        Processa um vídeo completo com múltiplos segmentos.

        Args:
            url: URL do YouTube
            segmentos: lista de tuplas (t_start, t_end)
            video_id: ID opcional do vídeo

        Returns:
            Dicionário com os resultados do processamento
        """
        print("\n[1/4] Baixando vídeo...")
        video_path = self.baixar_video(url, video_id)

        if not video_path:
            return None

        modelo_ativo = (
            config.GEMINI_MODEL if self.provider == "gemini"
            else config.GPT_MODEL
        )
        print(f"\n[2/4] Extraindo frames de {len(segmentos)} segmentos")
        print(f"[3/4] Gerando legendas com {modelo_ativo} ({self.provider})")

        resultados = []
        for i, (t_start, t_end) in enumerate(segmentos):
            resultado = self.processar_segmento(video_path, t_start, t_end, i)
            resultados.append(resultado)

        print("\n[4/4] Processamento concluído!")

        return {
            "video_id": video_id,
            "video_path": video_path,
            "url": url,
            "num_segments": len(segmentos),
            "results": resultados,
            "processed_at": datetime.now().isoformat()
        }

    def salvar_resultados(self, dados, nome_arquivo=None):
        """
        Salva os resultados em JSON.

        Args:
            dados: dados a salvar
            nome_arquivo: nome do arquivo JSON

        Returns:
            Caminho do arquivo salvo
        """
        try:
            if nome_arquivo is None:
                nome_arquivo = (
                    f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

            output_path = os.path.join(config.OUTPUT_DIR, nome_arquivo)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)

            print(f"\n✓ Resultados salvos em: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Erro ao salvar resultados: {e}")
            return None

    def limpar_cache(self):
        """Remove vídeos e frames temporários"""
        for pasta in [config.VIDEOS_DIR, config.FRAMES_DIR]:
            if os.path.exists(pasta):
                shutil.rmtree(pasta)
                os.makedirs(pasta, exist_ok=True)
        print("✓ Cache limpo")


# ─────────────────────────────────────────────────────────────────
# Exemplos de uso
# ─────────────────────────────────────────────────────────────────

def processar_dataset(
    videos_json="../outros/scripts/videos_com_urls.json",
    gt_json="../outros/descricoes/descricoes GT/anet_entities_test_1.json",
    output_file="predictions.json",
    limite=None
):
    """
    Processa o dataset real usando os timestamps do ground truth.

    - Retoma automaticamente de onde parou (pula vídeos já processados).
    - Salva no formato esperado por avaliar_legendas.py:
        { "videos": [ { "video_id": ..., "segments": [{"caption": ...}] } ] }

    Args:
        videos_json: caminho para videos_com_urls.json
        gt_json: caminho para o ground truth com timestamps
        output_file: nome do arquivo de saída em output/
        limite: número máximo de vídeos a processar (None = todos)
    """
    # Carregar dados
    with open(videos_json, "r") as f:
        videos_lista = json.load(f)

    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    # Montar mapa video_id → url
    url_map = {v["video_id"]: v["url"] for v in videos_lista}

    # Carregar progresso anterior (se existir)
    output_path = os.path.join(config.OUTPUT_DIR, output_file)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            saida = json.load(f)
        ja_processados = {v["video_id"] for v in saida.get("videos", [])}
        print(f"↩ Retomando: {len(ja_processados)} vídeos já processados.")
    else:
        saida = {"videos": []}
        ja_processados = set()

    agente = VideoCaptioningAgent()

    # Filtrar vídeos que têm URL e timestamps no GT
    pendentes = [
        vid for vid in url_map
        if vid in gt_data and vid not in ja_processados
    ]

    if limite:
        pendentes = pendentes[:limite]

    total = len(pendentes)
    print(f"\nVídeos a processar: {total}")

    for i, video_id in enumerate(pendentes, 1):
        url = url_map[video_id]
        timestamps = gt_data[video_id]["timestamps"]
        segmentos = [(t[0], t[1]) for t in timestamps]

        print(f"\n{'=' * 60}")
        print(f"[{i}/{total}] {video_id}")
        print(f"{'=' * 60}")

        resultado = agente.processar_video(url, segmentos, video_id=video_id)

        if resultado is None:
            print(f"✗ Falha ao processar {video_id}, pulando.")
            continue

        # Converter para o formato de avaliação
        entrada = {
            "video_id": video_id,
            "url": url,
            "segments": [
                {
                    "segment_id": r["segment_id"],
                    "timestamps": r["timestamps"],
                    "caption": r["caption"]
                }
                for r in resultado["results"]
            ]
        }

        saida["videos"].append(entrada)

        # Salvar após cada vídeo (permite retomar)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(saida, f, ensure_ascii=False, indent=2)

        print(f"✓ Salvo ({len(saida['videos'])} vídeos no total)")

    print(f"\n✓ Dataset processado. Resultado em: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BACKEND DE VIDEO CAPTIONING - GPT Vision")
    print("=" * 60)
    print("\nEste é o BACKEND - motor de processamento de vídeos.")
    print("Responsabilidade: gerar legendas com IA generativa.")
    print("\nPara avaliar as legendas geradas, use:")
    print("  cd ..")
    print("  python avaliar_legendas.py")
    print("=" * 60)

    processar_dataset(limite=3)