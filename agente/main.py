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

# Garantir que o diretório atual está no path
if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

import yt_dlp
from openai import OpenAI

try:
    import google.generativeai as genai
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
                    "google-generativeai não instalado. Execute: pip install google-generativeai"
                )
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_client = genai.GenerativeModel(config.GEMINI_MODEL)
            self.client = None
        else:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.gemini_client = None

        self._criar_diretorios()

    def _criar_diretorios(self):
        """Cria as pastas necessárias"""
        for pasta in [config.VIDEOS_DIR, config.FRAMES_DIR, config.OUTPUT_DIR]:
            os.makedirs(pasta, exist_ok=True)

    def _imagem_para_base64(self, caminho):
        """Converte uma imagem local para data URL base64"""
        with open(caminho, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

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
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
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

    def extrair_frames(self, video_path, t_start, t_end, segment_id):
        """
        Extrai 3 frames de um segmento usando ffmpeg:
        - frame no tempo inicial (t_start)
        - frame no tempo médio ((t_start + t_end) / 2)
        - frame no tempo final (t_end)

        Args:
            video_path: Caminho do vídeo
            t_start: Tempo de início em segundos
            t_end: Tempo de fim em segundos
            segment_id: ID do segmento

        Returns:
            Lista com os caminhos dos 3 frames, ou None em caso de erro
        """
        try:
            segment_dir = os.path.join(config.FRAMES_DIR, f"seg_{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)

            t_mid = (t_start + t_end) / 2

            tempos = [
                ("start.jpg", t_start),
                ("mid.jpg", t_mid),
                ("end.jpg", t_end),
            ]

            caminhos = []

            for nome_arquivo, tempo in tempos:
                saida = os.path.join(segment_dir, nome_arquivo)

                cmd = [
                    "ffmpeg",
                    "-y",
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

            return caminhos

        except Exception as e:
            print(f"✗ Erro ao extrair frames do segmento {segment_id}: {e}")
            return None

    def gerar_legenda(self, frames):
        """Dispatcher: envia para OpenAI ou Gemini conforme config.PROVIDER"""
        if self.provider == "gemini":
            return self._gerar_legenda_gemini(frames)
        return self._gerar_legenda_openai(frames)

    def _gerar_legenda_openai(self, frames):
        """
        Gera legenda usando GPT com 3 frames do segmento.

        Args:
            frames: lista com 3 caminhos de imagens

        Returns:
            String com a legenda gerada, ou None em caso de erro
        """
        try:
            if not frames or len(frames) != 3:
                print("✗ É necessário fornecer exatamente 3 frames.")
                return None

            f1 = self._imagem_para_base64(frames[0])
            f2 = self._imagem_para_base64(frames[1])
            f3 = self._imagem_para_base64(frames[2])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": config.CAPTION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f1}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f2}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f3}
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=messages,
                max_tokens=300,
                temperature=0.3
            )

            legenda = response.choices[0].message.content
            if legenda:
                return legenda.strip()
            return None

        except Exception as e:
            print(f"✗ Erro ao gerar legenda (OpenAI): {e}")
            return None

    def _gerar_legenda_gemini(self, frames):
        """
        Gera legenda usando Gemini Vision com 3 frames do segmento.

        Args:
            frames: lista com 3 caminhos de imagens

        Returns:
            String com a legenda gerada, ou None em caso de erro
        """
        try:
            if not frames or len(frames) != 3:
                print("✗ É necessário fornecer exatamente 3 frames.")
                return None

            partes = [config.CAPTION_PROMPT]
            for caminho in frames:
                with open(caminho, "rb") as f:
                    dados = f.read()
                partes.append({"mime_type": "image/jpeg", "data": dados})

            response = self.gemini_client.generate_content(
                partes,
                generation_config={
                    "max_output_tokens": 300,
                    "temperature": 0.3
                }
            )

            legenda = response.text
            if legenda:
                return legenda.strip()
            return None

        except Exception as e:
            print(f"✗ Erro ao gerar legenda (Gemini): {e}")
            return None

    def processar_segmento(self, video_path, t_start, t_end, segment_id):
        """
        Processa um segmento do vídeo:
        1. Extrai os 3 frames
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

        print(f"\n[2/4] Extraindo frames de {len(segmentos)} segmentos")
        modelo_ativo = config.GEMINI_MODEL if self.provider == "gemini" else config.GPT_MODEL
        print(f"[3/4] Gerando legendas com {modelo_ativo} ({self.provider})")

        resultados = []

        for i, (t_start, t_end) in enumerate(segmentos):
            resultado = self.processar_segmento(video_path, t_start, t_end, i)
            resultados.append(resultado)

        print(f"\n[4/4] Processamento concluído!")

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
                nome_arquivo = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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


def exemplo_uso_simples():
    """Exemplo simples: um vídeo, alguns segmentos"""
    agente = VideoCaptioningAgent()

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # troque pela URL real

    segmentos = [
        (0, 5),
        (5, 10),
        (10, 15)
    ]

    resultados = agente.processar_video(url, segmentos, video_id="exemplo")

    if resultados:
        agente.salvar_resultados(resultados, "exemplo_captions.json")


def exemplo_uso_avancado():
    """Exemplo com múltiplos vídeos"""
    agente = VideoCaptioningAgent()

    videos = [
        {
            "id": "video_001",
            "url": "https://www.youtube.com/watch?v=XXXXXXXXXXX",
            "segments": [(0, 5), (5, 10), (10, 15)]
        },
        {
            "id": "video_002",
            "url": "https://www.youtube.com/watch?v=YYYYYYYYYYY",
            "segments": [(0, 8), (8, 16)]
        }
    ]

    todos_resultados = []

    for video in videos:
        print(f"\n{'=' * 60}")
        print(f"Processando: {video['id']}")
        print(f"{'=' * 60}")

        resultado = agente.processar_video(
            video["url"],
            video["segments"],
            video_id=video["id"]
        )

        if resultado:
            todos_resultados.append(resultado)

    agente.salvar_resultados(
        {"videos": todos_resultados},
        "batch_captions.json"
    )


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

    # Rodar exemplo simples
    exemplo_uso_simples()