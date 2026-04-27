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
import time
from datetime import datetime
import sys

if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

import yt_dlp
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    ImageContentItem,
    ImageUrl,
    TextContentItem,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

import agente_config as config


class VideoCaptioningAgent:
    """Backend para geração de legendas de vídeos"""

    # Controle de rate limit do GitHub Models
    _github_daily_limit = 50

    def __init__(self):
        self.provider = config.PROVIDER

        # Clientes GitHub Models — alterna entre tokens quando um esgota
        self._github_clients = []
        self._github_calls = []
        for token in [config.GITHUB_TOKEN, getattr(config, "GITHUB_TOKEN_2", ""), getattr(config, "GITHUB_TOKEN_3", "")]:
            if token:
                self._github_clients.append(ChatCompletionsClient(
                    endpoint=config.GITHUB_ENDPOINT,
                    credential=AzureKeyCredential(token),
                    retry_total=0,
                ))
                self._github_calls.append(0)
        self._token_idx = 0  # índice do token ativo

        # mantém compatibilidade com código existente
        self.github_client = self._github_clients[0] if self._github_clients else None

        self._criar_diretorios()

    # ─────────────────────────────────────────────────────────────
    # Utilitários internos
    # ─────────────────────────────────────────────────────────────

    def _model_name(self) -> str:
        """Retorna o nome do modelo conforme o provider"""
        return {
            "github_gpt4o":    config.GITHUB_GPT4O,
            "github_llama":    config.GITHUB_LLAMA,
            "github_deepseek": config.GITHUB_DEEPSEEK,
        }.get(self.provider, config.GITHUB_GPT4O)

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

    def _github_disponivel(self) -> bool:
        """
        Verifica se há algum token GitHub disponível.
        Alterna para o próximo token quando o atual esgota a cota diária.
        """
        for _ in range(len(self._github_clients)):
            idx = self._token_idx
            if self._github_calls[idx] < self._github_daily_limit:
                self.github_client = self._github_clients[idx]
                return True
            # Token esgotado — tenta o próximo
            print(f"  ⚠️  Token {idx + 1} esgotado ({self._github_daily_limit} req/dia) — alternando")
            self._token_idx = (self._token_idx + 1) % len(self._github_clients)

        print(f"  ⚠️  Todos os tokens GitHub esgotaram a cota diária")
        return False

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
                        "player_skip": ["webpage", "config"],
                    }
                },
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
        """
        duracao = t_end - t_start
        n = max(3, int(duracao / 10))
        return min(n, 8)

    def extrair_frames(self, video_path, t_start, t_end, segment_id):
        """
        Extrai N frames espaçados uniformemente no segmento.
        N é proporcional à duração (1 frame/10s, mín 3, máx 8).
        """
        try:
            segment_dir = os.path.join(config.FRAMES_DIR, f"seg_{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)

            n_frames = self.calcular_n_frames(t_start, t_end)
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
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(tempo),
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    saida,
                ]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                caminhos.append(saida)

            print(f"    → {n_frames} frames extraídos " f"({t_end - t_start:.0f}s de duração)")
            return caminhos

        except Exception as e:
            print(f"✗ Erro ao extrair frames do segmento {segment_id}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Módulo 3: Captioning
    # ─────────────────────────────────────────────────────────────

    def gerar_legenda(self, frames) -> str | None:
        """Gera legenda usando GitHub Models, tentando todos os tokens disponíveis."""
        for _ in range(len(self._github_clients)):
            if not self._github_disponivel():
                break
            resultado = self._gerar_legenda_github(frames)
            if resultado:
                return resultado
            # _gerar_legenda_github já alternância o token em caso de rate limit
        print("✗ Todos os tokens GitHub esgotaram ou falharam")
        return None

    def _gerar_legenda_github(self, frames) -> str | None:
        """
        Gera legenda usando GitHub Models (Azure SDK).
        Trata rate limit por minuto e por dia.
        """
        try:
            if not frames:
                return None

            n = len(frames)
            rotulos = self._rotulos_temporais(n)

            content = [TextContentItem(text=config.CAPTION_PROMPT)]
            for i, caminho in enumerate(frames):
                content.append(TextContentItem(text=f"[Frame {i + 1}/{n} — {rotulos[i]}]"))
                content.append(
                    ImageContentItem(
                        image_url=ImageUrl(url=self._imagem_para_base64(caminho))
                    )
                )

            response = self.github_client.complete(
                messages=[UserMessage(content=content)],
                model=self._model_name(),
                max_tokens=300,
                temperature=0.3,
            )

            # Contabiliza chamada bem-sucedida no token atual
            self._github_calls[self._token_idx] += 1
            chamadas = self._github_calls[self._token_idx]
            print(f"    [Token {self._token_idx + 1}: {chamadas}/{self._github_daily_limit} req]")

            return response.choices[0].message.content.strip()

        except HttpResponseError as e:
            status = e.status_code if hasattr(e, "status_code") else 0
            mensagem = str(e).lower()

            # Rate limit por minuto (429) — esgota o token atual para forçar alternĉia
            if status == 429 or "too many requests" in mensagem:
                print(f"  ⚠️  Rate limit — marcando token {self._token_idx + 1} como esgotado")
                self._github_calls[self._token_idx] = self._github_daily_limit
                self._token_idx = (self._token_idx + 1) % len(self._github_clients)
                return None

            print(f"✗ Erro HTTP GitHub ({status}): {e}")
            return None

        except Exception as e:
            mensagem = str(e).lower()

            # Resposta HTML em vez de JSON (rate limit retorna página HTML)
            if "expecting value" in mensagem or "json is invalid" in mensagem:
                print(f"  ⚠️  Rate limit (HTML) — marcando token {self._token_idx + 1} como esgotado")
                self._github_calls[self._token_idx] = self._github_daily_limit
                self._token_idx = (self._token_idx + 1) % len(self._github_clients)
                return None

            print(f"✗ Erro GitHub inesperado: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Módulo 4: Persistência
    # ─────────────────────────────────────────────────────────────

    def processar_segmento(self, video_path, t_start, t_end, segment_id):
        """Processa um segmento: extrai frames e gera legenda."""
        print(f"  Segmento {segment_id}: [{t_start:.1f}s - {t_end:.1f}s]")

        frames = self.extrair_frames(video_path, t_start, t_end, segment_id)
        if not frames:
            return {
                "segment_id": segment_id,
                "timestamps": [t_start, t_end],
                "caption": None,
                "frames": None,
                "error": "Falha ao extrair frames",
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
            "frames": frames,
        }

    def processar_video(self, url, segmentos, video_id=None):
        """Processa um vídeo completo com múltiplos segmentos."""
        print("\n[1/4] Baixando vídeo...")
        video_path = self.baixar_video(url, video_id)

        if not video_path:
            return None

        print(f"\n[2/4] Extraindo frames de {len(segmentos)} segmentos")
        print(f"[3/4] Gerando legendas — provider: {self.provider}")

        resultados = []
        for i, (t_start, t_end) in enumerate(segmentos):
            if i > 0:
                time.sleep(6.5)  # respeita 10 req/min do GitHub
            resultado = self.processar_segmento(video_path, t_start, t_end, i)
            resultados.append(resultado)

        print("\n[4/4] Processamento concluído!")

        return {
            "video_id": video_id,
            "video_path": video_path,
            "url": url,
            "num_segments": len(segmentos),
            "results": resultados,
            "processed_at": datetime.now().isoformat(),
        }

    def salvar_resultados(self, dados, nome_arquivo=None):
        """Salva os resultados em JSON."""
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


# ─────────────────────────────────────────────────────────────────
# Processamento do dataset
# ─────────────────────────────────────────────────────────────────


def processar_dataset(
    disponiveis_json="../outros/scripts/videos_disponiveis.json",
    videos_json="../outros/scripts/videos_com_urls.json",
    gt_json="../outros/descricoes/descricoes GT/anet_entities_test_1.json",
    output_file="predictions.json",
    limite=None,
):
    """
    Processa o dataset real usando os timestamps do ground truth.

    - Processa apenas os vídeos listados em videos_disponiveis.json.
    - Retoma automaticamente de onde parou.
    - Salva no formato esperado por avaliar_legendas.py.
    """
    with open(disponiveis_json, "r") as f:
        ids_disponiveis = json.load(f)

    with open(videos_json, "r") as f:
        videos_lista = json.load(f)

    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    url_map = {v["video_id"]: v["url"] for v in videos_lista}

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

    pendentes = [
        vid
        for vid in ids_disponiveis
        if vid in url_map and vid in gt_data and vid not in ja_processados
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

        entrada = {
            "video_id": video_id,
            "url": url,
            "segments": [
                {
                    "segment_id": r["segment_id"],
                    "timestamps": r["timestamps"],
                    "caption": r["caption"],
                }
                for r in resultado["results"]
            ],
        }

        saida["videos"].append(entrada)

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
    print("BACKEND DE VIDEO CAPTIONING - GitHub Models")
    print("=" * 60)
    print("\nEste é o BACKEND - motor de processamento de vídeos.")
    print("Responsabilidade: gerar legendas com IA generativa.")
    print("\nPara avaliar as legendas geradas, use:")
    print("  cd ..")
    print("  python avaliar_legendas.py")
    print("=" * 60)

    processar_dataset(limite=10)
