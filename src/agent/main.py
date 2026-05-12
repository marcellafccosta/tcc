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

import argparse
import os
import json
import base64
import re
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

import yt_dlp
from azure.ai.inference.models import (
    ImageContentItem,
    ImageUrl,
    TextContentItem,
    UserMessage,
)
from azure.core.exceptions import HttpResponseError

import config
from token_manager import GerenciadorTokens


class VideoCaptioningAgent:
    """Backend para geração de legendas de vídeos"""

    def __init__(self, provider: str = None, max_workers: int = 2):
        self.provider = provider or config.PROVIDER
        # Número de segmentos processados em paralelo (limitado pela cota de API)
        self.max_workers = max_workers

        # Valida FFmpeg antes de qualquer processamento
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "FFmpeg não encontrado. Instale com:\n"
                "  macOS:  brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg\n"
                "  Windows: https://ffmpeg.org/download.html"
            )

        # Gerenciador de tokens com rotação automática e timeout
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
            limite_diario=50,
            timeout=config.LLM_TIMEOUT,
        )

        self._criar_diretorios()

    # ─────────────────────────────────────────────────────────────
    # Utilitários internos
    # ─────────────────────────────────────────────────────────────

    def _model_name(self) -> str:
        """Retorna o nome do modelo conforme o provider"""
        return {
            "github_gpt41": config.GITHUB_GPT41,
            "github_llama": config.GITHUB_LLAMA,
            "github_phi":   config.GITHUB_PHI,
        }.get(self.provider, config.GITHUB_GPT41)

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

    def _validar_frames(self, frames: list) -> list:
        """Filtra frames que existem e têm tamanho > 0."""
        validos = [f for f in frames if Path(f).exists() and Path(f).stat().st_size > 0]
        if len(validos) < len(frames):
            print(f"  ⚠️  {len(frames) - len(validos)} frame(s) inválidos ignorados")
        return validos

    def _github_disponivel(self) -> bool:
        """Verifica se há algum token GitHub disponível."""
        return not self._tokens.todos_esgotados()

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
        Mínimo: 1
        """
        duracao = t_end - t_start
        n = max(1, int(duracao / 10))
        return n

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
        if not frames:
            return None
        # Valida frames antes de chamar a API
        frames_validos = self._validar_frames(frames)
        if not frames_validos:
            print("✗ Nenhum frame válido para gerar legenda")
            return None
        if not self._github_disponivel():
            print("✗ Todos os tokens GitHub esgotaram ou falharam")
            return None
        return self._gerar_legenda_github(frames_validos)

    def _gerar_legenda_github(self, frames) -> str | None:
        """
        Gera legenda usando GitHub Models (Azure SDK).
        Trata rate limit por minuto e por dia via GerenciadorTokens.
        """
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

        n_tokens = len(self._tokens._clients)
        for tentativa in range(n_tokens):
            try:
                cliente = self._tokens.cliente_atual()
                response = cliente.complete(
                    messages=[UserMessage(content=content)],
                    model=self._model_name(),
                    max_tokens=300,
                    temperature=0.3,
                )
                chamadas = self._tokens.registrar_chamada()
                print(f"    [Token {self._tokens.idx + 1}: {chamadas}/{self._tokens.limite_diario} req]")
                texto = response.choices[0].message.content.strip()
                # Remove marcadores de frame caso o modelo os inclua na resposta
                texto = re.sub(r"\[Frame \d+/\d+[^\]]*\]\s*", "", texto).strip()
                return texto if texto else None

            except HttpResponseError as e:
                status = e.status_code if hasattr(e, "status_code") else 0
                mensagem = str(e).lower()
                if status == 429 or "too many requests" in mensagem:
                    print(f"  ⚠️  Rate limit (429) — alternando token (tentativa {tentativa + 1}/{n_tokens})")
                    tem_proximo = self._tokens.marcar_esgotado()
                    if not tem_proximo:
                        print("✗ Todos os tokens esgotados")
                        return None
                    continue
                if status in (502, 503, 500) or "bad gateway" in mensagem or "service unavailable" in mensagem or "internal server error" in mensagem or "unexpected eof" in mensagem:
                    espera = 20 * (tentativa + 1)
                    print(f"  ⚠️  {status} Bad Gateway — aguardando {espera}s (tentativa {tentativa + 1}/{n_tokens})")
                    time.sleep(espera)
                    continue
                print(f"✗ Erro HTTP GitHub ({status}): {e}")
                return None

            except RuntimeError as e:
                print(f"✗ {e}")
                return None

            except Exception as e:
                mensagem = str(e).lower()
                if "expecting value" in mensagem or "json is invalid" in mensagem:
                    print(f"  ⚠️  Rate limit (HTML) — alternando token (tentativa {tentativa + 1}/{n_tokens})")
                    tem_proximo = self._tokens.marcar_esgotado()
                    if not tem_proximo:
                        print("✗ Todos os tokens esgotados")
                        return None
                    continue
                print(f"✗ Erro GitHub inesperado: {e}")
                return None

        print("✗ Todos os tokens esgotados após todas as tentativas")
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

        # Processa segmentos com ThreadPoolExecutor para paralelismo limitado.
        # max_workers=2 respeita ~10 req/min do GitHub sem risco de colisão.
        resultados_map: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futuros = {
                ex.submit(self.processar_segmento, video_path, t_start, t_end, i): i
                for i, (t_start, t_end) in enumerate(segmentos)
            }
            for futuro in as_completed(futuros):
                i = futuros[futuro]
                try:
                    resultados_map[i] = futuro.result()
                except Exception as e:
                    print(f"✗ Erro no segmento {i}: {e}")
                    resultados_map[i] = {
                        "segment_id": i,
                        "timestamps": list(segmentos[i]),
                        "caption": None,
                        "frames": None,
                        "error": str(e),
                    }

        resultados = [resultados_map[i] for i in sorted(resultados_map)]
        print("\n[4/4] Processamento concluído!")

        return {
            "video_id": video_id,
            "video_path": video_path,
            "url": url,
            "num_segments": len(segmentos),
            "results": resultados,
            "processed_at": datetime.now().isoformat(),
        }

    def salvar_resultados(self, dados, nome_arquivo=None) -> str | None:
        """Salva os resultados em JSON usando escrita atômica (tmp → rename)."""
        try:
            if nome_arquivo is None:
                nome_arquivo = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            output_path = Path(config.OUTPUT_DIR) / nome_arquivo
            tmp_path = output_path.with_suffix(".json.tmp")

            tmp_path.write_text(
                json.dumps(dados, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(output_path)  # atômico no mesmo filesystem

            print(f"\n✓ Resultados salvos em: {output_path}")
            return str(output_path)

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
    disponiveis_json=str(Path(__file__).parent / "../../data/videos_disponiveis.json"),
    videos_json=str(Path(__file__).parent / "../../data/videos_com_urls.json"),
    gt_json=str(Path(__file__).parent / "../../data/ground_truth/anet_entities_test_1.json"),
    output_file="predictions/predictions_gpt.json",
    limite=None,
    max_workers=2,
    provider2=None,
    output_file2=None,
    retry_nulls=False,
):
    """
    Processa o dataset com um ou dois modelos de geração.

    Quando provider2 é fornecido:
    - O vídeo é baixado e os frames são extraídos UMA SÓ VEZ (modelo 1).
    - O modelo 2 reutiliza os mesmos frames, gerando legendas independentes.
    - Cada modelo salva seu próprio arquivo JSON de saída.

    Retoma automaticamente de onde parou (checkpoint por modelo).
    """
    with open(disponiveis_json, "r") as f:
        ids_disponiveis = json.load(f)

    with open(videos_json, "r") as f:
        videos_lista = json.load(f)

    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    url_map = {v["video_id"]: v["url"] for v in videos_lista}

    output_path = Path(config.OUTPUT_DIR) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, "r") as f:
            saida = json.load(f)
        ja_processados = {v["video_id"] for v in saida.get("videos", [])}
        if retry_nulls:
            com_null = {
                v["video_id"]
                for v in saida.get("videos", [])
                if any(s.get("caption") is None for s in v.get("segments", []))
            }
            if com_null:
                print(f"↩ retry-nulls: re-processando {len(com_null)} vídeo(s) com caption null: {com_null}")
                saida["videos"] = [v for v in saida["videos"] if v["video_id"] not in com_null]
                ja_processados -= com_null
                tmp = output_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(output_path)
            else:
                print("↩ retry-nulls: nenhum caption null encontrado.")
        print(f"↩ Retomando: {len(ja_processados)} vídeos já processados.")
    else:
        saida = {"videos": []}
        ja_processados = set()

    # ── Agente 1 (modelo principal) ───────────────────────────────
    agente1 = VideoCaptioningAgent(max_workers=max_workers)

    # ── Agente 2 (modelo de comparação, opcional) ─────────────────
    agente2 = None
    saida2: dict | None = None
    ja_processados2: set = set()
    output_path2: Path | None = None

    if provider2:
        agente2 = VideoCaptioningAgent(provider=provider2, max_workers=1)
        nome2 = output_file2 or f"predictions/predictions_{provider2.replace('github_', '')}.json"
        output_path2 = Path(config.OUTPUT_DIR) / nome2
        output_path2.parent.mkdir(parents=True, exist_ok=True)
        if output_path2.exists():
            with open(output_path2, "r") as f:
                saida2 = json.load(f)
            ja_processados2 = {v["video_id"] for v in saida2.get("videos", [])}
            print(f"↩ Modelo 2: {len(ja_processados2)} vídeos já processados.")
        else:
            saida2 = {"videos": []}
        print(f"\nModelos: {agente1.provider} (1)  +  {agente2.provider} (2)")
    else:
        print(f"\nModelo: {agente1.provider}")

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

        # ── Modelo 1: download + extração de frames + legenda ────────
        resultado = agente1.processar_video(url, segmentos, video_id=video_id)

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
        tmp = Path(output_path).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(output_path)
        print(f"✓ [{agente1.provider}] Salvo ({len(saida['videos'])} vídeos)")

        # ── Modelo 2: reutiliza frames já extraídos ───────────────────
        if agente2 and saida2 is not None and video_id not in ja_processados2:
            print(f"\n  [Modelo 2: {agente2.provider}] Gerando legendas com frames já extraídos...")
            segmentos2 = []
            for r in resultado["results"]:
                frames = r.get("frames") or []
                legenda2 = agente2.gerar_legenda(frames) if frames else None
                segmentos2.append({
                    "segment_id": r["segment_id"],
                    "timestamps": r["timestamps"],
                    "caption": legenda2,
                })
                if len(segmentos2) < len(resultado["results"]):
                    time.sleep(6.5)  # respeita rate limit entre segmentos do modelo 2

            entrada2 = {"video_id": video_id, "url": url, "segments": segmentos2}
            saida2["videos"].append(entrada2)
            ja_processados2.add(video_id)

            tmp2 = output_path2.with_suffix(".json.tmp")
            tmp2.write_text(json.dumps(saida2, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp2.replace(output_path2)
            print(f"  ✓ [{agente2.provider}] Salvo ({len(saida2['videos'])} vídeos)")

    print(f"\n✓ Modelo 1 ({agente1.provider}): {output_path}")
    if output_path2:
        print(f"✓ Modelo 2 ({agente2.provider}): {output_path2}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _BASE = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Backend de Video Captioning — gera legendas via GitHub Models."
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Número máximo de vídeos a processar (padrão: todos)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="predictions/predictions_gpt.json",
        help="Nome do arquivo de saída em output/ (padrão: predictions/predictions_gpt.json)",
    )
    parser.add_argument(
        "--provider", "-m", type=str, default=None,
        choices=["github_gpt41", "github_llama", "github_phi"],
        help=f"Modelo principal (padrão: {config.PROVIDER})",
    )
    parser.add_argument(
        "--provider2", "-m2", type=str, default=None,
        choices=["github_gpt41", "github_llama", "github_phi"],
        help=f"Segundo modelo de geração para comparação (padrão: {config.PROVIDER_2})",
    )
    parser.add_argument(
        "--output2", type=str, default=None,
        help="Arquivo de saída do segundo modelo em output/ (padrão: predictions_<provider2>.json)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Número de segmentos em paralelo (padrão: 2)",
    )
    parser.add_argument(
        "--disponiveis", type=str,
        default=str(_BASE / "../../data/videos_disponiveis.json"),
        help="JSON com IDs de vídeos disponíveis",
    )
    parser.add_argument(
        "--videos", type=str,
        default=str(_BASE / "../../data/videos_com_urls.json"),
        help="JSON com mapeamento video_id → URL",
    )
    parser.add_argument(
        "--gt", type=str,
        default=str(_BASE / "../../data/ground_truth/anet_entities_test_1.json"),
        help="JSON de ground truth com timestamps",
    )
    parser.add_argument(
        "--retry-nulls", action="store_true",
        help="Re-processa somente segmentos com caption null no arquivo de saída existente",
    )
    _args = parser.parse_args()

    print("=" * 60)
    print("BACKEND DE VIDEO CAPTIONING - GitHub Models")
    print("=" * 60)

    # Injeta providers no config para que os agentes usem os valores do CLI
    if _args.provider:
        config.PROVIDER = _args.provider
    if _args.provider2:
        config.PROVIDER_2 = _args.provider2

    processar_dataset(
        disponiveis_json=_args.disponiveis,
        videos_json=_args.videos,
        gt_json=_args.gt,
        output_file=_args.output,
        limite=_args.limit,
        max_workers=_args.workers,
        provider2=_args.provider2 or config.PROVIDER_2,
        output_file2=_args.output2,
        retry_nulls=_args.retry_nulls,
    )
