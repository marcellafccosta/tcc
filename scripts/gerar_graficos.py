#!/usr/bin/env python3
"""
gerar_graficos.py

Gera visualizações comparativas a partir dos outputs da pipeline de
video captioning: métricas automáticas, ACCR e correlações entre eles.

ARQUITETURA:
Módulo 1: Carregamento  → carregar_auto(), carregar_accr(), carregar_segmentos()
Módulo 2: Auto          → grafico_radar(), grafico_barras_gt()
Módulo 3: ACCR          → grafico_heatmap_accr(), grafico_kde_accr(), grafico_barras_accr_gt()
Módulo 4: Correlação    → grafico_correlacao(), grafico_scatter_accr_auto(), grafico_r4_conciseness()
Módulo 5: Rankings      → grafico_rank_stability()
Módulo 6: Análise Av.   → grafico_correlacao_sig(), grafico_bland_altman(),
                          grafico_parallel_coordinates(), grafico_calibration(),
                          grafico_delta_heatmap(), grafico_ranking_agreement(),
                          grafico_hexbin()

FLUXO:
JSONs de output → carregamento → normalização → geração de figuras → output/plots/
"""

import argparse
import itertools
import json
import random as _random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Dependências externas
# ─────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    print("❌ matplotlib não instalado. Execute: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.stats import spearmanr
    from scipy.stats import gaussian_kde
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("⚠️  scipy não instalado — correlações de Spearman e KDE usarão fallback")

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor_fn
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False

# ─────────────────────────────────────────────────────────────
# Caminhos padrão
# ─────────────────────────────────────────────────────────────

_ROOT  = Path(__file__).parent.parent
_AUTO  = _ROOT / "output" / "metrics" / "auto"
_ACCR  = _ROOT / "output" / "metrics" / "accr"
_PLOTS = _ROOT / "output" / "plots"

# Paleta de cores consistente para todos os gráficos
_CORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
# Cores para GT1 / GT2
_CORES_GT = ["#4C72B0", "#DD8452"]

# ─────────────────────────────────────────────────────────────
# Módulo 1: Carregamento
# ─────────────────────────────────────────────────────────────

# Normaliza variantes de nome para uma forma canônica consistente
_MODELO_CANON = {
    "skimcap": "SkimCap",
    "llama":   "LLaMA-4",
    "llama-4": "LLaMA-4",
    "gpt-4.1": "GPT-4.1",
}

def _canon(nome: str) -> str:
    """Retorna o nome canônico do modelo (case-insensitive lookup)."""
    return _MODELO_CANON.get(nome.lower(), nome)


def carregar_auto(auto_dir: Path) -> dict:
    """
    Lê todos os arquivos metricas_{modelo}_{gt}.json do diretório.

    Retorna dict no formato '{modelo} ({gt})': {metrica: valor}.
    """
    dados = {}
    for arquivo in sorted(auto_dir.glob("metricas_*.json")):
        # Ex: metricas_GPT-4.1_anet_entities_test_1.json
        nome = arquivo.stem[len("metricas_"):]
        for gt in ("anet_entities_test_1", "anet_entities_test_2"):
            if nome.endswith(f"_{gt}"):
                modelo = _canon(nome[: -len(f"_{gt}")])
                chave = f"{modelo} ({gt})"
                try:
                    dados[chave] = json.loads(arquivo.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"⚠️  Erro ao ler {arquivo.name}: {e}")
                break
    return dados


def carregar_accr(accr_dir: Path) -> dict:
    """
    Lê arquivos accr_{modelo}_{gt}.json (resumo agregado por modelo).

    Retorna dict no formato '{modelo} ({gt})': {dimensao: valor}.
    Ignora arquivos combinados (accr_predictions_*).
    """
    dados = {}
    for arquivo in sorted(accr_dir.glob("accr_*.json")):
        nome = arquivo.stem[len("accr_"):]
        # Ignora arquivo combinado de predições
        if nome.startswith("predictions_"):
            continue
        for gt in ("anet_entities_test_1", "anet_entities_test_2"):
            if nome.endswith(f"_{gt}"):
                modelo = _canon(nome[: -len(f"_{gt}")])
                chave = f"{modelo} ({gt})"
                try:
                    raw = json.loads(arquivo.read_text(encoding="utf-8"))
                    entrada = {}
                    for dim in ("accuracy", "completeness", "conciseness", "relevance"):
                        if dim in raw:
                            v = raw[dim]
                            entrada[dim] = v["media"] if isinstance(v, dict) else float(v)
                    if "media_geral" in raw:
                        entrada["media_geral"] = float(raw["media_geral"])
                    if entrada:
                        dados[chave] = entrada
                except Exception as e:
                    print(f"⚠️  Erro ao ler {arquivo.name}: {e}")
                break
    return dados


def _parse_sent(texto: str) -> list:
    """Tokenização simples: lowercase + split."""
    return texto.strip().lower().split()


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


def _rouge_l(refs: list, hyp: str) -> float:
    """ROUGE-L máximo entre a hipótese e as referências."""
    h = _parse_sent(hyp)
    if not h:
        return 0.0
    scores = []
    for ref in refs:
        r = _parse_sent(ref)
        if not r:
            continue
        lcs = _lcs(r, h)
        p = lcs / len(h)
        rc = lcs / len(r)
        scores.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)
    return max(scores) if scores else 0.0


def _meteor(refs: list, hyp: str) -> float:
    """METEOR usando nltk, ou -1.0 se indisponível."""
    if not _NLTK_OK or not hyp.strip():
        return -1.0
    return float(_meteor_fn([_parse_sent(r) for r in refs], _parse_sent(hyp)))


def _r4_video(captions: list) -> float:
    """
    R@4 = proporção de 4-gramas distintos que aparece em mais de um
    segmento do mesmo vídeo. Mede repetição entre segmentos (↓ melhor).
    """
    captions = [c for c in captions if c and c.strip()]
    if len(captions) <= 1:
        return 0.0
    ngrams_por_seg = [set(_ngrams(_parse_sent(c), 4)) for c in captions]
    counts: Counter = Counter()
    for ng_set in ngrams_por_seg:
        for ng in ng_set:
            counts[ng] += 1
    total = len(counts)
    repetidos = sum(1 for v in counts.values() if v > 1)
    return repetidos / total if total > 0 else 0.0


def _ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def carregar_segmentos(accr_dir: Path) -> list:
    """
    Extrai dados por segmento dos arquivos accr_predictions_{gt}.json.

    Cada registro retornado contém: video_id, segment, gt, modelo,
    caption, ground_truth, accuracy, completeness, conciseness,
    relevance, rouge_l, meteor, r4_video.
    """
    registros = []
    for gt in ("anet_entities_test_1", "anet_entities_test_2"):
        arquivo = accr_dir / f"accr_predictions_{gt}.json"
        if not arquivo.exists():
            continue
        try:
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Erro ao ler {arquivo.name}: {e}")
            continue

        por_video = dados.get("resultados_por_video", {})

        # Pré-computa captions por (video_id, modelo) para R@4
        captions_video: dict = defaultdict(list)
        for video_id, segmentos in por_video.items():
            for seg in segmentos:
                for modelo, av in seg.get("avaliacoes", {}).items():
                    cap = av.get("caption", "") or ""
                    captions_video[(video_id, modelo)].append(cap)

        # Constrói um registro por segmento × modelo
        for video_id, segmentos in por_video.items():
            for seg in segmentos:
                refs = seg.get("ground_truth", [])
                for modelo, av in seg.get("avaliacoes", {}).items():
                    cap = av.get("caption", "") or ""
                    scores = av.get("scores", {})
                    registros.append({
                        "video_id":    video_id,
                        "segment":     seg.get("segment", 0),
                        "gt":          gt,
                        "modelo":      _canon(modelo),
                        "caption":     cap,
                        "ground_truth": refs,
                        "accuracy":     scores.get("accuracy",     0),
                        "completeness": scores.get("completeness", 0),
                        "conciseness":  scores.get("conciseness",  0),
                        "relevance":    scores.get("relevance",    0),
                        "rouge_l":  _rouge_l(refs, cap),
                        "meteor":   _meteor(refs, cap),
                        "r4_video": _r4_video(captions_video[(video_id, modelo)]),
                    })
    return registros


# ─────────────────────────────────────────────────────────────
# Utilitário: modelos únicos e médias
# ─────────────────────────────────────────────────────────────

def _modelos_unicos(dados: dict) -> list:
    """Extrai lista de nomes de modelos únicos (sem sufixo GT) do dict."""
    vistos = []
    for chave in dados:
        modelo = chave.rsplit(" (", 1)[0]
        if modelo not in vistos:
            vistos.append(modelo)
    return vistos


def _media_modelo(dados: dict, modelo: str, key: str) -> float:
    """Média de uma métrica/dimensão para um modelo entre os dois GTs."""
    entradas = [v for k, v in dados.items() if k.startswith(f"{modelo} (")]
    vals = [e.get(key, 0.0) for e in entradas if key in e]
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────
# Módulo 2: Gráficos de métricas automáticas
# ─────────────────────────────────────────────────────────────

def grafico_radar(auto: dict, saida: Path) -> None:
    """
    Radar chart com 4 eixos de métricas automáticas normalizadas.
    Um polígono por modelo, médias entre GT1 e GT2.
    """
    metricas  = ["CIDEr", "Bleu_4", "ROUGE_L", "METEOR"]
    rotulos   = ["CIDEr-D", "BLEU-4", "ROUGE-L", "METEOR"]
    modelos   = _modelos_unicos(auto)
    if not modelos:
        return

    # Médias por modelo
    medias = {
        mo: {m: _media_modelo(auto, mo, m) for m in metricas}
        for mo in modelos
    }

    # Normaliza cada eixo pelo valor máximo entre os modelos
    max_val = {
        m: max(medias[mo][m] for mo in modelos) or 1.0
        for m in metricas
    }
    norm = {
        mo: [medias[mo][m] / max_val[m] for m in metricas]
        for mo in modelos
    }

    N = len(metricas)
    angulos = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angulos[:-1]), rotulos, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)

    for idx, (modelo, vals) in enumerate(norm.items()):
        cor = _CORES[idx % len(_CORES)]
        vals_plot = vals + [vals[0]]
        ax.plot(angulos, vals_plot, "o-", linewidth=2, color=cor, label=modelo)
        ax.fill(angulos, vals_plot, alpha=0.12, color=cor)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title(
        "Métricas Automáticas — Radar (normalizado)",
        pad=18, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_barras_gt(auto: dict, saida: Path) -> None:
    """
    Grouped bar chart: para cada métrica automática, mostra valores
    de GT1 e GT2 lado a lado por modelo.
    """
    metricas_cfg = [
        ("CIDEr",   "CIDEr-D"),
        ("Bleu_4",  "BLEU-4"),
        ("ROUGE_L", "ROUGE-L"),
        ("METEOR",  "METEOR"),
        ("R@4",     "R@4 ↓"),
    ]
    gts     = ["anet_entities_test_1", "anet_entities_test_2"]
    rot_gts = ["GT1", "GT2"]
    modelos = _modelos_unicos(auto)
    if not modelos:
        return

    n_met = len(metricas_cfg)
    fig, axes = plt.subplots(1, n_met, figsize=(3.5 * n_met, 5), sharey=False)

    for ax_idx, (key, label) in enumerate(metricas_cfg):
        ax = axes[ax_idx]
        x = np.arange(len(modelos))
        width = 0.35

        for gt_idx, (gt, rot_gt) in enumerate(zip(gts, rot_gts)):
            vals = [auto.get(f"{mo} ({gt})", {}).get(key, 0.0) for mo in modelos]
            offset = (gt_idx - 0.5) * width
            ax.bar(
                x + offset, vals, width,
                label=rot_gt,
                color=_CORES_GT[gt_idx],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "Métricas Automáticas por Modelo e Ground Truth",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


# ─────────────────────────────────────────────────────────────
# Módulo 3: Gráficos ACCR
# ─────────────────────────────────────────────────────────────

def grafico_heatmap_accr(accr: dict, saida: Path) -> None:
    """
    Heatmap de médias ACCR por modelo (linhas) e dimensão (colunas).
    Escala 0–100. Cores: vermelho → amarelo → verde.
    """
    dims    = ["accuracy", "completeness", "conciseness", "relevance", "media_geral"]
    rotulos = ["Accuracy", "Completeness", "Conciseness", "Relevance", "Média Geral"]
    modelos = _modelos_unicos(accr)
    if not modelos:
        return

    matriz = np.array([
        [_media_modelo(accr, mo, dim) for dim in dims]
        for mo in modelos
    ])

    fig, ax = plt.subplots(figsize=(9, max(3, len(modelos) * 1.4)))
    im = ax.imshow(matriz, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(rotulos, fontsize=11)
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos, fontsize=11)

    # Anota cada célula com o valor numérico
    for i in range(len(modelos)):
        for j in range(len(dims)):
            v = matriz[i, j]
            cor_txt = "black" if 25 < v < 78 else "white"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=10, color=cor_txt, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Score ACCR (0–100)")
    ax.set_title("ACCR — Médias por Modelo e Dimensão", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_barras_accr_gt(accr: dict, saida: Path) -> None:
    """
    Grouped bar chart do ACCR: para cada dimensão, mostra GT1 e GT2
    lado a lado por modelo.
    """
    dims_cfg = [
        ("accuracy",     "Accuracy"),
        ("completeness", "Completeness"),
        ("conciseness",  "Conciseness"),
        ("relevance",    "Relevance"),
        ("media_geral",  "Média Geral"),
    ]
    gts     = ["anet_entities_test_1", "anet_entities_test_2"]
    rot_gts = ["GT1", "GT2"]
    modelos = _modelos_unicos(accr)
    if not modelos:
        return

    n_dim = len(dims_cfg)
    fig, axes = plt.subplots(1, n_dim, figsize=(3.5 * n_dim, 5), sharey=False)

    for ax_idx, (key, label) in enumerate(dims_cfg):
        ax = axes[ax_idx]
        x = np.arange(len(modelos))
        width = 0.35

        for gt_idx, (gt, rot_gt) in enumerate(zip(gts, rot_gts)):
            vals = [accr.get(f"{mo} ({gt})", {}).get(key, 0.0) for mo in modelos]
            offset = (gt_idx - 0.5) * width
            ax.bar(
                x + offset, vals, width,
                label=rot_gt,
                color=_CORES_GT[gt_idx],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_ylim(0, 105)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score ACCR (0–100)", fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "ACCR por Modelo e Ground Truth",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_kde_accr(segmentos: list, saida: Path) -> None:
    """
    Curvas KDE (ou histograma fallback) da distribuição de scores ACCR
    por modelo, uma subplot por dimensão.
    """
    dims    = ["accuracy", "completeness", "conciseness", "relevance"]
    rotulos = ["Accuracy", "Completeness", "Conciseness", "Relevance"]
    modelos = sorted({s["modelo"] for s in segmentos})
    if not modelos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x_plot = np.linspace(0, 100, 300)

    for ax_idx, (dim, label) in enumerate(zip(dims, rotulos)):
        ax = axes[ax_idx]
        for idx, modelo in enumerate(modelos):
            vals = [s[dim] for s in segmentos if s["modelo"] == modelo and s[dim] > 0]
            if len(vals) < 3:
                continue
            cor = _CORES[idx % len(_CORES)]
            if _SCIPY_OK:
                kde = gaussian_kde(vals, bw_method=0.3)
                ax.plot(x_plot, kde(x_plot), linewidth=2.5, color=cor, label=modelo)
                ax.fill_between(x_plot, kde(x_plot), alpha=0.15, color=cor)
            else:
                ax.hist(vals, bins=15, density=True, alpha=0.5, color=cor, label=modelo,
                        edgecolor="white")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score ACCR (0–100)", fontsize=9)
        ax.set_ylabel("Densidade", fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)

    fig.suptitle("Distribuição dos Scores ACCR por Modelo",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


# ─────────────────────────────────────────────────────────────
# Módulo 4: Correlação ACCR × Métricas Automáticas
# ─────────────────────────────────────────────────────────────

def grafico_correlacao(segmentos: list, saida: Path) -> None:
    """
    Heatmap de correlação de Spearman entre dimensões ACCR e métricas
    automáticas calculadas por segmento (ROUGE-L, METEOR, R@4).
    Asterisco indica p < 0.05.
    """
    dims_accr = ["accuracy", "completeness", "conciseness", "relevance"]
    rot_accr  = ["Accuracy", "Completeness", "Conciseness", "Relevance"]

    metricas_auto = [("rouge_l", "ROUGE-L"), ("r4_video", "R@4")]
    if _NLTK_OK:
        metricas_auto.insert(1, ("meteor", "METEOR"))

    if not segmentos:
        return

    # Filtra segmentos com pelo menos ROUGE-L calculado
    validos = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    if len(validos) < 5:
        print("⚠️  Segmentos insuficientes para correlação")
        return

    n_accr = len(dims_accr)
    n_auto = len(metricas_auto)
    matriz  = np.zeros((n_accr, n_auto))
    p_mat   = np.ones((n_accr, n_auto))

    for i, dim in enumerate(dims_accr):
        for j, (met_key, _) in enumerate(metricas_auto):
            pares = [
                (s[dim], s[met_key])
                for s in validos
                if s.get(met_key, -1) >= 0
            ]
            if len(pares) < 5:
                continue
            x_arr, y_arr = zip(*pares)
            if _SCIPY_OK:
                r, p = spearmanr(x_arr, y_arr)
                matriz[i, j] = r
                p_mat[i, j]  = p
            else:
                # Fallback: correlação de Pearson
                c = np.corrcoef(x_arr, y_arr)
                matriz[i, j] = c[0, 1]

    fig, ax = plt.subplots(figsize=(max(5, n_auto * 2.2), 5))
    im = ax.imshow(matriz, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(n_auto))
    ax.set_xticklabels([l for _, l in metricas_auto], fontsize=11)
    ax.set_yticks(range(n_accr))
    ax.set_yticklabels(rot_accr, fontsize=11)

    for i in range(n_accr):
        for j in range(n_auto):
            v   = matriz[i, j]
            sig = "*" if p_mat[i, j] < 0.05 else ""
            cor_txt = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                    fontsize=11, color=cor_txt)

    plt.colorbar(im, ax=ax, label="Spearman r")
    titulo = ("Correlação Spearman: ACCR × Métricas Automáticas (por segmento)"
              if _SCIPY_OK else
              "Correlação Pearson: ACCR × Métricas Automáticas (por segmento)")
    ax.set_title(titulo + "\n(* p < 0.05)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_scatter_accr_auto(segmentos: list, saida: Path) -> None:
    """
    Grid de scatter plots: dimensões ACCR (eixo Y) vs métricas automáticas
    (eixo X), com linha de tendência por modelo.
    """
    if not segmentos:
        return

    metricas_auto = [("rouge_l", "ROUGE-L")]
    if _NLTK_OK:
        metricas_auto.append(("meteor", "METEOR"))

    dims_accr = [
        ("accuracy",     "Accuracy"),
        ("completeness", "Completeness"),
        ("conciseness",  "Conciseness"),
        ("relevance",    "Relevance"),
    ]

    modelos = sorted({s["modelo"] for s in segmentos})
    n_rows = len(dims_accr)
    n_cols = len(metricas_auto)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for row, (dim, dim_lbl) in enumerate(dims_accr):
        for col, (met_key, met_lbl) in enumerate(metricas_auto):
            ax = axes[row][col]
            for idx, modelo in enumerate(modelos):
                segs = [
                    s for s in segmentos
                    if s["modelo"] == modelo and s.get(met_key, -1) >= 0
                ]
                if not segs:
                    continue
                x = [s[met_key] for s in segs]
                y = [s[dim]     for s in segs]
                cor = _CORES[idx % len(_CORES)]
                ax.scatter(x, y, alpha=0.45, s=25, color=cor, label=modelo)
                # Linha de tendência linear
                if len(x) >= 3:
                    try:
                        coef = np.polyfit(x, y, 1)
                        x_ln = np.linspace(min(x), max(x), 100)
                        ax.plot(x_ln, np.polyval(coef, x_ln), "--",
                                color=cor, alpha=0.7, linewidth=1.5)
                    except Exception:
                        pass
            ax.set_xlabel(met_lbl, fontsize=9)
            ax.set_ylabel(f"ACCR {dim_lbl}", fontsize=9)
            if row == 0:
                ax.set_title(met_lbl, fontsize=11, fontweight="bold")
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.suptitle(
        "Scatter: ACCR vs Métricas Automáticas (por segmento)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_r4_conciseness(segmentos: list, saida: Path) -> None:
    """
    Scatter de R@4 (repetição de 4-gramas) vs ACCR Conciseness por segmento.
    Valida se R@4 é proxy válido de concisão avaliada por LLM.
    """
    modelos = sorted({s["modelo"] for s in segmentos})
    if not modelos:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, modelo in enumerate(modelos):
        segs = [s for s in segmentos if s["modelo"] == modelo]
        if not segs:
            continue
        x = [s["r4_video"]   for s in segs]
        y = [s["conciseness"] for s in segs]
        cor = _CORES[idx % len(_CORES)]
        ax.scatter(x, y, alpha=0.5, s=45, color=cor, label=modelo, edgecolors="none")
        if len(x) >= 3:
            try:
                coef = np.polyfit(x, y, 1)
                x_ln = np.linspace(min(x), max(x), 100)
                ax.plot(x_ln, np.polyval(coef, x_ln), "--",
                        color=cor, linewidth=2.0, alpha=0.7)
            except Exception:
                pass

    ax.set_xlabel("R@4 (repetição de 4-gramas, ↓ melhor)", fontsize=11)
    ax.set_ylabel("ACCR Conciseness", fontsize=11)
    ax.set_title(
        "R@4 vs ACCR Conciseness\n(valida R@4 como proxy de concisão)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


# ─────────────────────────────────────────────────────────────
# Módulo 5: Rankings
# ─────────────────────────────────────────────────────────────

def grafico_rank_stability(auto: dict, accr: dict, saida: Path) -> None:
    """
    Bump chart mostrando estabilidade de ranking entre GT1 e GT2.
    Uma faixa por métrica de referência (CIDEr e ACCR Geral).
    """
    todas = {**auto, **accr}
    modelos = _modelos_unicos(todas)
    if not modelos:
        return

    metricas_rank = []
    if auto:
        metricas_rank.append(("Auto: CIDEr-D", auto, "CIDEr", False))
    if accr:
        metricas_rank.append(("ACCR: Média Geral", accr, "media_geral", False))
    if not metricas_rank:
        return

    gts     = ["anet_entities_test_1", "anet_entities_test_2"]
    rot_gts = ["GT1", "GT2"]
    n_met   = len(metricas_rank)

    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met, max(4, len(modelos) * 1.6)))
    if n_met == 1:
        axes = [axes]

    for ax_idx, (titulo, fonte, key, inv) in enumerate(metricas_rank):
        ax = axes[ax_idx]

        # Calcula ranking por GT (1 = melhor)
        rankings = []
        for gt in gts:
            scores_gt = [
                (mo, fonte.get(f"{mo} ({gt})", {}).get(key, 0.0))
                for mo in modelos
            ]
            scores_gt.sort(key=lambda t: t[1], reverse=not inv)
            rankings.append({mo: pos + 1 for pos, (mo, _) in enumerate(scores_gt)})

        for idx, modelo in enumerate(modelos):
            cor = _CORES[idx % len(_CORES)]
            y_vals = [rankings[g].get(modelo, len(modelos)) for g in range(len(gts))]
            ax.plot(range(len(gts)), y_vals, "o-", color=cor, linewidth=2.5,
                    markersize=11, zorder=3)
            # Rótulo do modelo na direita
            ax.text(len(gts) - 1 + 0.06, y_vals[-1],
                    f"  {modelo}", fontsize=9, va="center", color=cor)

        ax.set_xticks(range(len(gts)))
        ax.set_xticklabels(rot_gts, fontsize=12)
        ax.set_yticks(range(1, len(modelos) + 1))
        ax.set_yticklabels([f"{i}º" for i in range(1, len(modelos) + 1)], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(-0.3, len(gts) - 0.5)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Estabilidade de Ranking entre Ground Truths",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


# ─────────────────────────────────────────────────────────────
# Módulo 6: Análise Avançada
# ─────────────────────────────────────────────────────────────

def _norm_accr(v: float) -> float:
    """Normaliza ACCR (0–100) → (0–1)."""
    return v / 100.0


def _melhor_auto_seg(seg: dict) -> float:
    """
    Melhor métrica automática disponível no segmento, normalizada 0–1.
    Preferência: METEOR > ROUGE-L.
    """
    if _NLTK_OK and seg.get("meteor", -1) >= 0:
        return seg["meteor"]
    return max(seg.get("rouge_l", 0.0), 0.0)


def grafico_correlacao_sig(segmentos: list, saida: Path) -> None:
    """
    Correlation Matrix Heatmap com significância em dois níveis:
      ** p < 0.01   * p < 0.05   (sem marca) = não significativo
    Células não significativas exibem o valor em cinza para indicar ruído.
    """
    dims_accr = ["accuracy", "completeness", "conciseness", "relevance"]
    rot_accr  = ["Accuracy", "Completeness", "Conciseness", "Relevance"]

    metricas_auto = [("rouge_l", "ROUGE-L"), ("r4_video", "R@4")]
    if _NLTK_OK:
        metricas_auto.insert(1, ("meteor", "METEOR"))

    validos = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    if len(validos) < 10:
        return

    n_accr = len(dims_accr)
    n_auto = len(metricas_auto)
    matriz = np.zeros((n_accr, n_auto))
    p_mat  = np.ones((n_accr, n_auto))

    for i, dim in enumerate(dims_accr):
        for j, (met_key, _) in enumerate(metricas_auto):
            pares = [
                (s[dim], s[met_key])
                for s in validos
                if s.get(met_key, -1) >= 0
            ]
            if len(pares) < 5:
                continue
            x_arr, y_arr = zip(*pares)
            if _SCIPY_OK:
                r, p = spearmanr(x_arr, y_arr)
                matriz[i, j] = r
                p_mat[i, j]  = p
            else:
                matriz[i, j] = np.corrcoef(x_arr, y_arr)[0, 1]

    fig, ax = plt.subplots(figsize=(max(5, n_auto * 2.5), 5))
    im = ax.imshow(matriz, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(n_auto))
    ax.set_xticklabels([lbl for _, lbl in metricas_auto], fontsize=12)
    ax.set_yticks(range(n_accr))
    ax.set_yticklabels(rot_accr, fontsize=12)

    for i in range(n_accr):
        for j in range(n_auto):
            v = matriz[i, j]
            p = p_mat[i, j]
            if p < 0.01:
                marca = "**"
            elif p < 0.05:
                marca = "*"
            else:
                marca = ""
            is_sig  = p < 0.05
            cor_txt = ("white" if abs(v) > 0.55 else "black") if is_sig else "#aaaaaa"
            ax.text(j, i, f"{v:.2f}{marca}", ha="center", va="center",
                    fontsize=12, color=cor_txt,
                    fontweight="bold" if is_sig else "normal")

    plt.colorbar(im, ax=ax, label="Spearman r" if _SCIPY_OK else "Pearson r")
    tipo = "Spearman" if _SCIPY_OK else "Pearson"
    ax.set_title(
        f"Correlação {tipo}: ACCR × Métricas Automáticas\n"
        "** p<0.01   * p<0.05   cinza = não significativo",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_bland_altman(segmentos: list, saida: Path) -> None:
    """
    Bland-Altman Plot: compara ACCR Accuracy normalizado com a melhor
    métrica automática disponível por segmento.
    Eixo X = média dos dois; Eixo Y = diferença (ACCR − auto).
    Linhas tracejadas em ±1.96 DP (limites de concordância).
    Um subplot por modelo.
    """
    validos   = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    modelos   = sorted({s["modelo"] for s in validos})
    if not modelos or not validos:
        return

    met_label = "METEOR" if _NLTK_OK else "ROUGE-L"

    fig, axes = plt.subplots(
        1, len(modelos),
        figsize=(6 * len(modelos), 5),
        sharey=True,
    )
    if len(modelos) == 1:
        axes = [axes]

    for idx, (modelo, ax) in enumerate(zip(modelos, axes)):
        segs_m = [s for s in validos if s["modelo"] == modelo]
        if not segs_m:
            ax.set_title(modelo)
            continue

        accr_n = np.array([_norm_accr(s["accuracy"]) for s in segs_m])
        auto_n = np.array([_melhor_auto_seg(s)        for s in segs_m])

        media  = (accr_n + auto_n) / 2.0
        dif    = accr_n - auto_n
        mean_d = float(np.mean(dif))
        sd_d   = float(np.std(dif, ddof=1))
        loa_up = mean_d + 1.96 * sd_d
        loa_lo = mean_d - 1.96 * sd_d

        cor = _CORES[idx % len(_CORES)]
        ax.scatter(media, dif, alpha=0.55, s=40, color=cor, edgecolors="none")
        ax.axhline(mean_d, color="black",   linestyle="-",  linewidth=1.8,
                   label=f"Bias: {mean_d:+.3f}")
        ax.axhline(loa_up, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"+1.96 SD: {loa_up:+.3f}")
        ax.axhline(loa_lo, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"−1.96 SD: {loa_lo:+.3f}")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_title(modelo, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Média (ACCR norm, {met_label})", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Diferença (ACCR − Auto)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Bland-Altman: ACCR Accuracy vs {met_label} (por modelo)\n"
        "Positivo = ACCR mais generoso que a métrica automática",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_parallel_coordinates(segmentos: list, saida: Path) -> None:
    """
    Parallel Coordinates Plot: cada segmento é uma polilinha atravessando
    eixos normalizados (ROUGE-L, METEOR*, R@4, Accuracy, Completeness,
    Conciseness, Relevance).  Linhas grossas = média por modelo.
    (* omitido se nltk indisponível)
    """
    eixos_cfg = [("rouge_l", "ROUGE-L"), ("r4_video", "R@4")]
    if _NLTK_OK:
        eixos_cfg.insert(1, ("meteor", "METEOR"))
    eixos_cfg += [
        ("accuracy",     "Accuracy"),
        ("completeness", "Completeness"),
        ("conciseness",  "Conciseness"),
        ("relevance",    "Relevance"),
    ]

    validos = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    if not validos:
        return

    modelos = sorted({s["modelo"] for s in validos})
    n_eixos = len(eixos_cfg)

    # Min-max por eixo para normalização
    vals_por_eixo: dict = {}
    for key, _ in eixos_cfg:
        col = [s[key] for s in validos if s.get(key, -1) >= 0]
        mn, mx = (min(col), max(col)) if col else (0.0, 1.0)
        vals_por_eixo[key] = (mn, mx if mx > mn else mn + 1e-9)

    def _norm(v: float, key: str) -> float:
        mn, mx = vals_por_eixo[key]
        return (v - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(2.5 * n_eixos, 6))

    # Subsample para não sobrecarregar a figura
    _rng    = _random.Random(42)
    amostra = validos if len(validos) <= 150 else _rng.sample(validos, 150)

    for seg in amostra:
        idx_m  = modelos.index(seg["modelo"]) if seg["modelo"] in modelos else 0
        cor    = _CORES[idx_m % len(_CORES)]
        x_vals, y_vals = [], []
        for ei, (key, _) in enumerate(eixos_cfg):
            raw = seg.get(key, -1)
            if raw >= 0:
                x_vals.append(ei)
                y_vals.append(_norm(raw, key))
        if len(x_vals) >= 2:
            ax.plot(x_vals, y_vals, color=cor, alpha=0.18, linewidth=0.8)

    # Médias por modelo (linhas grossas)
    for idx_m, modelo in enumerate(modelos):
        cor    = _CORES[idx_m % len(_CORES)]
        segs_m = [s for s in validos if s["modelo"] == modelo]
        x_med, y_med = [], []
        for ei, (key, _) in enumerate(eixos_cfg):
            col = [s[key] for s in segs_m if s.get(key, -1) >= 0]
            if col:
                x_med.append(ei)
                y_med.append(_norm(float(np.mean(col)), key))
        if len(x_med) >= 2:
            ax.plot(x_med, y_med, "o-", color=cor, linewidth=3.0,
                    markersize=7, zorder=5, label=modelo, alpha=0.95)

    ax.set_xticks(range(n_eixos))
    ax.set_xticklabels([lbl for _, lbl in eixos_cfg], fontsize=10,
                       rotation=20, ha="right")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["min", "25%", "50%", "75%", "max"], fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", linewidth=1.2, alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(
        "Parallel Coordinates — Perfil Completo por Segmento\n"
        "(linhas grossas = média do modelo; linhas finas = segmentos individuais)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_calibration(segmentos: list, saida: Path) -> None:
    """
    Calibration Curves: segmentos divididos em 5 faixas (quintis) pelo
    score automático; exibe ACCR médio em cada faixa.
    Diagonal perfeita = calibração ideal.
    """
    metricas_auto = [("rouge_l", "ROUGE-L")]
    if _NLTK_OK:
        metricas_auto.append(("meteor", "METEOR"))

    dims_cfg = [
        ("accuracy",     "Accuracy"),
        ("completeness", "Completeness"),
        ("conciseness",  "Conciseness"),
        ("relevance",    "Relevance"),
    ]

    validos = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    if not validos:
        return
    modelos = sorted({s["modelo"] for s in validos})
    n_bins  = 5
    n_met   = len(metricas_auto)
    n_dims  = len(dims_cfg)

    fig, axes = plt.subplots(
        n_dims, n_met,
        figsize=(5 * n_met, 4 * n_dims),
        squeeze=False,
    )

    for row, (dim_key, dim_lbl) in enumerate(dims_cfg):
        for col, (met_key, met_lbl) in enumerate(metricas_auto):
            ax = axes[row][col]
            for idx_m, modelo in enumerate(modelos):
                segs_m = [
                    s for s in validos
                    if s["modelo"] == modelo and s.get(met_key, -1) >= 0
                ]
                if len(segs_m) < n_bins:
                    continue
                cor = _CORES[idx_m % len(_CORES)]

                segs_ord = sorted(segs_m, key=lambda s: s[met_key])
                chunk    = len(segs_ord) // n_bins
                bin_x, bin_y = [], []
                for b in range(n_bins):
                    ini   = b * chunk
                    fim   = ini + chunk if b < n_bins - 1 else len(segs_ord)
                    bloco = segs_ord[ini:fim]
                    bin_x.append(float(np.mean([s[met_key]  for s in bloco])))
                    bin_y.append(float(np.mean([s[dim_key]  for s in bloco])))
                ax.plot(bin_x, bin_y, "o-", color=cor, linewidth=2,
                        markersize=7, label=modelo)

            # Linha de calibração ideal (escalonada para o range da métrica)
            met_max = max(
                (s[met_key] for s in validos if s.get(met_key, -1) >= 0),
                default=1.0,
            )
            ax.plot([0, met_max], [0, 100], ":", color="gray",
                    linewidth=1.5, alpha=0.7, label="Ideal")
            ax.set_xlabel(met_lbl, fontsize=9)
            ax.set_ylabel(f"ACCR {dim_lbl} médio", fontsize=9)
            if row == 0:
                ax.set_title(met_lbl, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 105)
            if row == 0 and col == n_met - 1:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Calibration Curves — ACCR médio por quintil de métrica automática\n"
        "Curva diagonal = calibração ideal",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_delta_heatmap(segmentos: list, saida: Path) -> None:
    """
    Delta Heatmap: (ACCR_norm − Auto_norm) por vídeo × modelo.
    Verde = ACCR > auto (humano mais generoso).
    Vermelho = auto > ACCR (métrica automática superestima).
    Delta = média dos 4 eixos ACCR normalizado − melhor auto disponível.
    """
    validos   = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    modelos   = sorted({s["modelo"] for s in validos})
    video_ids = sorted({s["video_id"] for s in validos})
    if not video_ids or not modelos:
        return

    dims_accr = ["accuracy", "completeness", "conciseness", "relevance"]

    # Agrega deltas por (video_id, modelo)
    delta_map: dict = defaultdict(list)
    for seg in validos:
        accr_n = float(np.mean([_norm_accr(seg[d]) for d in dims_accr]))
        auto_n = _melhor_auto_seg(seg)
        delta_map[(seg["video_id"], seg["modelo"])].append(accr_n - auto_n)

    def _vid_lbl(vid: str) -> str:
        return vid[:10] + "…" if len(vid) > 11 else vid

    matriz = np.full((len(video_ids), len(modelos)), np.nan)
    for i, vid in enumerate(video_ids):
        for j, mod in enumerate(modelos):
            vals = delta_map[(vid, mod)]
            if vals:
                matriz[i, j] = float(np.mean(vals))

    vmax = max(0.01, float(np.nanmax(np.abs(matriz))))

    fig, ax = plt.subplots(
        figsize=(max(5, len(modelos) * 2.5), max(5, len(video_ids) * 0.65))
    )
    im = ax.imshow(matriz, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels(modelos, fontsize=10)
    ax.set_yticks(range(len(video_ids)))
    ax.set_yticklabels([_vid_lbl(v) for v in video_ids], fontsize=8)

    for i in range(len(video_ids)):
        for j in range(len(modelos)):
            v = matriz[i, j]
            if not np.isnan(v):
                cor_txt = "black" if abs(v) < vmax * 0.6 else "white"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color=cor_txt)

    plt.colorbar(im, ax=ax, label="Δ = ACCR_norm − Auto_norm")
    ax.set_title(
        "Delta Heatmap: ACCR vs Métrica Automática por Vídeo\n"
        "Verde = humano mais generoso  |  Vermelho = auto superestima",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_ranking_agreement(segmentos: list, saida: Path) -> None:
    """
    Model Ranking Agreement: % de vídeos em que cada métrica automática
    concorda com o ACCR sobre qual modelo performa melhor.
    Barras separadas por par de modelos; linha pontilhada = aleatório (50%).
    """
    validos   = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    modelos   = sorted({s["modelo"] for s in validos})
    video_ids = sorted({s["video_id"] for s in validos})
    if len(modelos) < 2 or not video_ids:
        return

    metricas_cfg = [("rouge_l", "ROUGE-L"), ("r4_video", "R@4 ↓")]
    if _NLTK_OK:
        metricas_cfg.insert(1, ("meteor", "METEOR"))

    dims_accr = ["accuracy", "completeness", "conciseness", "relevance"]
    pares     = list(itertools.combinations(modelos, 2))

    def _score_vid(vid: str, mod: str, key: str) -> float:
        segs = [s for s in validos
                if s["video_id"] == vid and s["modelo"] == mod
                and s.get(key, -1) >= 0]
        return float(np.mean([s[key] for s in segs])) if segs else 0.0

    def _accr_vid(vid: str, mod: str) -> float:
        segs = [s for s in validos if s["video_id"] == vid and s["modelo"] == mod]
        return float(np.mean([np.mean([s[d] for d in dims_accr]) for s in segs])) if segs else 0.0

    n_met  = len(metricas_cfg)
    n_par  = len(pares)
    acordos = np.zeros((n_met, n_par))

    for j, (ma, mb) in enumerate(pares):
        accr_a   = [_accr_vid(v, ma) for v in video_ids]
        accr_b   = [_accr_vid(v, mb) for v in video_ids]
        rank_ref = [1 if a > b else (-1 if b > a else 0)
                    for a, b in zip(accr_a, accr_b)]
        non_ties = [i for i, r in enumerate(rank_ref) if r != 0]
        if not non_ties:
            continue

        for i, (met_key, _) in enumerate(metricas_cfg):
            inv    = met_key == "r4_video"  # R@4: menor é melhor
            auto_a = [_score_vid(v, ma, met_key) for v in video_ids]
            auto_b = [_score_vid(v, mb, met_key) for v in video_ids]
            rank_auto = [
                1 if (a < b if inv else a > b) else (-1 if (b < a if inv else b > a) else 0)
                for a, b in zip(auto_a, auto_b)
            ]
            concorda = sum(1 for idx in non_ties if rank_auto[idx] == rank_ref[idx])
            acordos[i, j] = 100.0 * concorda / len(non_ties)

    x           = np.arange(n_par)
    width       = 0.8 / max(n_met, 1)
    par_labels  = [f"{a}\nvs\n{b}" for a, b in pares]

    fig, ax = plt.subplots(figsize=(max(6, n_par * 3.5), 5))
    for i, (_, met_lbl) in enumerate(metricas_cfg):
        offset = (i - (n_met - 1) / 2) * width
        ax.bar(x + offset, acordos[i], width,
               label=met_lbl, color=_CORES[i % len(_CORES)],
               alpha=0.82, edgecolor="white")

    ax.axhline(50, color="black", linestyle="--", linewidth=1.3,
               alpha=0.6, label="Aleatório (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(par_labels, fontsize=10)
    ax.set_ylabel("% vídeos com ranking correto", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.set_title(
        "Ranking Agreement: métrica automática vs ACCR\n"
        "% vídeos onde a métrica acerta o modelo melhor  |  pontilhada = aleatório",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


def grafico_hexbin(segmentos: list, saida: Path) -> None:
    """
    Hexbin Density Plot: ACCR Accuracy vs melhor métrica automática.
    Cada hexágono colorido pela concentração de segmentos.
    Um subplot por modelo.
    """
    validos   = [s for s in segmentos if s.get("rouge_l", -1) >= 0]
    modelos   = sorted({s["modelo"] for s in validos})
    if not modelos:
        return

    met_label = "METEOR" if _NLTK_OK else "ROUGE-L"
    all_auto  = [_melhor_auto_seg(s) for s in validos]
    xlim      = (0, max(all_auto) * 1.05) if all_auto else (0, 1)

    fig, axes = plt.subplots(
        1, len(modelos),
        figsize=(5 * len(modelos), 5),
        sharey=True, sharex=True,
    )
    if len(modelos) == 1:
        axes = [axes]

    for idx, (modelo, ax) in enumerate(zip(modelos, axes)):
        segs_m = [s for s in validos if s["modelo"] == modelo]
        if not segs_m:
            ax.set_title(modelo)
            continue
        x  = [_melhor_auto_seg(s)  for s in segs_m]
        y  = [s["accuracy"]        for s in segs_m]
        hb = ax.hexbin(x, y, gridsize=10, cmap="YlOrRd",
                       mincnt=1, linewidths=0.3)
        plt.colorbar(hb, ax=ax, label="Nº segmentos")
        ax.set_title(modelo, fontsize=11, fontweight="bold")
        ax.set_xlabel(met_label, fontsize=10)
        if idx == 0:
            ax.set_ylabel("ACCR Accuracy", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 105)

    fig.suptitle(
        f"Densidade Hexbin: ACCR Accuracy vs {met_label}\n"
        "Amarelo/vermelho = maior concentração de segmentos",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {saida.name}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos a partir dos outputs da pipeline."
    )
    parser.add_argument(
        "--auto-dir", type=Path, default=_AUTO,
        help=f"Diretório de métricas automáticas (padrão: {_AUTO})",
    )
    parser.add_argument(
        "--accr-dir", type=Path, default=_ACCR,
        help=f"Diretório de métricas ACCR (padrão: {_ACCR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_PLOTS,
        help=f"Diretório de saída dos gráficos (padrão: {_PLOTS})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Carregando dados...")
    auto      = carregar_auto(args.auto_dir)
    accr      = carregar_accr(args.accr_dir)
    segmentos = carregar_segmentos(args.accr_dir)

    if not auto and not accr:
        print("❌ Nenhum dado encontrado. Verifique os diretórios de entrada.")
        sys.exit(1)

    print(f"  Métricas automáticas : {len(auto)} entradas")
    print(f"  ACCR                 : {len(accr)} entradas")
    print(f"  Segmentos (ACCR)     : {len(segmentos)}")

    print("\n📈 Gerando gráficos...\n")

    # ── Métricas automáticas ──────────────────────────────────
    if auto:
        grafico_radar(auto,       args.output_dir / "radar_auto.png")
        grafico_barras_gt(auto,   args.output_dir / "barras_auto_gt.png")

    # ── ACCR ─────────────────────────────────────────────────
    if accr:
        grafico_heatmap_accr(accr,       args.output_dir / "heatmap_accr.png")
        grafico_barras_accr_gt(accr,     args.output_dir / "barras_accr_gt.png")

    if segmentos:
        grafico_kde_accr(segmentos,          args.output_dir / "kde_accr.png")
        grafico_correlacao(segmentos,        args.output_dir / "correlacao_accr_auto.png")
        grafico_scatter_accr_auto(segmentos, args.output_dir / "scatter_accr_auto.png")
        grafico_r4_conciseness(segmentos,    args.output_dir / "scatter_r4_conciseness.png")

    # ── Rankings ─────────────────────────────────────────────
    if auto or accr:
        grafico_rank_stability(auto, accr, args.output_dir / "rank_stability.png")

    # ── Análise Avançada (Módulo 6) ──────────────────────────
    if segmentos:
        grafico_correlacao_sig(segmentos,        args.output_dir / "correlacao_sig.png")
        grafico_bland_altman(segmentos,          args.output_dir / "bland_altman.png")
        grafico_parallel_coordinates(segmentos,  args.output_dir / "parallel_coordinates.png")
        grafico_calibration(segmentos,           args.output_dir / "calibration.png")
        grafico_delta_heatmap(segmentos,         args.output_dir / "delta_heatmap.png")
        grafico_ranking_agreement(segmentos,     args.output_dir / "ranking_agreement.png")
        grafico_hexbin(segmentos,                args.output_dir / "hexbin.png")

    print(f"\n✅ Gráficos salvos em: {args.output_dir}/")


if __name__ == "__main__":
    main()
