#!/usr/bin/env python3
"""
statistical_analysis.py

Análise estatística final do TCC.

Entradas padrão:
    output/metrics/human/human_scores_by_video_model.csv
    output/metrics/human/human_scores_by_dimension.csv
    output/metrics/human/human_vs_llm_scores.csv
    output/metrics/auto/metricas_por_video_multiref.csv

Saídas:
    output/metrics/statistical/
        kendall_human_vs_llm.csv
        kendall_human_vs_auto_metrics.csv
        kendall_llm_vs_auto_metrics.csv
        friedman_human.csv
        wilcoxon_human_holm.csv
        friedman_llm.csv
        wilcoxon_llm_holm.csv
        friedman_auto_metrics.csv
        wilcoxon_auto_metrics_holm.csv
        merged_human_auto.csv
        merged_llm_auto.csv
        statistical_summary.json

Dependências:
    pip install pandas scipy

Uso:
    python src/evaluation/statistical_analysis.py

Observações metodológicas:
- Kendall é calculado em nível de vídeo × modelo:
      10 vídeos × 3 modelos = n esperado de 30 por dimensão/métrica.
- Friedman compara os 3 modelos usando o vídeo como bloco pareado:
      n esperado de 10 vídeos.
- Wilcoxon pareado é aplicado como pós-teste apenas quando o Friedman
  da respectiva variável apresenta p < alpha.
- A correção de Holm é implementada diretamente neste script.
- R@4 é "menor = melhor". O script NÃO inverte a métrica; portanto,
  correlação negativa com escores humanos/ACCR pode indicar concordância
  de direção.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, kendalltau, wilcoxon


# =============================================================================
# CAMINHOS
# =============================================================================

_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DIR.parent.parent

DEFAULT_HUMAN_VIDEO = (
    PROJECT_ROOT / "output" / "metrics" / "human"
    / "human_scores_by_video_model.csv"
)

DEFAULT_HUMAN_DIM = (
    PROJECT_ROOT / "output" / "metrics" / "human"
    / "human_scores_by_dimension.csv"
)

DEFAULT_HUMAN_LLM = (
    PROJECT_ROOT / "output" / "metrics" / "human"
    / "human_vs_llm_scores.csv"
)

DEFAULT_AUTO = (
    PROJECT_ROOT / "output" / "metrics" / "auto"
    / "metricas_por_video_multiref.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "metrics" / "statistical"
)


# =============================================================================
# CONSTANTES
# =============================================================================

MODELS = ["GPT-4.1", "Llama 4", "SkimCap"]

DIMENSIONS = [
    "accuracy",
    "completeness",
    "conciseness",
    "relevance",
]

AUTO_METRICS = [
    "Bleu_4",
    "ROUGE_L",
    "CIDEr",
    "METEOR",
    "R@4",
]

METRIC_DIRECTION = {
    "Bleu_4": "higher_better",
    "ROUGE_L": "higher_better",
    "CIDEr": "higher_better",
    "METEOR": "higher_better",
    "R@4": "lower_better",
}

MODEL_ALIASES = {
    "gpt": "GPT-4.1",
    "gpt-4.1": "GPT-4.1",
    "gpt 4.1": "GPT-4.1",
    "predictions_gpt": "GPT-4.1",
    "predictions-gpt": "GPT-4.1",

    "llama": "Llama 4",
    "llama 4": "Llama 4",
    "llama4": "Llama 4",
    "llama-4": "Llama 4",
    "predictions_llama": "Llama 4",
    "predictions-llama": "Llama 4",

    "skimcap": "SkimCap",
    "greedy_pred_test": "SkimCap",
    "greedy-pred-test": "SkimCap",
}


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def canonical_video_id(value) -> str:
    value = str(value).strip()
    if value.startswith("v_"):
        return value[2:]
    return value


def normalize_model(value) -> str:
    raw = str(value).strip()
    return MODEL_ALIASES.get(raw.lower(), raw)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: colunas obrigatórias ausentes: {missing}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} não encontrado: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} está vazio: {path}")

    return df


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Alguns arquivos têm video_key; outros só video_id.
    source_video = "video_key" if "video_key" in df.columns else "video_id"
    df["video_key"] = df[source_video].map(canonical_video_id)

    if "model" in df.columns:
        df["model"] = df["model"].map(normalize_model)

    return df


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def kendall_pair(x: pd.Series, y: pd.Series) -> dict:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(pair)

    result = {
        "n": n,
        "tau_b": None,
        "p_b": None,
        "tau_c": None,
        "p_c": None,
    }

    if n < 2:
        return result

    # Kendall não é definido se uma das variáveis for constante.
    if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        b = kendalltau(pair["x"], pair["y"], variant="b")
        c = kendalltau(pair["x"], pair["y"], variant="c")

    result.update({
        "tau_b": safe_float(b.statistic),
        "p_b": safe_float(b.pvalue),
        "tau_c": safe_float(c.statistic),
        "p_c": safe_float(c.pvalue),
    })

    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Correção de Holm step-down.

    Retorna p-valores ajustados na mesma ordem da lista original.
    """
    m = len(p_values)

    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])

    adjusted_sorted = []
    running_max = 0.0

    for rank, (original_index, p) in enumerate(indexed):
        multiplier = m - rank
        adjusted = min(1.0, multiplier * float(p))
        running_max = max(running_max, adjusted)
        adjusted_sorted.append((original_index, min(1.0, running_max)))

    adjusted = [1.0] * m

    for original_index, p_adj in adjusted_sorted:
        adjusted[original_index] = p_adj

    return adjusted


def significance_label(p_value, alpha: float) -> str:
    if p_value is None or pd.isna(p_value):
        return "not_computable"
    return "yes" if float(p_value) < alpha else "no"


def warn_count(label: str, actual: int, expected: int) -> None:
    marker = "✓" if actual == expected else "⚠"
    print(f"  {marker} {label:<38} {actual:>4}  (esperado: {expected})")


# =============================================================================
# CARREGAMENTO / VALIDAÇÃO
# =============================================================================

def load_inputs(args):
    human_video = read_csv(Path(args.human_video), "human_scores_by_video_model.csv")
    human_dim = read_csv(Path(args.human_dimension), "human_scores_by_dimension.csv")
    human_llm = read_csv(Path(args.human_llm), "human_vs_llm_scores.csv")
    auto = read_csv(Path(args.auto_metrics), "metricas_por_video_multiref.csv")

    require_columns(
        human_video,
        ["video_id", "model", *DIMENSIONS, "mean_accr"],
        "human_scores_by_video_model.csv",
    )

    require_columns(
        human_dim,
        ["video_id", "model", "dimension", "score_human"],
        "human_scores_by_dimension.csv",
    )

    require_columns(
        human_llm,
        ["video_id", "model", "dimension", "score_human", "score_llm"],
        "human_vs_llm_scores.csv",
    )

    require_columns(
        auto,
        ["video_id", "model", *AUTO_METRICS],
        "metricas_por_video_multiref.csv",
    )

    human_video = add_keys(human_video)
    human_dim = add_keys(human_dim)
    human_llm = add_keys(human_llm)
    auto = add_keys(auto)

    human_video = numeric(human_video, [*DIMENSIONS, "mean_accr"])
    human_dim = numeric(human_dim, ["score_human"])
    human_llm = numeric(human_llm, ["score_human", "score_llm"])
    auto = numeric(auto, AUTO_METRICS)

    human_dim["dimension"] = human_dim["dimension"].astype(str).str.lower().str.strip()
    human_llm["dimension"] = human_llm["dimension"].astype(str).str.lower().str.strip()

    return human_video, human_dim, human_llm, auto


def validate_inputs(human_video, human_dim, human_llm, auto):
    print("\nVALIDAÇÃO DOS DADOS")
    print("─" * 64)

    warn_count(
        "Humanos: vídeo × modelo",
        len(human_video[["video_key", "model"]].drop_duplicates()),
        30,
    )

    warn_count(
        "Humanos: vídeo × modelo × dimensão",
        len(human_dim[["video_key", "model", "dimension"]].drop_duplicates()),
        120,
    )

    warn_count(
        "Humano × LLM: pares",
        len(human_llm[["video_key", "model", "dimension"]].drop_duplicates()),
        120,
    )

    warn_count(
        "Métricas automáticas: vídeo × modelo",
        len(auto[["video_key", "model"]].drop_duplicates()),
        30,
    )

    print("\n  Modelos humanos :", sorted(human_dim["model"].dropna().unique()))
    print("  Modelos ACCR    :", sorted(human_llm["model"].dropna().unique()))
    print("  Modelos auto    :", sorted(auto["model"].dropna().unique()))

    cider_missing = int(auto["CIDEr"].isna().sum())
    if cider_missing:
        print(f"\n  ⚠ CIDEr possui {cider_missing} valor(es) ausente(s).")
    else:
        print("\n  ✓ CIDEr completo.")


# =============================================================================
# KENDALL
# =============================================================================

def kendall_human_vs_llm(human_llm: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dimension in DIMENSIONS:
        sub = human_llm[human_llm["dimension"] == dimension]
        stats = kendall_pair(sub["score_human"], sub["score_llm"])

        rows.append({
            "dimension": dimension,
            **stats,
            "expected_n": 30,
        })

    return pd.DataFrame(rows)


def build_human_auto_merge(
    human_dim: pd.DataFrame,
    auto: pd.DataFrame,
) -> pd.DataFrame:

    auto_keep = auto[
        ["video_key", "model", "video_id", *AUTO_METRICS]
    ].copy()

    auto_keep = auto_keep.rename(columns={"video_id": "auto_video_id"})

    merged = human_dim.merge(
        auto_keep,
        on=["video_key", "model"],
        how="inner",
        validate="many_to_one",
    )

    return merged


def kendall_human_vs_auto(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dimension in DIMENSIONS:
        sub = merged[merged["dimension"] == dimension]

        for metric in AUTO_METRICS:
            stats = kendall_pair(sub["score_human"], sub[metric])
            direction = METRIC_DIRECTION[metric]

            rows.append({
                "dimension": dimension,
                "metric": metric,
                "metric_direction": direction,
                **stats,
                "expected_n": 30,
                "direction_note": (
                    "negative_tau_can_indicate_agreement"
                    if direction == "lower_better"
                    else "positive_tau_indicates_same_direction"
                ),
            })

    return pd.DataFrame(rows)


def build_llm_auto_merge(
    human_llm: pd.DataFrame,
    auto: pd.DataFrame,
) -> pd.DataFrame:

    llm = human_llm[
        ["video_key", "model", "dimension", "score_llm"]
    ].drop_duplicates(
        ["video_key", "model", "dimension"]
    )

    auto_keep = auto[
        ["video_key", "model", "video_id", *AUTO_METRICS]
    ].copy()

    auto_keep = auto_keep.rename(columns={"video_id": "auto_video_id"})

    merged = llm.merge(
        auto_keep,
        on=["video_key", "model"],
        how="inner",
        validate="many_to_one",
    )

    return merged


def kendall_llm_vs_auto(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dimension in DIMENSIONS:
        sub = merged[merged["dimension"] == dimension]

        for metric in AUTO_METRICS:
            stats = kendall_pair(sub["score_llm"], sub[metric])
            direction = METRIC_DIRECTION[metric]

            rows.append({
                "dimension": dimension,
                "metric": metric,
                "metric_direction": direction,
                **stats,
                "expected_n": 30,
                "direction_note": (
                    "negative_tau_can_indicate_agreement"
                    if direction == "lower_better"
                    else "positive_tau_indicates_same_direction"
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# FRIEDMAN + WILCOXON/HOLM
# =============================================================================

def paired_matrix(
    df: pd.DataFrame,
    value_col: str,
    extra_filter: tuple[str, str] | None = None,
) -> pd.DataFrame:

    sub = df.copy()

    if extra_filter:
        column, value = extra_filter
        sub = sub[sub[column] == value]

    pivot = sub.pivot_table(
        index="video_key",
        columns="model",
        values=value_col,
        aggfunc="mean",
    )

    existing = [m for m in MODELS if m in pivot.columns]

    if len(existing) < 3:
        return pd.DataFrame()

    pivot = pivot[MODELS].dropna()

    return pivot


def friedman_one(pivot: pd.DataFrame) -> dict:
    n = len(pivot)
    k = len(MODELS)

    if n < 2:
        return {
            "n_blocks": n,
            "chi_square": None,
            "p_value": None,
            "kendalls_w": None,
        }

    arrays = [pivot[model].to_numpy() for model in MODELS]

    try:
        result = friedmanchisquare(*arrays)
        chi2 = safe_float(result.statistic)
        p = safe_float(result.pvalue)
    except ValueError:
        chi2 = None
        p = None

    w = None
    if chi2 is not None and n > 0 and k > 1:
        w = chi2 / (n * (k - 1))

    return {
        "n_blocks": n,
        "chi_square": chi2,
        "p_value": p,
        "kendalls_w": safe_float(w),
    }


def wilcoxon_pairwise(
    pivot: pd.DataFrame,
    family: str,
    variable: str,
    friedman_p,
    alpha: float,
    direction: str = "higher_better",
) -> list[dict]:

    if friedman_p is None or pd.isna(friedman_p) or friedman_p >= alpha:
        return []

    raw_rows = []

    for model_a, model_b in combinations(MODELS, 2):
        a = pivot[model_a]
        b = pivot[model_b]

        diff = a - b

        # scipy falha quando todas as diferenças são zero.
        if (diff == 0).all():
            statistic = 0.0
            p_value = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = wilcoxon(
                    a,
                    b,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            statistic = safe_float(result.statistic)
            p_value = safe_float(result.pvalue)

        raw_rows.append({
            "family": family,
            "variable": variable,
            "model_a": model_a,
            "model_b": model_b,
            "n_pairs": len(pivot),
            "statistic": statistic,
            "p_raw": p_value,
            "median_a": safe_float(a.median()),
            "median_b": safe_float(b.median()),
            "median_difference_a_minus_b": safe_float(diff.median()),
            "direction": direction,
        })

    valid_p = [
        row["p_raw"] if row["p_raw"] is not None else 1.0
        for row in raw_rows
    ]

    adjusted = holm_adjust(valid_p)

    for row, p_adj in zip(raw_rows, adjusted):
        row["p_holm"] = p_adj
        row["significant_holm"] = significance_label(p_adj, alpha)

    return raw_rows


def friedman_human(
    human_video: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    friedman_rows = []
    wilcoxon_rows = []

    variables = [*DIMENSIONS, "mean_accr"]

    for variable in variables:
        pivot = paired_matrix(human_video, variable)
        stats = friedman_one(pivot)

        friedman_rows.append({
            "family": "human",
            "variable": variable,
            **stats,
            "expected_n_blocks": 10,
            "alpha": alpha,
            "significant": significance_label(stats["p_value"], alpha),
            "direction": "higher_better",
        })

        wilcoxon_rows.extend(
            wilcoxon_pairwise(
                pivot=pivot,
                family="human",
                variable=variable,
                friedman_p=stats["p_value"],
                alpha=alpha,
                direction="higher_better",
            )
        )

    return pd.DataFrame(friedman_rows), pd.DataFrame(wilcoxon_rows)


def friedman_llm(
    human_llm: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    friedman_rows = []
    wilcoxon_rows = []

    for dimension in DIMENSIONS:
        pivot = paired_matrix(
            human_llm,
            value_col="score_llm",
            extra_filter=("dimension", dimension),
        )

        stats = friedman_one(pivot)

        friedman_rows.append({
            "family": "llm_accr",
            "variable": dimension,
            **stats,
            "expected_n_blocks": 10,
            "alpha": alpha,
            "significant": significance_label(stats["p_value"], alpha),
            "direction": "higher_better",
        })

        wilcoxon_rows.extend(
            wilcoxon_pairwise(
                pivot=pivot,
                family="llm_accr",
                variable=dimension,
                friedman_p=stats["p_value"],
                alpha=alpha,
                direction="higher_better",
            )
        )

    return pd.DataFrame(friedman_rows), pd.DataFrame(wilcoxon_rows)


def friedman_auto(
    auto: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    friedman_rows = []
    wilcoxon_rows = []

    for metric in AUTO_METRICS:
        pivot = paired_matrix(auto, metric)
        stats = friedman_one(pivot)
        direction = METRIC_DIRECTION[metric]

        friedman_rows.append({
            "family": "automatic_metrics",
            "variable": metric,
            **stats,
            "expected_n_blocks": 10,
            "alpha": alpha,
            "significant": significance_label(stats["p_value"], alpha),
            "direction": direction,
        })

        wilcoxon_rows.extend(
            wilcoxon_pairwise(
                pivot=pivot,
                family="automatic_metrics",
                variable=metric,
                friedman_p=stats["p_value"],
                alpha=alpha,
                direction=direction,
            )
        )

    return pd.DataFrame(friedman_rows), pd.DataFrame(wilcoxon_rows)


# =============================================================================
# SAÍDA
# =============================================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    cleaned = df.where(pd.notnull(df), None)

    records = []
    for row in cleaned.to_dict(orient="records"):
        records.append({
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in row.items()
        })

    return records


def print_kendall_preview(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("─" * 64)

    if df.empty:
        print("  Nenhum resultado.")
        return

    cols = [
        col for col in
        ["dimension", "metric", "n", "tau_b", "p_b", "tau_c", "p_c"]
        if col in df.columns
    ]

    print(df[cols].to_string(index=False))


def print_friedman_preview(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("─" * 64)

    if df.empty:
        print("  Nenhum resultado.")
        return

    cols = [
        "variable",
        "n_blocks",
        "chi_square",
        "p_value",
        "kendalls_w",
        "significant",
    ]

    print(df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Análise estatística final do TCC."
    )

    parser.add_argument(
        "--human-video",
        default=str(DEFAULT_HUMAN_VIDEO),
        help="human_scores_by_video_model.csv",
    )

    parser.add_argument(
        "--human-dimension",
        default=str(DEFAULT_HUMAN_DIM),
        help="human_scores_by_dimension.csv",
    )

    parser.add_argument(
        "--human-llm",
        default=str(DEFAULT_HUMAN_LLM),
        help="human_vs_llm_scores.csv",
    )

    parser.add_argument(
        "--auto-metrics",
        default=str(DEFAULT_AUTO),
        help="metricas_por_video_multiref.csv",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório de saída",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância. Padrão: 0.05",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  ANÁLISE ESTATÍSTICA FINAL")
    print("=" * 72)

    human_video, human_dim, human_llm, auto = load_inputs(args)

    validate_inputs(
        human_video=human_video,
        human_dim=human_dim,
        human_llm=human_llm,
        auto=auto,
    )

    # -------------------------------------------------------------------------
    # Kendall
    # -------------------------------------------------------------------------

    k_human_llm = kendall_human_vs_llm(human_llm)

    merged_human_auto = build_human_auto_merge(human_dim, auto)
    k_human_auto = kendall_human_vs_auto(merged_human_auto)

    merged_llm_auto = build_llm_auto_merge(human_llm, auto)
    k_llm_auto = kendall_llm_vs_auto(merged_llm_auto)

    # -------------------------------------------------------------------------
    # Friedman + Wilcoxon/Holm
    # -------------------------------------------------------------------------

    f_human, w_human = friedman_human(human_video, args.alpha)
    f_llm, w_llm = friedman_llm(human_llm, args.alpha)
    f_auto, w_auto = friedman_auto(auto, args.alpha)

    # -------------------------------------------------------------------------
    # Salva
    # -------------------------------------------------------------------------

    outputs = {
        "kendall_human_vs_llm.csv": k_human_llm,
        "kendall_human_vs_auto_metrics.csv": k_human_auto,
        "kendall_llm_vs_auto_metrics.csv": k_llm_auto,
        "friedman_human.csv": f_human,
        "wilcoxon_human_holm.csv": w_human,
        "friedman_llm.csv": f_llm,
        "wilcoxon_llm_holm.csv": w_llm,
        "friedman_auto_metrics.csv": f_auto,
        "wilcoxon_auto_metrics_holm.csv": w_auto,
        "merged_human_auto.csv": merged_human_auto,
        "merged_llm_auto.csv": merged_llm_auto,
    }

    for filename, df in outputs.items():
        save_csv(df, out_dir / filename)

    summary = {
        "alpha": args.alpha,
        "expected_design": {
            "participants_final": 36,
            "videos": 10,
            "models": 3,
            "dimensions": 4,
            "kendall_n_per_dimension_or_metric": 30,
            "friedman_blocks": 10,
        },
        "metric_direction": METRIC_DIRECTION,
        "notes": [
            "R@4 is lower-is-better and is not inverted before Kendall.",
            "A negative Kendall tau for R@4 can therefore indicate directional agreement with higher-is-better human/ACCR scores.",
            "Wilcoxon post-hoc tests are generated only when the corresponding Friedman test has p < alpha.",
            "Holm correction is applied within each family of three pairwise model comparisons for each variable.",
        ],
        "kendall_human_vs_llm": dataframe_records(k_human_llm),
        "friedman_human": dataframe_records(f_human),
        "friedman_llm": dataframe_records(f_llm),
        "friedman_auto_metrics": dataframe_records(f_auto),
    }

    (out_dir / "statistical_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal
    # -------------------------------------------------------------------------

    print_kendall_preview(
        "KENDALL — HUMANO × ACCR AUTOMÁTICO",
        k_human_llm,
    )

    print_friedman_preview(
        "FRIEDMAN — AVALIAÇÃO HUMANA",
        f_human,
    )

    print_friedman_preview(
        "FRIEDMAN — ACCR AUTOMÁTICO",
        f_llm,
    )

    print_friedman_preview(
        "FRIEDMAN — MÉTRICAS TRADICIONAIS",
        f_auto,
    )

    print("\n" + "=" * 72)
    print(f"  Resultados salvos em: {out_dir.resolve()}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
