#!/usr/bin/env python3
"""
human_eval_final.py

Processa a avaliação humana ACCR exportada pelo Google Forms.

Compatível com a planilha real, cujas respostas são textos como:
    "1 - Discordo totalmente"
    "2 - Discordo"
    "3 - Neutro"
    "4 - Concordo"
    "5 - Concordo totalmente"

Fluxo:
1. Lê o CSV do Forms.
2. Lê o gabarito Video_ID + A/B/C.
3. Desfaz o cegamento e identifica GPT-4.1, Llama 4 e SkimCap.
4. Converte Likert 1–5 para POMP 0–100:
       POMP = ((Likert - 1) / 4) * 100
5. Calcula a média humana por vídeo × modelo × dimensão.
6. Calcula ICC(2,k) por dimensão.
7. Se --llm-results for informado, cruza com llm_eval.py e calcula
   Kendall tau-b e tau-c.

Uso:
    python human_eval_final.py \
      --responses "respostas.csv" \
      --gabarito "gabarito_legendas_videos.csv" \
      --output-dir output/metrics/human

Opcional:
    python human_eval_final.py \
      --responses "respostas.csv" \
      --gabarito "gabarito_legendas_videos.csv" \
      --llm-results output/metrics/accr/accr_predictions_multiref.json \
      --output-dir output/metrics/human
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path

DIMENSIONS = ["accuracy", "completeness", "conciseness", "relevance"]
LETTERS = ["A", "B", "C"]

MODEL_ALIASES = {
    "GPT": "GPT-4.1",
    "GPT-4.1": "GPT-4.1",
    "Llama": "Llama 4",
    "Llama 4": "Llama 4",
    "LLaMA": "Llama 4",
    "SkimCap": "SkimCap",
}

# Pasta onde está o human_eval.py
BASE = Path(__file__).parent

# Pasta dos arquivos da avaliação humana
HUMAN_EVAL_DIR = BASE / "human eval"

DEFAULT_RESPONSES = HUMAN_EVAL_DIR / "respostas_forms.csv"
DEFAULT_GABARITO = HUMAN_EVAL_DIR / "gabarito_legendas_videos.csv"

# src/evaluation -> raiz do projeto
PROJECT_ROOT = BASE.parent.parent

DEFAULT_LLM_RESULTS = (
    PROJECT_ROOT
    / "output"
    / "metrics"
    / "accr"
    / "accr_predictions_multiref.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "metrics"
    / "human"
)

def normalize_model_name(name):
    raw = str(name).strip()

    aliases = {
        "gpt": "GPT-4.1",
        "gpt-4.1": "GPT-4.1",
        "gpt 4.1": "GPT-4.1",

        "llama": "Llama 4",
        "llama 4": "Llama 4",
        "llama4": "Llama 4",

        "skimcap": "SkimCap",
    }

    return aliases.get(raw.lower(), raw)


def canonical_video_id(video_id):
    video_id = str(video_id).strip()
    return video_id[2:] if video_id.startswith("v_") else video_id


def pomp(likert):
    return ((likert - 1) / 4.0) * 100.0


def parse_likert(value):
    """
    Aceita tanto 4 quanto '4 - Concordo'.
    """
    text = str(value).strip()
    match = re.match(r"^([1-5])(?:\D|$)", text)
    if not match:
        raise ValueError(f"Valor Likert inválido: {value!r}")
    return int(match.group(1))


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        return list(csv.reader(f, delimiter=delimiter))


def read_gabarito(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError("Gabarito vazio.")

    header = rows[0]
    required = ["Video", "Video_ID", "A", "B", "C"]

    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Colunas ausentes no gabarito: {missing}")

    idx = {name: header.index(name) for name in header}
    mapping = []

    for row in rows[1:]:
        if not row or not any(str(v).strip() for v in row):
            continue

        item = {
            "video_index": int(row[idx["Video"]]),
            "video_id": row[idx["Video_ID"]].strip(),
            "A": normalize_model_name(row[idx["A"]]),
            "B": normalize_model_name(row[idx["B"]]),
            "C": normalize_model_name(row[idx["C"]]),
        }

        models = [item["A"], item["B"], item["C"]]
        if len(set(models)) != 3:
            raise ValueError(
                f"Vídeo {item['video_index']}: A/B/C devem apontar para 3 modelos diferentes."
            )

        mapping.append(item)

    mapping.sort(key=lambda x: x["video_index"])
    return mapping


def validate_headers(header, n_videos):
    expected_total = 3 + n_videos * 12

    if len(header) != expected_total:
        raise ValueError(
            f"A planilha possui {len(header)} colunas; eram esperadas "
            f"{expected_total} (3 metadados + {n_videos}×12 avaliações)."
        )

    expected_dims = ["Acurácia", "Completude", "Concisão", "Relevância"]
    expected = []

    for _ in range(n_videos):
        for letter in LETTERS:
            for dim in expected_dims:
                expected.append(f"Avaliação da Legenda {letter} [{dim}]")

    if header[3:] != expected:
        for i, (actual, exp) in enumerate(zip(header[3:], expected), start=4):
            if actual != exp:
                raise ValueError(
                    f"Ordem inesperada na coluna {i}: {actual!r}; esperado {exp!r}"
                )


def build_long_records(response_rows, mapping):
    header = response_rows[0]
    data = response_rows[1:]

    validate_headers(header, len(mapping))

    records = []

    for participant_idx, row in enumerate(data, start=1):
        if not row or not any(str(v).strip() for v in row):
            continue

        if len(row) != len(header):
            raise ValueError(
                f"Participante {participant_idx}: linha possui {len(row)} colunas, "
                f"mas o cabeçalho possui {len(header)}."
            )

        timestamp = row[0]
        consent = row[1]
        english = row[2]

        for video_pos, cfg in enumerate(mapping):
            base_video = 3 + video_pos * 12

            for letter_pos, letter in enumerate(LETTERS):
                model = cfg[letter]
                base_caption = base_video + letter_pos * 4

                for dim_pos, dimension in enumerate(DIMENSIONS):
                    raw_value = row[base_caption + dim_pos]

                    if not str(raw_value).strip():
                        continue

                    likert = parse_likert(raw_value)

                    records.append({
                        "participant_id": participant_idx,
                        "timestamp": timestamp,
                        "consent": consent,
                        "english_proficiency": english,
                        "video_index": cfg["video_index"],
                        "video_id": cfg["video_id"],
                        "video_key": canonical_video_id(cfg["video_id"]),
                        "legend": letter,
                        "model": model,
                        "dimension": dimension,
                        "likert": likert,
                        "score_pomp": pomp(likert),
                    })

    return records


def aggregate(records):
    grouped = defaultdict(list)

    for r in records:
        key = (
            r["video_index"],
            r["video_id"],
            r["video_key"],
            r["model"],
            r["dimension"],
        )
        grouped[key].append(r["score_pomp"])

    by_dimension = []

    for key in sorted(grouped):
        video_index, video_id, video_key, model, dimension = key
        values = grouped[key]

        by_dimension.append({
            "video_index": video_index,
            "video_id": video_id,
            "video_key": video_key,
            "model": model,
            "dimension": dimension,
            "n": len(values),
            "score_human": round(statistics.mean(values), 4),
            "sd_human": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        })

    wide_map = defaultdict(dict)

    for row in by_dimension:
        key = (
            row["video_index"],
            row["video_id"],
            row["video_key"],
            row["model"],
        )
        wide_map[key][row["dimension"]] = row["score_human"]

    by_video_model = []

    for key in sorted(wide_map):
        video_index, video_id, video_key, model = key
        scores = wide_map[key]

        row = {
            "video_index": video_index,
            "video_id": video_id,
            "video_key": video_key,
            "model": model,
        }

        for d in DIMENSIONS:
            row[d] = scores.get(d)

        valid = [row[d] for d in DIMENSIONS if row[d] is not None]
        row["mean_accr"] = round(statistics.mean(valid), 4) if valid else None
        by_video_model.append(row)

    model_groups = defaultdict(lambda: defaultdict(list))

    for row in by_video_model:
        for d in DIMENSIONS + ["mean_accr"]:
            if row[d] is not None:
                model_groups[row["model"]][d].append(row[d])

    model_summary = []

    for model in ["GPT-4.1", "Llama 4", "SkimCap"]:
        row = {"model": model}
        for d in DIMENSIONS + ["mean_accr"]:
            vals = model_groups[model][d]
            row[d] = round(statistics.mean(vals), 4) if vals else None
        model_summary.append(row)

    return by_dimension, by_video_model, model_summary


def icc_2k(records):
    """
    ICC(2,k): two-way random effects, absolute agreement,
    average measures.

    Calculado separadamente por dimensão.

    Cada target = vídeo|modelo
    Cada rater  = participante
    """
    results = {}

    for dimension in DIMENSIONS:
        sub = [r for r in records if r["dimension"] == dimension]

        targets = sorted({
            (r["video_key"], r["model"]) for r in sub
        })
        raters = sorted({r["participant_id"] for r in sub})

        lookup = {
            ((r["video_key"], r["model"]), r["participant_id"]): r["likert"]
            for r in sub
        }

        # Para ICC balanceado, todos os avaliadores precisam ter avaliado todos os targets.
        if any((target, rater) not in lookup for target in targets for rater in raters):
            results[dimension] = {
                "type": "ICC(2,k)",
                "icc": None,
                "warning": "Dados incompletos; ICC balanceado não calculado.",
            }
            continue

        n = len(targets)
        k = len(raters)

        matrix = [
            [float(lookup[(target, rater)]) for rater in raters]
            for target in targets
        ]

        grand = sum(sum(row) for row in matrix) / (n * k)
        row_means = [sum(row) / k for row in matrix]
        col_means = [
            sum(matrix[i][j] for i in range(n)) / n
            for j in range(k)
        ]

        ss_rows = k * sum((m - grand) ** 2 for m in row_means)
        ss_cols = n * sum((m - grand) ** 2 for m in col_means)

        ss_error = 0.0
        for i in range(n):
            for j in range(k):
                residual = matrix[i][j] - row_means[i] - col_means[j] + grand
                ss_error += residual ** 2

        ms_rows = ss_rows / (n - 1)
        ms_cols = ss_cols / (k - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))

        denominator = ms_rows + (ms_cols - ms_error) / n
        icc = (ms_rows - ms_error) / denominator if denominator != 0 else None

        results[dimension] = {
            "type": "ICC(2,k)",
            "targets": n,
            "raters": k,
            "icc": round(icc, 6) if icc is not None else None,
            "MS_target": round(ms_rows, 6),
            "MS_rater": round(ms_cols, 6),
            "MS_error": round(ms_error, 6),
        }

    return results


def load_llm_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for video_id, segments in data.get("resultados_por_video", {}).items():
        for seg in segments:
            for model, evaluation in seg.get("avaliacoes", {}).items():
                scores = evaluation.get("scores", {})

                for dimension in DIMENSIONS:
                    value = scores.get(dimension)
                    if value is None:
                        continue

                    rows.append({
                        "video_key": canonical_video_id(video_id),
                        "llm_video_id": video_id,
                        "model": normalize_model_name(model),
                        "dimension": dimension,
                        "score_llm": float(value),
                    })

    if not rows:
        raise ValueError(
            "O JSON informado não contém scores em resultados_por_video."
        )

    return rows


def compare_with_llm(by_dimension, llm_rows):
    try:
        from scipy.stats import kendalltau
    except ImportError as exc:
        raise RuntimeError(
            "Para calcular Kendall, instale scipy: pip install scipy"
        ) from exc

    llm_lookup = {
        (r["video_key"], r["model"], r["dimension"]): r
        for r in llm_rows
    }

    merged = []

    for human in by_dimension:
        key = (
            human["video_key"],
            human["model"],
            human["dimension"],
        )

        llm = llm_lookup.get(key)
        if not llm:
            continue

        merged.append({
            **human,
            "llm_video_id": llm["llm_video_id"],
            "score_llm": llm["score_llm"],
        })

    if not merged:
        raise ValueError(
            "O cruzamento humano × LLM ficou vazio. Verifique Video_IDs "
            "e nomes dos modelos."
        )

    stats = {}

    for dimension in DIMENSIONS:
        sub = [r for r in merged if r["dimension"] == dimension]

        human_scores = [r["score_human"] for r in sub]
        llm_scores = [r["score_llm"] for r in sub]

        if len(sub) < 2:
            stats[dimension] = {
                "n": len(sub),
                "tau_b": None,
                "p_b": None,
                "tau_c": None,
                "p_c": None,
            }
            continue

        b = kendalltau(human_scores, llm_scores, variant="b")
        c = kendalltau(human_scores, llm_scores, variant="c")

        stats[dimension] = {
            "n": len(sub),
            "tau_b": round(float(b.statistic), 6),
            "p_b": round(float(b.pvalue), 6),
            "tau_c": round(float(c.statistic), 6),
            "p_c": round(float(c.pvalue), 6),
        }

    return merged, stats


def write_csv(path, rows, fieldnames=None):
    path = Path(path)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação humana ACCR do Google Forms."
    )

    parser.add_argument(
    "--responses", "-r",
    default=str(DEFAULT_RESPONSES),
    help="CSV exportado do Google Forms",
)

    parser.add_argument(
        "--gabarito", "-g",
        default=str(DEFAULT_GABARITO),
        help="CSV contendo Video, Video_ID, A, B e C",
    )

    parser.add_argument(
        "--llm-results", "-l",
        default=str(DEFAULT_LLM_RESULTS),
        help="accr_predictions_multiref.json produzido pelo llm_eval.py",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Pasta de saída",
    )

    args = parser.parse_args()
    
    print(f"Respostas: {args.responses}")
    print(f"Gabarito: {args.gabarito}")
    print(f"Saída: {args.output_dir}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    response_rows = read_csv(args.responses)
    mapping = read_gabarito(args.gabarito)

    records = build_long_records(response_rows, mapping)
    by_dimension, by_video_model, model_summary = aggregate(records)

    n_participants = len({r["participant_id"] for r in records})
    consent_counts = Counter(r["consent"] for r in records)
    english_counts = Counter(
        next(
            r["english_proficiency"]
            for r in records
            if r["participant_id"] == participant
        )
        for participant in sorted({r["participant_id"] for r in records})
    )

    write_csv(out / "human_ratings_long.csv", records)
    write_csv(out / "human_scores_by_dimension.csv", by_dimension)
    write_csv(out / "human_scores_by_video_model.csv", by_video_model)
    write_csv(out / "human_model_summary.csv", model_summary)
    write_csv(out / "mapping_used.csv", mapping)

    icc = icc_2k(records)
    write_json(out / "human_icc.json", icc)

    metadata = {
        "participants": n_participants,
        "individual_ratings": len(records),
        "consent_counts": dict(consent_counts),
        "english_proficiency_counts": dict(english_counts),
        "normalization": "POMP = ((Likert - 1) / 4) * 100",
        "likert_mapping": {
            "1": 0,
            "2": 25,
            "3": 50,
            "4": 75,
            "5": 100,
        },
    }
    write_json(out / "human_metadata.json", metadata)

    print("=" * 72)
    print("AVALIAÇÃO HUMANA ACCR")
    print("=" * 72)
    print(f"Participantes : {n_participants}")
    print(f"Avaliações    : {len(records)}")
    print(f"Vídeos        : {len(mapping)}")
    print()

    print("Resumo por modelo (0–100):")
    for row in model_summary:
        print(
            f"{row['model']:<10} "
            f"Acc={row['accuracy']:6.2f} "
            f"Comp={row['completeness']:6.2f} "
            f"Conc={row['conciseness']:6.2f} "
            f"Rel={row['relevance']:6.2f} "
            f"Média={row['mean_accr']:6.2f}"
        )

    print("\nICC(2,k):")
    for dimension, value in icc.items():
        print(f"  {dimension:<13}: {value.get('icc')}")

    if args.llm_results:
        llm_rows = load_llm_results(args.llm_results)
        merged, kendall = compare_with_llm(by_dimension, llm_rows)

        write_csv(out / "human_vs_llm_scores.csv", merged)
        write_json(out / "human_vs_llm_kendall.json", kendall)

        print("\nKendall humano × ACCR automático:")
        for dimension, value in kendall.items():
            print(
                f"  {dimension:<13} "
                f"n={value['n']:>2} "
                f"tau-b={value['tau_b']} "
                f"tau-c={value['tau_c']}"
            )

    print(f"\nSaída: {out.resolve()}")


if __name__ == "__main__":
    main()
