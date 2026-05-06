#!/usr/bin/env python3
"""
pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline completo de geração + avaliação de legendas.

Etapas:
  1. Geração  — dois modelos LLM (model1 + model2) via main.py
  2. ACCR     — LLM como avaliador (model1 + model2 + SkimCap) via llm_eval.py
  3. Auto     — métricas automáticas (model1 + model2 + SkimCap) via auto_metrics.py
  4. Plots    — gráficos comparativos via scripts/gerar_graficos.py

Uso rápido:
  python pipeline.py                          # roda tudo com defaults
  python pipeline.py --skip-gen               # pula geração (usa JSONs existentes)
  python pipeline.py --limit 5               # processa só 5 vídeos
  python pipeline.py --skip-accr             # pula avaliação ACCR (economiza tokens)
  python pipeline.py --skip-plots            # pula geração de gráficos
  python pipeline.py --help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Importa config do agente para alinhar defaults de provider
sys.path.insert(0, str(Path(__file__).parent / "src" / "agent"))
import config as _cfg

# ─── Caminhos base ────────────────────────────────────────────────
_ROOT    = Path(__file__).parent
_SRC     = _ROOT / "src"
_DATA    = _ROOT / "data"
_OUTPUT  = _ROOT / "output"
_PRED    = _OUTPUT / "predictions"   # predictions.json, predictions_llama.json ...
_AUTO    = _OUTPUT / "metrics" / "auto"  # metricas_*.json
_ACCR    = _OUTPUT / "metrics" / "accr"  # accr_checkpoint_*.json
_PLOTS   = _OUTPUT / "plots"              # gráficos gerados
_GT_1    = str(_DATA / "ground_truth" / "anet_entities_test_1.json")
_GT_2    = str(_DATA / "ground_truth" / "anet_entities_test_2.json")
_SKIMCAP = str(_DATA / "baselines" / "greedy_pred_test.json")

# ─── Python executável (prefere o venv do projeto) ────────────────
def _python() -> str:
    """Retorna o Python do venv se existir, senão sys.executable."""
    for candidato in (_ROOT / ".venv" / "bin" / "python3", _ROOT / ".venv" / "bin" / "python"):
        if candidato.exists():
            return str(candidato)
    return sys.executable

_PYTHON = _python()

# ─── Helpers ─────────────────────────────────────────────────────

def _passo(n: int, titulo: str) -> None:
    SEP = "═" * 60
    print(f"\n{SEP}")
    print(f"  PASSO {n}: {titulo}")
    print(SEP)


def _rodar(args: list[str], descricao: str) -> bool:
    """Executa um subprocesso. Retorna True se bem-sucedido."""
    print(f"\n▶ {descricao}")
    print(f"  $ {' '.join(args)}\n")
    resultado = subprocess.run(args, check=False)
    if resultado.returncode != 0:
        print(f"\n❌ Falha em: {descricao} (código {resultado.returncode})")
        return False
    print(f"\n✅ Concluído: {descricao}")
    return True


# ═════════════════════════════════════════════════════════════════
# TABELA COMPARATIVA FINAL
# ═════════════════════════════════════════════════════════════════

def _tabela_comparativa(args, pred1_path: str, pred2_path: str) -> None:
    """Gera relatório comparativo em Markdown com tabelas e normalização."""
    from datetime import datetime

    nome_sc = "SkimCap"
    modelos_labels = [args.nome1, args.nome2, nome_sc]
    auto_metricas  = ["CIDEr", "Bleu_4", "ROUGE_L", "METEOR", "R@4"]
    accr_dims      = ["accuracy", "completeness", "conciseness", "relevance"]

    # ── Coleta métricas automáticas ───────────────────────────────
    auto_dados: dict[str, dict] = {}
    for gt_file in (_GT_1, _GT_2):
        sufixo = Path(gt_file).stem
        for label in modelos_labels:
            arquivo = _AUTO / f"metricas_{label}_{sufixo}.json"
            if not arquivo.exists():
                continue
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
            auto_dados[f"{label} ({sufixo})"] = {m: dados.get(m, 0.0) for m in auto_metricas}

    # ── Coleta ACCR (escala 0–100) ────────────────────────────────
    accr_dados: dict[str, dict] = {}
    for gt_file in (_GT_1, _GT_2):
        sufixo = Path(gt_file).stem
        for label in modelos_labels:
            nome_label = label.replace("/", "_")
            arquivo = _ACCR / f"accr_{nome_label}_{sufixo}.json"
            if not arquivo.exists():
                continue
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
            chave = f"{label} ({sufixo})"
            accr_dados[chave] = {d: dados[d]["media"] for d in accr_dims if d in dados}
            accr_dados[chave]["media_geral"] = dados.get("media_geral", 0.0)

    if not auto_dados and not accr_dados:
        return

    todas_chaves = sorted(set(auto_dados) | set(accr_dados))
    auto_chaves  = sorted(auto_dados)
    accr_chaves  = sorted(accr_dados)

    # ── Médias por modelo (agrega test_1 + test_2) ────────────────
    def _avg_auto(model: str) -> dict:
        entries = [v for k, v in auto_dados.items() if k.startswith(f"{model} (")]
        if not entries:
            return {}
        return {m: sum(e.get(m, 0.0) for e in entries) / len(entries) for m in auto_metricas}

    def _avg_accr(model: str) -> dict:
        entries = [v for k, v in accr_dados.items() if k.startswith(f"{model} (")]
        if not entries:
            return {}
        keys = accr_dims + ["media_geral"]
        return {d: sum(e.get(d, 0.0) for e in entries) / len(entries) for d in keys}

    # ── Helpers ───────────────────────────────────────────────────
    def _fa(d: dict, key: str) -> str:
        return "—" if not d or key not in d else f"{d[key]:.3f}"

    def _fr(d: dict, key: str) -> str:
        return "—" if not d or key not in d else f"{d[key]:.1f}"

    linhas: list[str] = []

    def L(s: str = "") -> None:
        linhas.append(s)

    # ══════════════════════════════════════════════════════════════
    # Cabeçalho
    # ══════════════════════════════════════════════════════════════
    L("# Relatório Comparativo Final")
    L()
    L(f"> Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ")
    L(f"> Modelos: **{args.nome1}** × **{args.nome2}** × **SkimCap**  ")
    L(f"> Ground truths: `anet_entities_test_1` e `anet_entities_test_2`")
    L()
    L("---")
    L()

    # ══════════════════════════════════════════════════════════════
    # SEÇÃO 1 — RESUMO POR MODELO (média dos dois GTs)
    # ══════════════════════════════════════════════════════════════
    L("## Resumo por Modelo")
    L()
    L("> Valores médios entre os dois arquivos de ground truth.")
    L()

    # 1a — Métricas automáticas (resumo)
    if auto_dados:
        L("### Métricas Automáticas — Média")
        L()
        L("| Modelo | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |")
        L("|---|---:|---:|---:|---:|---:|")
        for model in modelos_labels:
            avg = _avg_auto(model)
            if not avg:
                continue
            L(f"| **{model}** | {_fa(avg,'CIDEr')} | {_fa(avg,'Bleu_4')} | {_fa(avg,'ROUGE_L')} | {_fa(avg,'METEOR')} | {_fa(avg,'R@4')} |")
        L()

    # 1b — ACCR (resumo)
    if accr_dados:
        L("### ACCR — Média (escala 0–100)")
        L()
        L("| Modelo | Accuracy | Completeness | Conciseness | Relevance | Média Geral |")
        L("|---|---:|---:|---:|---:|---:|")
        for model in modelos_labels:
            avg = _avg_accr(model)
            if not avg:
                continue
            L(f"| **{model}** | {_fr(avg,'accuracy')} | {_fr(avg,'completeness')} | {_fr(avg,'conciseness')} | {_fr(avg,'relevance')} | {_fr(avg,'media_geral')} |")
        L()

    L("---")
    L()

    # ══════════════════════════════════════════════════════════════
    # SEÇÃO 2 — DETALHAMENTO POR GROUND TRUTH
    # ══════════════════════════════════════════════════════════════
    L("## Detalhamento por Ground Truth")
    L()

    # Tabela 1 — Auto métricas detalhada
    if auto_dados:
        L("### Tabela 1 — Métricas Automáticas por GT")
        L()
        L("| Modelo / GT | CIDEr-D | BLEU-4 | ROUGE-L | METEOR | R@4 ↓ |")
        L("|---|---:|---:|---:|---:|---:|")
        prev_model = None
        for chave in auto_chaves:
            model = chave.split(" (")[0]
            if prev_model and model != prev_model:
                L("| | | | | | |")
            prev_model = model
            a = auto_dados[chave]
            L(f"| {chave} | {_fa(a,'CIDEr')} | {_fa(a,'Bleu_4')} | {_fa(a,'ROUGE_L')} | {_fa(a,'METEOR')} | {_fa(a,'R@4')} |")
        L()

    # Tabela 2 — ACCR detalhada
    if accr_dados:
        L("### Tabela 2 — ACCR por GT (escala 0–100)")
        L()
        L("| Modelo / GT | Accuracy | Completeness | Conciseness | Relevance | Média |")
        L("|---|---:|---:|---:|---:|---:|")
        prev_model = None
        for chave in accr_chaves:
            model = chave.split(" (")[0]
            if prev_model and model != prev_model:
                L("| | | | | | |")
            prev_model = model
            r = accr_dados[chave]
            L(f"| {chave} | {_fr(r,'accuracy')} | {_fr(r,'completeness')} | {_fr(r,'conciseness')} | {_fr(r,'relevance')} | {_fr(r,'media_geral')} |")
        L()

    # Tabela 3 — Ranking detalhado
    spec_rank = []
    if auto_dados:
        spec_rank += [("CIDEr", auto_dados, "CIDEr"), ("BLEU-4", auto_dados, "Bleu_4"),
                      ("ROUGE-L", auto_dados, "ROUGE_L"), ("METEOR", auto_dados, "METEOR"),
                      ("R@4", auto_dados, "R@4")]
    if accr_dados:
        spec_rank += [("Accuracy", accr_dados, "accuracy"), ("Completeness", accr_dados, "completeness"),
                      ("Conciseness", accr_dados, "conciseness"), ("Relevance", accr_dados, "relevance"),
                      ("ACCR", accr_dados, "media_geral")]

    if spec_rank:
        ranks: dict[str, list] = {c: [] for c in todas_chaves}
        cols_rank = []
        for col_nome, fonte, key in spec_rank:
            vals = {c: fonte.get(c, {}).get(key) for c in todas_chaves}
            vals_v = {c: v for c, v in vals.items() if v is not None}
            if not vals_v:
                continue
            cols_rank.append(col_nome)
            reverse = (key != "R@4")
            ordenado = sorted(vals_v, key=lambda c: vals_v[c], reverse=reverse)
            for chave in todas_chaves:
                ranks[chave].append(ordenado.index(chave) + 1 if chave in vals_v else None)

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        L("### Tabela 3 — Ranking por Posição (🥇 = melhor, R@4 ↓ = menor é melhor)")
        L()
        L("| Modelo / GT | " + " | ".join(cols_rank) + " | Rank Médio |")
        L("|---|" + ":---:|" * len(cols_rank) + "---:|")
        prev_model = None
        for chave in todas_chaves:
            model = chave.split(" (")[0]
            if prev_model and model != prev_model:
                L("| | " + " | ".join("" for _ in cols_rank) + " | |")
            prev_model = model
            rs = ranks[chave]
            rs_v = [r for r in rs if r is not None]
            media_rank = sum(rs_v) / len(rs_v) if rs_v else None
            celulas = " | ".join(medals.get(r, str(r)) if r else "—" for r in rs)
            L(f"| {chave} | {celulas} | {f'{media_rank:.2f}' if media_rank else '—'} |")
        L()

    L("---")
    L()

    # ══════════════════════════════════════════════════════════════
    # SEÇÃO 3 — CONCLUSÃO
    # ══════════════════════════════════════════════════════════════
    L("## Conclusão")
    L()

    # Tabela de melhores por métrica automática
    if auto_dados:
        L("### Melhores por Métrica Automática")
        L()
        L("| Métrica | Melhor Modelo | Valor |")
        L("|---|---|---:|")
        for met, key in [("CIDEr-D","CIDEr"),("BLEU-4","Bleu_4"),("ROUGE-L","ROUGE_L"),("METEOR","METEOR")]:
            melhor = max(auto_chaves, key=lambda k: auto_dados[k].get(key, 0.0))
            L(f"| {met} | {melhor} | {auto_dados[melhor][key]:.3f} |")
        # R@4: menor é melhor
        melhor_r4 = min(auto_chaves, key=lambda k: auto_dados[k].get("R@4", 1.0))
        L(f"| R@4 ↓ (menor = melhor) | {melhor_r4} | {auto_dados[melhor_r4]['R@4']:.3f} |")
        L()

    if accr_dados:
        L("### Melhores por ACCR")
        L()
        L("| Dimensão | Melhor Modelo | Score |")
        L("|---|---|---:|")
        for dim, label in [("accuracy","Accuracy"),("completeness","Completeness"),
                           ("conciseness","Conciseness"),("relevance","Relevance"),
                           ("media_geral","ACCR Média Geral")]:
            melhor = max(accr_chaves, key=lambda k: accr_dados[k].get(dim, 0.0))
            L(f"| {label} | {melhor} | {accr_dados[melhor][dim]:.1f}/100 |")
        L()

    # ── Análise narrativa (comparando modelos, não variantes de GT) ──
    partes: list[str] = []

    if auto_dados and accr_dados:
        # Médias por modelo para comparação justa
        avg_a  = {m: _avg_auto(m) for m in modelos_labels}
        avg_r  = {m: _avg_accr(m) for m in modelos_labels}
        modelos_gen = [args.nome1, args.nome2]

        def _score_auto(model: str) -> float:
            a = avg_a.get(model, {})
            return sum(a.get(m, 0.0) for m in ["CIDEr", "Bleu_4", "ROUGE_L", "METEOR"]) / 4

        # — Auto: qual modelo gerado se sai melhor?
        melhor_auto  = max(modelos_gen, key=_score_auto)
        pior_auto    = min(modelos_gen, key=_score_auto)
        partes.append(
            f"Nas métricas automáticas (CIDEr-D, BLEU-4, ROUGE-L, METEOR), "
            f"**{melhor_auto}** obteve médias superiores a **{pior_auto}** "
            f"(CIDEr médio: {avg_a[melhor_auto].get('CIDEr',0):.3f} vs "
            f"{avg_a[pior_auto].get('CIDEr',0):.3f})."
        )

        # — SkimCap vs gerados (auto)
        sc_cider = avg_a.get("SkimCap", {}).get("CIDEr", 0.0)
        gen_cider = max(_score_auto(m) for m in modelos_gen)
        if sc_cider > max(avg_a[m].get("CIDEr", 0) for m in modelos_gen):
            partes.append(
                f"O baseline SkimCap dominou as métricas automáticas "
                f"(CIDEr médio: {sc_cider:.3f} vs {max(avg_a[m].get('CIDEr',0) for m in modelos_gen):.3f}). "
                f"Isso é esperado: o SkimCap foi treinado diretamente no dataset ActivityNet Entities, "
                f"favorecendo sobreposição lexical com o ground truth."
            )
        else:
            partes.append(
                f"Os modelos gerativos superaram o baseline SkimCap nas métricas automáticas "
                f"(CIDEr médio do melhor: {max(avg_a[m].get('CIDEr',0) for m in modelos_gen):.3f} "
                f"vs SkimCap: {sc_cider:.3f})."
            )

        # — ACCR: qual modelo gerado se sai melhor?
        melhor_accr_gen  = max(modelos_gen, key=lambda m: avg_r.get(m, {}).get("media_geral", 0.0))
        pior_accr_gen    = min(modelos_gen, key=lambda m: avg_r.get(m, {}).get("media_geral", 0.0))
        m_melhor = avg_r.get(melhor_accr_gen, {}).get("media_geral", 0.0)
        m_pior   = avg_r.get(pior_accr_gen,   {}).get("media_geral", 0.0)
        partes.append(
            f"Na avaliação ACCR (LLM como juiz), **{melhor_accr_gen}** obteve média de "
            f"{m_melhor:.1f}/100 contra {m_pior:.1f}/100 de **{pior_accr_gen}**."
        )

        # — SkimCap vs gerados (ACCR)
        sc_accr = avg_r.get("SkimCap", {}).get("media_geral", 0.0)
        if sc_accr > m_melhor:
            partes.append(
                f"Na avaliação ACCR, o SkimCap ({sc_accr:.1f}/100) também superou os modelos gerativos "
                f"({melhor_accr_gen}: {m_melhor:.1f}/100)."
            )
        else:
            partes.append(
                f"Diferentemente das métricas automáticas, na avaliação ACCR os modelos gerativos "
                f"({melhor_accr_gen}: {m_melhor:.1f}/100) superaram o SkimCap ({sc_accr:.1f}/100). "
                f"Isso evidencia a dissociação entre sobreposição lexical e qualidade semântica: "
                f"modelos generativos produzem legendas semanticamente melhores, mas com vocabulário "
                f"diferente do ground truth, penalizando-os nas métricas automáticas."
            )

        # — Vencedor geral
        if melhor_auto == melhor_accr_gen:
            partes.append(
                f"**{melhor_auto}** foi o modelo com melhor desempenho em ambos os critérios de avaliação, "
                f"consolidando-se como a abordagem superior neste experimento."
            )
        else:
            partes.append(
                f"Há uma divergência entre os critérios: **{melhor_auto}** liderou as métricas automáticas "
                f"enquanto **{melhor_accr_gen}** foi superior na avaliação ACCR. "
                f"A escolha do melhor modelo depende da prioridade — fidelidade lexical ao corpus "
                f"ou qualidade semântica percebida."
            )

    elif auto_dados:
        partes.append("Apenas métricas automáticas disponíveis. A avaliação ACCR não foi executada.")
    elif accr_dados:
        partes.append("Apenas avaliação ACCR disponível. As métricas automáticas não foram executadas.")

    L("### Análise")
    L()
    for paragrafo in partes:
        L(paragrafo)
        L()

    # ── Salvar ────────────────────────────────────────────────────
    texto = "\n".join(linhas)
    print("\n" + texto)

    arquivo_saida = _OUTPUT / "relatorio_comparativo.md"
    tmp = arquivo_saida.with_suffix(".md.tmp")
    tmp.write_text(texto, encoding="utf-8")
    tmp.replace(arquivo_saida)
    print(f"\n📄 Relatório salvo em: {arquivo_saida}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: geração + ACCR + métricas automáticas."
    )

    # ── Controle de etapas ────────────────────────────────────────
    parser.add_argument(
        "--skip-gen", action="store_true",
        help="Pula a etapa de geração (usa predictions.json existentes)",
    )
    parser.add_argument(
        "--retry-nulls", action="store_true",
        help="Re-processa somente vídeos com caption null no arquivo de saída existente",
    )
    parser.add_argument(
        "--skip-accr", action="store_true",
        help="Pula a avaliação ACCR (economiza tokens do GitHub Models)",
    )
    parser.add_argument(
        "--skip-auto", action="store_true",
        help="Pula as métricas automáticas",
    )

    # ── Geração ───────────────────────────────────────────────────
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Número máximo de vídeos a gerar legendas (padrão: todos)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Segmentos em paralelo na geração (padrão: 2)",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["github_gpt41", "github_llama", "github_phi"],
        help="Modelo principal de geração",
    )
    parser.add_argument(
        "--provider2", type=str, default=None,
        choices=["github_gpt41", "github_llama", "github_phi"],
        help="Segundo modelo de geração",
    )
    parser.add_argument(
        "--pred1", type=str, default="predictions_gpt.json",
        help="Nome do arquivo de saída do modelo 1 em output/ (padrão: predictions_gpt.json)",
    )
    parser.add_argument(
        "--pred2", type=str, default=None,
        help="Nome do arquivo de saída do modelo 2 em output/ (padrão: predictions_<provider2>.json)",
    )

    # ── Rótulos para relatórios ───────────────────────────────────
    parser.add_argument(
        "--nome1", type=str, default="Modelo1",
        help="Rótulo do modelo 1 nos relatórios (padrão: Modelo1)",
    )
    parser.add_argument(
        "--nome2", type=str, default="Modelo2",
        help="Rótulo do modelo 2 nos relatórios (padrão: Modelo2)",
    )

    # ── Dados ─────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Pula a geração de gráficos (passo 4)",
    )
    parser.add_argument(
        "--no-skimcap", action="store_true",
        help="Exclui SkimCap da avaliação",
    )

    args = parser.parse_args()

    for d in (_OUTPUT, _PRED, _AUTO, _ACCR, _PLOTS):
        d.mkdir(parents=True, exist_ok=True)

    pred1_path = str(_PRED / args.pred1)
    _provider2 = args.provider2 or _cfg.PROVIDER_2
    pred2_path = str(_PRED / (args.pred2 or f"predictions_{_provider2.replace('github_', '')}.json"))

    print("\n" + "═" * 60)
    print("  PIPELINE DE VIDEO CAPTIONING — INÍCIO")
    print("═" * 60)
    print(f"  Modelo 1    : {args.nome1} → {args.pred1}")
    print(f"  Modelo 2    : {args.nome2} → {Path(pred2_path).name}")
    print(f"  SkimCap     : {'não' if args.no_skimcap else 'sim'}")
    print(f"  Etapas      : {'geração ' if not args.skip_gen else ''}{'ACCR ' if not args.skip_accr else ''}{'auto-métricas ' if not args.skip_auto else ''}{'plots' if not args.skip_plots else ''}")
    print("═" * 60)

    ok = True

    # ══ PASSO 1: GERAÇÃO ══════════════════════════════════════════
    if not args.skip_gen:
        _passo(1, "GERAÇÃO DE LEGENDAS (dois modelos)")

        cmd = [
            _PYTHON, str(_SRC / "agent" / "main.py"),
            "--output", f"predictions/{args.pred1}",
            "--workers", str(args.workers),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.provider:
            cmd += ["--provider", args.provider]
        if args.provider2:
            cmd += ["--provider2", args.provider2]
        if args.pred2:
            cmd += ["--output2", f"predictions/{args.pred2}"]
        if args.retry_nulls:
            cmd += ["--retry-nulls"]

        ok = _rodar(cmd, f"Gerando com {args.nome1} + {args.nome2}")
        if not ok:
            sys.exit(1)
    else:
        _passo(1, "GERAÇÃO — pulada (--skip-gen)")
        for p, nome in [(pred1_path, args.nome1), (pred2_path, args.nome2)]:
            if Path(p).exists():
                print(f"  ✓ {nome}: {p}")
            else:
                print(f"  ⚠ {nome}: {p} NÃO encontrado — avaliação pode falhar")

    # ══ PASSO 2: AVALIAÇÃO ACCR ═══════════════════════════════════
    if not args.skip_accr:
        _passo(2, "AVALIAÇÃO ACCR (LLM como avaliador)")

        cmd = [
            _PYTHON, str(_SRC / "evaluation" / "llm_eval.py"),
            "--predictions", pred1_path,
            "--predictions2", pred2_path,
            "--gt", _GT_1, _GT_2,
            "--modelo-nome", args.nome1,
            "--modelo2-nome", args.nome2,
            "--output-dir", str(_ACCR),
        ]
        if not args.no_skimcap:
            cmd += ["--skimcap", _SKIMCAP]

        ok = _rodar(cmd, "Avaliação ACCR")
        if not ok:
            print("  ⚠ ACCR falhou — continuando para métricas automáticas")
    else:
        _passo(2, "AVALIAÇÃO ACCR — pulada (--skip-accr)")

    # ══ PASSO 3: MÉTRICAS AUTOMÁTICAS ════════════════════════════
    if not args.skip_auto:
        _passo(3, "MÉTRICAS AUTOMÁTICAS (BLEU · METEOR · ROUGE-L · CIDEr · R@4)")

        cmd = [
            _PYTHON, str(_SRC / "evaluation" / "auto_metrics.py"),
            "--predictions", pred1_path,
            "--predictions2", pred2_path,
            "--references", _GT_1, _GT_2,
            "--modelo-nome", args.nome1,
            "--modelo2-nome", args.nome2,
            "--output", str(_AUTO),
        ]
        if not args.no_skimcap:
            cmd += ["--skimcap", _SKIMCAP]

        ok = _rodar(cmd, "Métricas automáticas")
    else:
        _passo(3, "MÉTRICAS AUTOMÁTICAS — pulada (--skip-auto)")

    # ══ PASSO 4: GRÁFICOS ════════════════════════════════════════
    if not args.skip_plots:
        _passo(4, "GRÁFICOS COMPARATIVOS")
        cmd = [
            _PYTHON, str(_ROOT / "scripts" / "gerar_graficos.py"),
            "--auto-dir",   str(_AUTO),
            "--accr-dir",   str(_ACCR),
            "--output-dir", str(_PLOTS),
        ]
        ok = _rodar(cmd, "Gerando gráficos")
        if not ok:
            print("  ⚠ Geração de gráficos falhou — verifique se matplotlib e scipy estão instalados")
    else:
        _passo(4, "GRÁFICOS — pulados (--skip-plots)")

    # ══ RESUMO FINAL ══════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PIPELINE CONCLUÍDO")
    print("═" * 60)
    print(f"  Outputs em: {_OUTPUT}/")
    print()
    for f in sorted(_OUTPUT.rglob("*.json")):
        print(f"    {f.relative_to(_OUTPUT)}")
    print()

    _tabela_comparativa(args, pred1_path, pred2_path)


if __name__ == "__main__":
    main()
