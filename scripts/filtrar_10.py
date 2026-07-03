import json

# ── Caminho do seu JSON com os 10 vídeos selecionados ──
ARQUIVO_VIDEOS = "data/videos_com_urls.json"   # ajuste o nome/caminho

# Lê os IDs direto do seu JSON
_sel_raw = json.load(open(ARQUIVO_VIDEOS, encoding="utf-8"))
VIDEOS = [v["video_id"] for v in _sel_raw]    # ex: "v_bXdq2zI1Ms0"

# Aceita as duas formas (com e sem "v_") por segurança
sel = set(VIDEOS) | {v[2:] for v in VIDEOS if v.startswith("v_")}
print(f"IDs selecionados: {len(VIDEOS)}")

def filtrar_gt_skimcap(entrada, saida):
    """
    Converte GT {video_id: {'sentences': [...]}} para
    {video_id: 'parágrafo concatenado'} — formato que o para-evaluate.py espera.
    """
    data = json.load(open(entrada, encoding="utf-8"))
    novo = {}
    for k, v in data.items():
        if k not in sel:
            continue
        # extrai as sentenças (lida com dict OU lista)
        sentencas = v.get("sentences", []) if isinstance(v, dict) else v
        paragrafo = " ".join(s.strip() for s in sentencas if s and s.strip())
        novo[k] = paragrafo          # ← agora é STRING, não dict
    json.dump(novo, open(saida, "w", encoding="utf-8"))
    print(f"  {saida}: {len(novo)} vídeos")

def filtrar_pred(entrada, saida):
    """Predição SkimCap: {'results': {video_id: [...]}} — mantém só os 10."""
    data = json.load(open(entrada, encoding="utf-8"))
    res = data.get("results", data)
    novo = {k: v for k, v in res.items() if k in sel}
    json.dump({"results": novo}, open(saida, "w", encoding="utf-8"))
    print(f"  {saida}: {len(novo)} vídeos")

# ── Ajuste os caminhos conforme sua estrutura ──
print("Filtrando ground truths:")
filtrar_gt_skimcap("data/ground_truth/anet_entities_test_1.json", "gt1_10.json")
filtrar_gt_skimcap("data/ground_truth/anet_entities_test_2.json", "gt2_10.json")

print("Filtrando predição SkimCap:")
filtrar_pred("data/baselines/greedy_pred_test.json", "skimcap_10.json")