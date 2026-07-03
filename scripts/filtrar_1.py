import json

VIDEO = "v_eS1r2Qi0qUM"          # o vídeo que você quer
sel = {VIDEO, VIDEO[2:]}         # aceita com e sem "v_"

def filtrar_gt_skimcap(entrada, saida):
    """GT {video_id: {'sentences':[...]}} → {video_id: 'parágrafo'} (string)."""
    data = json.load(open(entrada, encoding="utf-8"))
    novo = {}
    for k, v in data.items():
        if k not in sel:
            continue
        sentencas = v.get("sentences", []) if isinstance(v, dict) else v
        novo[k] = " ".join(s.strip() for s in sentencas if s and s.strip())
    json.dump(novo, open(saida, "w", encoding="utf-8"))
    print(f"  {saida}: {len(novo)} vídeo(s)")

def filtrar_pred(entrada, saida):
    """SkimCap {'results': {video_id:[...]}} → mantém só o vídeo."""
    data = json.load(open(entrada, encoding="utf-8"))
    res = data.get("results", data)
    novo = {k: v for k, v in res.items() if k in sel}
    json.dump({"results": novo}, open(saida, "w", encoding="utf-8"))
    print(f"  {saida}: {len(novo)} vídeo(s)")

# Ajuste os caminhos para os seus arquivos originais:
filtrar_gt_skimcap("/Users/4016dtidigital/Documents/GitHub/tcc/data/ground_truth/anet_entities_test_1.json", "gt1_1.json")
filtrar_gt_skimcap("/Users/4016dtidigital/Documents/GitHub/tcc/data/ground_truth/anet_entities_test_2.json", "gt2_1.json")
filtrar_pred("/Users/4016dtidigital/Documents/GitHub/tcc/data/baselines/greedy_pred_test.json", "skimcap_1.json")