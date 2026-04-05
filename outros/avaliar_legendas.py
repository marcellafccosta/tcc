"""
Script de avaliação de legendas geradas
Compara as legendas do agente GPT-4 com Ground Truth e SkimCap

Este script é SEPARADO do agente de geração.
Ele lê os JSONs gerados pelo agente e calcula métricas.
"""
import json
import os
from typing import List, Dict


def carregar_resultados_agente(arquivo):
    """Carrega resultados gerados pelo agente GPT-4"""
    with open(arquivo, "r") as f:
        return json.load(f)


def carregar_ground_truth(arquivo):
    """Carrega anotações ground truth do ActivityNet"""
    with open(arquivo, "r") as f:
        return json.load(f)


def carregar_skimcap(arquivo):
    """Carrega predições do SkimCap"""
    with open(arquivo, "r") as f:
        return json.load(f)


def preparar_para_comparacao(resultados_agente, ground_truth_data, skimcap_data=None):
    """
    Organiza os dados para comparação
    
    Returns:
        Dict com {
            'video_id': {
                'gpt4': [lista de captions],
                'ground_truth': [lista de referências],
                'skimcap': [lista de captions] (opcional)
            }
        }
    """
    dados_comparacao = {}
    
    # Processa cada vídeo dos resultados do agente
    for video in resultados_agente.get("videos", []):
        video_id = video["video_id"]
        
        # Captions gerados pelo GPT-4
        gpt_captions = [seg["caption"] for seg in video["segments"]]
        
        # Ground truth correspondente
        gt = ground_truth_data.get(video_id, {})
        gt_captions = gt.get("sentences", [])
        
        # SkimCap (se disponível)
        skimcap_captions = None
        if skimcap_data and video_id in skimcap_data:
            skimcap_captions = skimcap_data[video_id].get("results", [])
        
        dados_comparacao[video_id] = {
            "gpt4": gpt_captions,
            "ground_truth": gt_captions,
            "skimcap": skimcap_captions,
            "timestamps": [seg["timestamps"] for seg in video["segments"]]
        }
    
    return dados_comparacao


def calcular_metricas_simples(candidato: str, referencias: List[str]) -> Dict[str, float]:
    """
    Métricas simples: precision, recall, F1
    
    Para métricas mais avançadas (BLEU, METEOR, CIDEr):
    - pip install nltk
    - pip install pycocoevalcap
    """
    
    def tokenizar(texto):
        """Tokenização simples por espaços"""
        return set(texto.lower().split())
    
    candidato_tokens = tokenizar(candidato)
    
    if not candidato_tokens:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    # Calcula métricas contra cada referência e faz média
    precisions = []
    recalls = []
    
    for referencia in referencias:
        ref_tokens = tokenizar(referencia)
        
        # Tokens em comum
        comuns = candidato_tokens & ref_tokens
        
        # Precision: % dos tokens do candidato que estão na referência
        precision = len(comuns) / len(candidato_tokens) if candidato_tokens else 0
        
        # Recall: % dos tokens da referência que estão no candidato
        recall = len(comuns) / len(ref_tokens) if ref_tokens else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Média
    precision_avg = sum(precisions) / len(precisions) if precisions else 0
    recall_avg = sum(recalls) / len(recalls) if recalls else 0
    
    # F1
    f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0
    
    return {
        "precision": precision_avg,
        "recall": recall_avg,
        "f1": f1
    }


def avaliar_video(video_id, dados):
    """Avalia um vídeo específico"""
    gpt_captions = dados["gpt4"]
    gt_captions = dados["ground_truth"]
    skimcap_captions = dados.get("skimcap")
    timestamps = dados.get("timestamps", [])
    
    print(f"\n{'='*60}")
    print(f"Vídeo: {video_id}")
    print(f"{'='*60}")
    print(f"Segmentos GPT-4: {len(gpt_captions)}")
    print(f"Segmentos GT: {len(gt_captions)}")
    
    resultados_segmentos = []
    
    # Avalia cada segmento
    for i, gpt_caption in enumerate(gpt_captions):
        # Usa GT correspondente (ou todos se não houver correspondência)
        if i < len(gt_captions):
            referencias = [gt_captions[i]]
        else:
            referencias = gt_captions if gt_captions else [""]
        
        metricas = calcular_metricas_simples(gpt_caption, referencias)
        
        timestamp = timestamps[i] if i < len(timestamps) else [0, 0]
        
        print(f"\n  Segmento {i} [{timestamp[0]:.1f}-{timestamp[1]:.1f}s]:")
        print(f"    GPT-4: {gpt_caption[:80]}...")
        if referencias and referencias[0]:
            print(f"    GT:    {referencias[0][:80]}...")
        print(f"    F1: {metricas['f1']:.3f} | P: {metricas['precision']:.3f} | R: {metricas['recall']:.3f}")
        
        resultado_seg = {
            "segment": i,
            "timestamps": timestamp,
            "gpt4_caption": gpt_caption,
            "ground_truth": referencias,
            "metrics": metricas
        }
        
        # Adiciona SkimCap se disponível
        if skimcap_captions and i < len(skimcap_captions):
            resultado_seg["skimcap_caption"] = skimcap_captions[i]
        
        resultados_segmentos.append(resultado_seg)
    
    return resultados_segmentos


def gerar_relatorio(todos_resultados, arquivo_saida):
    """
    Gera relatório consolidado de avaliação
    """
    # Coleta todas as métricas
    todas_metricas = []
    num_videos = len(todos_resultados)
    num_segmentos = 0
    
    for video_id, segmentos in todos_resultados.items():
        num_segmentos += len(segmentos)
        for seg in segmentos:
            todas_metricas.append(seg["metrics"])
    
    # Calcula estatísticas agregadas
    f1_scores = [m["f1"] for m in todas_metricas]
    precision_scores = [m["precision"] for m in todas_metricas]
    recall_scores = [m["recall"] for m in todas_metricas]
    
    relatorio = {
        "dataset": "ActivityNet",
        "modelo": "GPT-4 Vision (3 frames)",
        "num_videos": num_videos,
        "num_segmentos": num_segmentos,
        "metricas_agregadas": {
            "f1": {
                "media": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "min": min(f1_scores) if f1_scores else 0,
                "max": max(f1_scores) if f1_scores else 0
            },
            "precision": {
                "media": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
                "min": min(precision_scores) if precision_scores else 0,
                "max": max(precision_scores) if precision_scores else 0
            },
            "recall": {
                "media": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
                "min": min(recall_scores) if recall_scores else 0,
                "max": max(recall_scores) if recall_scores else 0
            }
        },
        "resultados_por_video": todos_resultados
    }
    
    # Salva
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    
    # Imprime resumo
    print("\n" + "="*60)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("="*60)
    print(f"Dataset: {relatorio['dataset']}")
    print(f"Modelo: {relatorio['modelo']}")
    print(f"Vídeos avaliados: {num_videos}")
    print(f"Segmentos avaliados: {num_segmentos}")
    
    print(f"\nMÉTRICAS MÉDIAS:")
    print(f"  F1:        {relatorio['metricas_agregadas']['f1']['media']:.3f} "
          f"(min: {relatorio['metricas_agregadas']['f1']['min']:.3f}, "
          f"max: {relatorio['metricas_agregadas']['f1']['max']:.3f})")
    print(f"  Precision: {relatorio['metricas_agregadas']['precision']['media']:.3f}")
    print(f"  Recall:    {relatorio['metricas_agregadas']['recall']['media']:.3f}")
    
    print(f"\n✓ Relatório completo salvo em: {arquivo_saida}")
    
    return relatorio


def main():
    print("="*60)
    print("AVALIAÇÃO DE LEGENDAS DO AGENTE")
    print("="*60)
    
    # 1. Carrega resultados do agente
    print("\n1. Carregando resultados do agente...")
    arquivo_agente = input("   Arquivo JSON do agente (ex: agente/output/activitynet_batch_5.json): ")
    
    if not os.path.exists(arquivo_agente):
        print(f"   ✗ Arquivo não encontrado: {arquivo_agente}")
        return
    
    resultados_agente = carregar_resultados_agente(arquivo_agente)
    print(f"   ✓ {len(resultados_agente.get('videos', []))} vídeos carregados")
    
    # 2. Carrega Ground Truth
    print("\n2. Carregando Ground Truth...")
    arquivo_gt = input("   Arquivo JSON do GT (ex: descricoes/descricoes GT/anet_entities_test_1.json): ")
    
    if not os.path.exists(arquivo_gt):
        print(f"   ✗ Arquivo não encontrado: {arquivo_gt}")
        return
    
    ground_truth = carregar_ground_truth(arquivo_gt)
    print(f"   ✓ Ground Truth carregado")
    
    # 3. (Opcional) Carrega SkimCap
    print("\n3. Carregar SkimCap? (opcional)")
    carregar_sk = input("   Deseja comparar com SkimCap? (s/n): ")
    
    skimcap = None
    if carregar_sk.lower() == 's':
        arquivo_sk = input("   Arquivo JSON do SkimCap (ex: descricoes/descricoes skimcap/greedy_pred_test.json): ")
        if os.path.exists(arquivo_sk):
            skimcap = carregar_skimcap(arquivo_sk)
            print(f"   ✓ SkimCap carregado")
        else:
            print(f"   ⚠️  Arquivo não encontrado, continuando sem SkimCap")
    
    # 4. Prepara dados
    print("\n4. Preparando dados para comparação...")
    dados = preparar_para_comparacao(resultados_agente, ground_truth, skimcap)
    print(f"   ✓ {len(dados)} vídeos preparados")
    
    # 5. Avalia cada vídeo
    print("\n5. Avaliando vídeos...")
    todos_resultados = {}
    
    for video_id, dados_video in dados.items():
        resultados_video = avaliar_video(video_id, dados_video)
        todos_resultados[video_id] = resultados_video
    
    # 6. Gera relatório
    print("\n6. Gerando relatório...")
    arquivo_saida = f"avaliacao_{os.path.basename(arquivo_agente)}"
    gerar_relatorio(todos_resultados, arquivo_saida)
    
    print("\n✨ Avaliação concluída!")


def instalar_metricas_avancadas_info():
    """Informações sobre métricas avançadas"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       MÉTRICAS AVANÇADAS (BLEU, METEOR, CIDEr, SPICE)        ║
╚══════════════════════════════════════════════════════════════╝

Este script usa métricas simples (Precision, Recall, F1).

Para métricas mais avançadas usadas em papers de captioning:

1. Instale as bibliotecas:
   pip install nltk
   pip install pycocoevalcap

2. Use a biblioteca pycocoevalcap:
   from pycocoevalcap.bleu.bleu import Bleu
   from pycocoevalcap.meteor.meteor import Meteor
   from pycocoevalcap.cider.cider import Cider
   
3. Formato dos dados:
   gts = {'video_id': ['ref1', 'ref2'], ...}
   res = {'video_id': ['candidato'], ...}

4. Calcule:
   scorer = Bleu(4)
   score, scores = scorer.compute_score(gts, res)

Documentação: https://github.com/tylin/coco-caption

Para ACCR (ActivityNet Captions metric):
   Veja: https://github.com/ranjaykrishna/densevid_eval
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help-metrics":
        instalar_metricas_avancadas_info()
    else:
        main()
        
        print("\n💡 Para informações sobre métricas avançadas:")
        print("   python avaliar_legendas.py --help-metrics")
