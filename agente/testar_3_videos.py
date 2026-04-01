"""
Script de teste para processar os 3 primeiros vídeos
"""

import json
import sys
from main import VideoCaptioningAgent

# Dados dos 3 primeiros vídeos
VIDEOS = [
    {
        "video_id": "v_bXdq2zI1Ms0",
        "url": "https://www.youtube.com/watch?v=bXdq2zI1Ms0",
        "timestamps": [[0, 10.23], [10.6, 39.84], [38.01, 73.1]]
    },
    {
        "video_id": "v_FsS_NCZEfaI",
        "url": "https://www.youtube.com/watch?v=FsS_NCZEfaI",
        "timestamps": [[0, 91.48], [73.4, 174.45], [168.07, 212.74]]
    },
    {
        "video_id": "v_HWV_ccmZVPA",
        "url": "https://www.youtube.com/watch?v=HWV_ccmZVPA",
        "timestamps": [[0, 7.04], [8.3, 24.15], [26.17, 26.92], [35.73, 50.32]]
    }
]

def main():
    print("=" * 70)
    print("TESTE: Processando 3 primeiros vídeos")
    print("=" * 70)
    
    # Inicializa o agente
    agente = VideoCaptioningAgent()
    
    todos_resultados = []
    
    # Processa cada vídeo
    for idx, video_data in enumerate(VIDEOS, 1):
        print(f"\n{'='*70}")
        print(f"VÍDEO {idx}/3: {video_data['video_id']}")
        print(f"{'='*70}")
        
        resultado = agente.processar_video(
            url=video_data['url'],
            segmentos=video_data['timestamps'],
            video_id=video_data['video_id']
        )
        
        if resultado:
            todos_resultados.append(resultado)
            # Salva resultado individual
            agente.salvar_resultados(
                resultado, 
                f"{video_data['video_id']}_captions.json"
            )
        else:
            print(f"✗ Falha ao processar {video_data['video_id']}")
    
    # Salva todos os resultados juntos
    print(f"\n{'='*70}")
    print("SALVANDO RESULTADOS CONSOLIDADOS")
    print(f"{'='*70}")
    
    agente.salvar_resultados(
        {
            "test_name": "3_primeiros_videos",
            "total_videos": len(VIDEOS),
            "processed_videos": len(todos_resultados),
            "results": todos_resultados
        },
        "teste_3_videos_completo.json"
    )
    
    # Resumo
    print(f"\n{'='*70}")
    print("RESUMO")
    print(f"{'='*70}")
    print(f"Vídeos processados: {len(todos_resultados)}/{len(VIDEOS)}")
    
    total_segments = sum(r['num_segments'] for r in todos_resultados)
    print(f"Total de segmentos: {total_segments}")
    
    # Conta legendas geradas com sucesso
    legendas_ok = sum(
        1 for r in todos_resultados 
        for seg in r['results'] 
        if seg.get('caption')
    )
    print(f"Legendas geradas: {legendas_ok}/{total_segments}")
    
    print(f"\n{'='*70}")
    print("✓ TESTE CONCLUÍDO!")
    print(f"{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Teste interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
