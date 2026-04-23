import json
import yt_dlp
import os
import random

# 1. Carregar os IDs do AE-TEST
with open("anet_entities_test_1.json") as f:
 data = json.load(f)

video_ids = list(data.keys())  # formato: "v_XXXXXXXXXXX"
random.shuffle(video_ids)
print(f"Total de vídeos no AE-TEST: {len(video_ids)}")

# 2. Verificar disponibilidade e baixar
os.makedirs("videos", exist_ok=True)

disponiveis = []
indisponiveis = []

ydl_opts = {
 'quiet': True,
 'format': 'mp4',
 'outtmpl': 'videos/%(id)s.%(ext)s',
 'ignoreerrors': True,
}

for video_key in video_ids:  # percorre até encontrar 10 disponíveis
 if len(disponiveis) >= 10:
     break
 vid_id = video_key[2:]  # remove o "v_"
 url = f"https://www.youtube.com/watch?v={vid_id}"
 
 try:
     with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
         info = ydl.extract_info(url, download=False)
         if info:
             duracao = info.get("duration", 0)
             if 50 < duracao < 120:
                 disponiveis.append(video_key)
             else:
                 indisponiveis.append(video_key)
 except:
     indisponiveis.append(video_key)

print(f"Disponíveis: {len(disponiveis)}")
print(f"Indisponíveis: {len(indisponiveis)}")

# 3. Salvar lista dos disponíveis
with open("videos_disponiveis.json", "w") as f:
 json.dump(disponiveis, f)