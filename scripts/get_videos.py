import json
import yt_dlp
import os

# 1. Carregar os IDs do AE-TEST
with open("anet_entities_test_1.json") as f:
 data = json.load(f)

video_ids = list(data.keys())  # formato: "v_XXXXXXXXXXX"
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

for video_key in video_ids:  # todos os vídeos
 vid_id = video_key[2:]  # remove o "v_"
 url = f"https://www.youtube.com/watch?v={vid_id}"
 
 try:
     with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
         info = ydl.extract_info(url, download=False)
         if info:
             disponiveis.append(video_key)
 except:
     indisponiveis.append(video_key)

print(f"Disponíveis: {len(disponiveis)}")
print(f"Indisponíveis: {len(indisponiveis)}")

# 3. Salvar lista dos disponíveis
with open("videos_disponiveis.json", "w") as f:
 json.dump(disponiveis, f)