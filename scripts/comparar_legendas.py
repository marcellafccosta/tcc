import json
from pathlib import Path


VIDEOS_FILE = Path("data/videos_com_urls.json")

SKIMCAP_FILE = Path("data/baselines/greedy_pred_test_full.json")
GPT_FILE = Path("output/predictions/predictions_gpt_full.json")
LLAMA_FILE = Path("output/predictions/predictions_llama_full.json")

OUTPUT_FILE = Path("output/predictions/comparacao_10_videos.json")


ORDER_IDS = [
    "eS1r2Qi0qUM",
    "UXi0Cy16-0Y",
    "VtIMPJjcdn4",
    "rMZtiiLAqoY",
    "bvnXdr-Hre4",
    "j4Ru2L4u0Qk",
    "7ZbH4vHTmVs",
    "rmoa-Ffel2k",
    "bXdq2zI1Ms0",
    "-UwqKYkkKlU"
]


CAPTION_KEYS = [
    "sentence",
    "caption",
    "generated_caption",
    "prediction",
    "pred",
    "generated_text",
    "output",
    "text",
    "description",
    "response",
    "answer",
    "content",
    "result"
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_video_id(video_id):
    """
    Garante que o ID fique no formato com v_.

    Exemplo:
    eS1r2Qi0qUM -> v_eS1r2Qi0qUM
    v_eS1r2Qi0qUM -> v_eS1r2Qi0qUM
    """

    if not video_id:
        return ""

    video_id = str(video_id).strip()

    if video_id.startswith("v_"):
        return video_id

    return f"v_{video_id}"


def get_youtube_id(video_id):
    """
    Remove o prefixo v_ para gerar o YouTube_id.
    """

    if not video_id:
        return ""

    video_id = str(video_id).strip()

    if video_id.startswith("v_"):
        return video_id[2:]

    return video_id


def extract_caption(item):
    """
    Procura a caption dentro de diferentes estruturas possíveis.
    Funciona para dict, list e string.
    """

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, list):
        for element in item:
            caption = extract_caption(element)
            if caption:
                return caption
        return ""

    if isinstance(item, dict):
        for key in CAPTION_KEYS:
            if key in item and item[key]:
                caption = extract_caption(item[key])
                if caption:
                    return caption

        for key, value in item.items():
            if key in ["video_id", "YouTube_id", "youtube_id", "id", "url"]:
                continue

            if isinstance(value, (dict, list)):
                caption = extract_caption(value)
                if caption:
                    return caption

    return ""


def normalize_predictions(data):
    """
    Padroniza os arquivos de predição para este formato:

    {
        "v_eS1r2Qi0qUM": "caption do vídeo",
        "v_UXi0Cy16-0Y": "caption do vídeo"
    }
    """

    if isinstance(data, dict):
        for key in ["results", "predictions", "data", "videos"]:
            if key in data:
                data = data[key]
                break

    normalized = {}

    if isinstance(data, dict):
        for video_id, value in data.items():
            normalized_id = normalize_video_id(video_id)
            caption = extract_caption(value)

            if normalized_id:
                normalized[normalized_id] = caption

    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            video_id = (
                item.get("video_id")
                or item.get("YouTube_id")
                or item.get("youtube_id")
                or item.get("id")
            )

            normalized_id = normalize_video_id(video_id)
            caption = extract_caption(item)

            if normalized_id:
                normalized[normalized_id] = caption

    return normalized


def normalize_videos_file(videos):
    """
    Cria um dicionário dos vídeos do videos_com_urls.json
    para conseguir buscar pela ordem definida em ORDER_IDS.
    """

    videos_by_id = {}

    for video in videos:
        video_id = (
            video.get("video_id")
            or video.get("YouTube_id")
            or video.get("youtube_id")
            or video.get("id")
        )

        normalized_id = normalize_video_id(video_id)

        if normalized_id:
            videos_by_id[normalized_id] = video

    return videos_by_id


def count_non_empty(predictions):
    return sum(1 for caption in predictions.values() if caption)


def main():
    videos = load_json(VIDEOS_FILE)

    videos_by_id = normalize_videos_file(videos)

    skimcap_data = normalize_predictions(load_json(SKIMCAP_FILE))
    gpt_data = normalize_predictions(load_json(GPT_FILE))
    llama_data = normalize_predictions(load_json(LLAMA_FILE))

    print("SkimCap IDs encontrados:", len(skimcap_data))
    print("GPT IDs encontrados:", len(gpt_data))
    print("LLaMA IDs encontrados:", len(llama_data))

    print("SkimCap com caption:", count_non_empty(skimcap_data))
    print("GPT com caption:", count_non_empty(gpt_data))
    print("LLaMA com caption:", count_non_empty(llama_data))

    resultado = []

    for youtube_id in ORDER_IDS:
        video_id = normalize_video_id(youtube_id)

        video_info = videos_by_id.get(video_id, {})

        url = video_info.get("url") or f"https://www.youtube.com/watch?v={youtube_id}"

        resultado.append({
            "video_id": video_id,
            "YouTube_id": youtube_id,
            "url": url,
            "skimcap": {
                "caption": skimcap_data.get(video_id, "")
            },
            "gpt": {
                "caption": gpt_data.get(video_id, "")
            },
            "llama": {
                "caption": llama_data.get(video_id, "")
            }
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print()
    print(f"Arquivo gerado em: {OUTPUT_FILE}")
    print(f"Total de vídeos processados: {len(resultado)}")

    missing_skimcap = [
        item["video_id"]
        for item in resultado
        if not item["skimcap"]["caption"]
    ]

    missing_gpt = [
        item["video_id"]
        for item in resultado
        if not item["gpt"]["caption"]
    ]

    missing_llama = [
        item["video_id"]
        for item in resultado
        if not item["llama"]["caption"]
    ]

    print()
    print(f"Vídeos sem caption SkimCap: {len(missing_skimcap)}")
    print(f"Vídeos sem caption GPT: {len(missing_gpt)}")
    print(f"Vídeos sem caption LLaMA: {len(missing_llama)}")

    if missing_skimcap:
        print("Sem SkimCap:", missing_skimcap)

    if missing_gpt:
        print("Sem GPT:", missing_gpt)

    if missing_llama:
        print("Sem LLaMA:", missing_llama)


if __name__ == "__main__":
    main()