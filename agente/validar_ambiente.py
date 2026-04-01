"""
Script de validação do ambiente
Verifica se tudo está configurado corretamente antes de começar
"""
import os
import sys
import subprocess


def verificar_python():
    """Verifica versão do Python"""
    versao = sys.version_info
    print(f"Python: {versao.major}.{versao.minor}.{versao.micro}", end=" ")
    
    if versao.major >= 3 and versao.minor >= 8:
        print("✓")
        return True
    else:
        print("✗ (necessário Python 3.8+)")
        return False


def verificar_ffmpeg():
    """Verifica se FFmpeg está instalado"""
    print("FFmpeg: ", end="")
    try:
        resultado = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if resultado.returncode == 0:
            versao = resultado.stdout.split("\n")[0]
            print(f"{versao.split()[2]} ✓")
            return True
        else:
            print("✗ (não encontrado)")
            return False
    except FileNotFoundError:
        print("✗ (não instalado)")
        print("\n  Instale com:")
        print("    macOS:   brew install ffmpeg")
        print("    Linux:   sudo apt install ffmpeg")
        print("    Windows: https://ffmpeg.org/download.html")
        return False


def verificar_dependencias():
    """Verifica se as dependências Python estão instaladas"""
    dependencias = {
        "yt_dlp": "yt-dlp",
        "openai": "openai",
        "dotenv": "python-dotenv"
    }
    
    todas_ok = True
    
    for modulo, nome_pip in dependencias.items():
        print(f"{nome_pip}: ", end="")
        try:
            __import__(modulo)
            print("✓")
        except ImportError:
            print("✗ (não instalado)")
            todas_ok = False
    
    if not todas_ok:
        print("\n  Instale as dependências com:")
        print("    pip install -r requirements.txt")
    
    return todas_ok


def verificar_api_key():
    """Verifica se a API key da OpenAI está configurada"""
    print("API Key OpenAI: ", end="")
    
    # Verifica se o arquivo .env existe
    if not os.path.exists(".env"):
        print("✗ (arquivo .env não encontrado)")
        print("\n  Crie o arquivo .env:")
        print("    cp .env.example .env")
        print("    # Edite .env e adicione sua API key")
        return False
    
    # Tenta carregar a API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("✗ (não definida no .env)")
            print("\n  Edite o arquivo .env e adicione:")
            print("    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx")
            return False
        
        if api_key == "your_openai_api_key_here":
            print("✗ (usando valor padrão, não configurada)")
            print("\n  Edite o arquivo .env com sua API key real")
            return False
        
        if not api_key.startswith("sk-"):
            print("✗ (formato inválido)")
            print("\n  A API key deve começar com 'sk-'")
            return False
        
        print(f"{api_key[:8]}...{api_key[-4:]} ✓")
        return True
    
    except Exception as e:
        print(f"✗ (erro: {e})")
        return False


def verificar_pastas():
    """Verifica se as pastas necessárias podem ser criadas"""
    print("Permissões de escrita: ", end="")
    
    try:
        # Tenta criar as pastas temporárias
        pastas = ["videos", "frames", "output"]
        for pasta in pastas:
            os.makedirs(pasta, exist_ok=True)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ (erro: {e})")
        return False


def testar_conexao_youtube():
    """Testa conexão com o YouTube"""
    print("Conexão YouTube: ", end="")
    
    try:
        import yt_dlp
        
        # Testa com um vídeo curto e público
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "worst"
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Vídeo de teste público
            info = ydl.extract_info("https://www.youtube.com/watch?v=jNQXAC9IVRw", download=False)
            print("✓")
            return True
    
    except Exception as e:
        print(f"✗ (erro: {e})")
        print("\n  Verifique sua conexão com a internet")
        return False


def testar_api_openai():
    """Testa a conexão com a API da OpenAI"""
    print("API OpenAI: ", end="")
    
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Faz uma chamada simples para testar
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        print("✓")
        return True
    
    except Exception as e:
        erro_str = str(e)
        if "api_key" in erro_str.lower():
            print("✗ (API key inválida)")
        elif "quota" in erro_str.lower():
            print("✗ (sem créditos)")
        elif "rate_limit" in erro_str.lower():
            print("⚠️  (limite de requisições, mas key válida)")
            return True
        else:
            print(f"✗ ({erro_str[:50]}...)")
        return False


def main():
    print("="*60)
    print("VALIDAÇÃO DO AMBIENTE")
    print("="*60)
    print("\nVerificando requisitos...\n")
    
    resultados = {
        "Python 3.8+": verificar_python(),
        "FFmpeg": verificar_ffmpeg(),
        "Dependências Python": verificar_dependencias(),
        "API Key OpenAI": verificar_api_key(),
        "Permissões de escrita": verificar_pastas()
    }
    
    print("\n" + "="*60)
    
    # Se tudo estiver ok até aqui, testa conexões
    if all(resultados.values()):
        print("Testando conexões...\n")
        
        resultados["YouTube"] = testar_conexao_youtube()
        resultados["OpenAI API"] = testar_api_openai()
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    
    passou = sum(resultados.values())
    total = len(resultados)
    
    for item, ok in resultados.items():
        status = "✓" if ok else "✗"
        print(f"{status} {item}")
    
    print(f"\n{passou}/{total} verificações passaram")
    
    if passou == total:
        print("\n✨ Ambiente configurado corretamente!")
        print("\nVocê pode começar a usar o agente:")
        print("  python main.py")
        print("  python processar_activitynet.py")
    else:
        print("\n⚠️  Corrija os itens marcados com ✗ antes de continuar")
    
    print("="*60)


if __name__ == "__main__":
    main()
