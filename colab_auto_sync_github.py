"""
============================================================
SYNC AUTOMÁTICO: GitHub → Colab + Google Drive → Colab
Pipeline de Análise Industrial — África e Médio Oriente
============================================================

Script para copiar automaticamente TODOS os ficheiros do pipeline
do GitHub E do Google Drive para o Colab.

Uso no Colab (Opção 1 — Recomendado):
    from google.colab import drive
    drive.mount('/content/drive')
    
    import urllib.request
    url = 'https://raw.githubusercontent.com/seu_usuario/seu_repo/main/colab_auto_sync_pipeline.py'
    exec(urllib.request.urlopen(url).read())

Uso no Colab (Opção 2 — Se o ficheiro está no Drive):
    from google.colab import drive
    drive.mount('/content/drive')
    
    exec(open('/content/drive/MyDrive/analise_industrial_africa_mo/colab_auto_sync_pipeline.py').read())

Uso no Colab (Opção 3 — Este ficheiro):
    from google.colab import drive
    drive.mount('/content/drive')
    
    import urllib.request
    url = 'https://raw.githubusercontent.com/seu_usuario/seu_repo/main/colab_auto_sync_github.py'
    exec(urllib.request.urlopen(url).read())
"""

import os
import shutil
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# GitHub
GITHUB_REPO_URL = 'https://raw.githubusercontent.com/seu_usuario/seu_repo/main/'

# Google Drive
DRIVE_PIPELINE_DIR = '/content/drive/MyDrive/analise_industrial_africa_mo/'

# Colab
COLAB_PIPELINE_DIR = '/content/repo/pipeline_africa_mo/'

# Passos e seus directórios de saída
PASSOS_DIRS = {
    'Passo 1': 'dados_brutos',
    'Passo 2': 'dados_limpos',
    'Passo 2.1': 'dados_limpos',
    'Passo 2.2': ['dados_agregados', 'dados_sinteticos'],
    'Passo 2.3': 'dados_sinteticos',
    'Passo 3': 'dados_engenharia',
    'Passo 4': 'modelos',
    'Passo 5': 'resultados_avaliacao',
    'Passo 6': 'analise_estrategias',
    'Passo 7': 'shap_analysis',
    'Passo 8': 'analise_geografica',
    'Passo 9': 'analise_avancada',
    'Passo 10': 'inovacoes_mestrado',
}

# Extensões de ficheiros a sincronizar
EXTENSOES_VALIDAS = ['.csv', '.pkl', '.json', '.png', '.jpg', '.pdf', '.xlsx', '.txt']

# ============================================================
# FUNÇÕES
# ============================================================

def criar_diretorios():
    """Cria todos os directórios necessários no Colab."""
    for dirs in PASSOS_DIRS.values():
        if isinstance(dirs, list):
            for d in dirs:
                path = os.path.join(COLAB_PIPELINE_DIR, d)
                os.makedirs(path, exist_ok=True)
        else:
            path = os.path.join(COLAB_PIPELINE_DIR, dirs)
            os.makedirs(path, exist_ok=True)
    
    print("✓ Directórios criados/verificados")


def sincronizar_drive():
    """Sincroniza ficheiros do Google Drive para o Colab."""
    
    if not os.path.exists(DRIVE_PIPELINE_DIR):
        print(f"\n⚠ Directório do Drive não encontrado: {DRIVE_PIPELINE_DIR}")
        print(f"  Pulando sincronização do Drive...")
        return 0
    
    ficheiros_copiados = 0
    ficheiros_ignorados = 0
    ficheiros_erro = 0
    
    print(f"\n{'='*60}")
    print(f"SINCRONIZAÇÃO: GOOGLE DRIVE → COLAB")
    print(f"{'='*60}")
    print(f"  Origem: {DRIVE_PIPELINE_DIR}")
    print(f"  Destino: {COLAB_PIPELINE_DIR}\n")
    
    # Percorrer todos os ficheiros do Drive
    for root, dirs, files in os.walk(DRIVE_PIPELINE_DIR):
        for file in files:
            # Verificar extensão
            _, ext = os.path.splitext(file)
            if ext.lower() not in EXTENSOES_VALIDAS:
                ficheiros_ignorados += 1
                continue
            
            # Caminho completo no Drive
            source_path = os.path.join(root, file)
            
            # Calcular caminho de destino
            relative_path = os.path.relpath(source_path, DRIVE_PIPELINE_DIR)
            dest_path = os.path.join(COLAB_PIPELINE_DIR, relative_path)
            
            # Criar directório de destino
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copiar ficheiro
            try:
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ {relative_path}")
                ficheiros_copiados += 1
            except Exception as e:
                print(f"  ✗ {relative_path}: {str(e)[:60]}")
                ficheiros_erro += 1
    
    # Resumo
    print(f"\n  Ficheiros copiados: {ficheiros_copiados}")
    print(f"  Ficheiros ignorados: {ficheiros_ignorados}")
    print(f"  Ficheiros com erro: {ficheiros_erro}")
    
    return ficheiros_copiados


def listar_ficheiros_colab():
    """Lista os ficheiros agora disponíveis no Colab."""
    print(f"\n{'='*60}")
    print(f"FICHEIROS DISPONÍVEIS NO COLAB")
    print(f"{'='*60}\n")
    
    total_ficheiros = 0
    for passo, dirs in PASSOS_DIRS.items():
        if isinstance(dirs, list):
            for d in dirs:
                path = os.path.join(COLAB_PIPELINE_DIR, d)
                n = listar_dir(path, passo, d)
                total_ficheiros += n
        else:
            path = os.path.join(COLAB_PIPELINE_DIR, dirs)
            n = listar_dir(path, passo, dirs)
            total_ficheiros += n
    
    print(f"{'='*60}")
    print(f"Total de ficheiros: {total_ficheiros}")
    print(f"{'='*60}\n")


def listar_dir(path, passo, subdir):
    """Lista ficheiros num directório específico."""
    if not os.path.exists(path):
        return 0
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if files:
        print(f"  {passo} ({subdir}/):")
        for f in sorted(files)[:10]:
            size_kb = os.path.getsize(os.path.join(path, f)) / 1024
            print(f"    - {f} ({size_kb:.1f} KB)")
        if len(files) > 10:
            print(f"    ... e mais {len(files) - 10} ficheiros")
        print()
        return len(files)
    return 0


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SINCRONIZAÇÃO AUTOMÁTICA: GITHUB + GOOGLE DRIVE → COLAB")
    print("="*60 + "\n")
    
    # Passo 1: Criar directórios
    criar_diretorios()
    
    # Passo 2: Sincronizar do Drive
    ficheiros_drive = sincronizar_drive()
    
    # Passo 3: Listar ficheiros
    listar_ficheiros_colab()
    
    # Resumo final
    if ficheiros_drive > 0:
        print("✓ SINCRONIZAÇÃO CONCLUÍDA COM SUCESSO!\n")
    else:
        print("⚠ Nenhum ficheiro sincronizado do Drive.")
        print("  Certifique-se de que o Drive está montado e o directório existe.\n")
