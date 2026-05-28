"""
============================================================
SYNC AUTOMÁTICO: Google Drive → Colab
Pipeline de Análise Industrial — África e Médio Oriente
============================================================

Script para copiar automaticamente TODOS os ficheiros do pipeline
do Google Drive para o Colab, sem necessidade de IDs hardcoded.

Uso no Colab:
    1. Montar o Drive:
        from google.colab import drive
        drive.mount('/content/drive')
    
    2. Executar este script:
        exec(open('/content/drive/MyDrive/analise_industrial_africa_mo/colab_auto_sync_pipeline.py').read())

Ou simplesmente:
    import shutil
    exec(open('/content/drive/MyDrive/analise_industrial_africa_mo/colab_auto_sync_pipeline.py').read())
"""

import os
import shutil
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Directório raiz do pipeline no Drive
DRIVE_PIPELINE_DIR = '/content/drive/MyDrive/analise_industrial_africa_mo/'

# Directório raiz do pipeline no Colab
COLAB_PIPELINE_DIR = '/content/repo/pipeline_africa_mo/'

# Passos e seus directórios de saída
PASSOS_DIRS = {
    'Passo 1': 'dados_brutos',
    'Passo 2': 'dados_limpos',
    'Passo 2.1': 'dados_limpos',  # Mesmo directório
    'Passo 2.2': ['dados_agregados', 'dados_sinteticos'],
    'Passo 2.3': 'dados_sinteticos',  # Mesmo directório
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


def sincronizar_ficheiros():
    """Sincroniza ficheiros do Drive para o Colab."""
    
    if not os.path.exists(DRIVE_PIPELINE_DIR):
        print(f"✗ ERRO: Directório do Drive não encontrado: {DRIVE_PIPELINE_DIR}")
        print(f"  Certifique-se de que:")
        print(f"    1. O Drive está montado: from google.colab import drive; drive.mount('/content/drive')")
        print(f"    2. O directório existe no Drive: {DRIVE_PIPELINE_DIR}")
        return False
    
    ficheiros_copiados = 0
    ficheiros_ignorados = 0
    ficheiros_erro = 0
    
    print(f"\nSincronizando ficheiros...")
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
            
            # Calcular caminho de destino (manter estrutura)
            relative_path = os.path.relpath(source_path, DRIVE_PIPELINE_DIR)
            dest_path = os.path.join(COLAB_PIPELINE_DIR, relative_path)
            
            # Criar directório de destino se não existir
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
    print(f"\n{'='*60}")
    print(f"RESUMO DA SINCRONIZAÇÃO")
    print(f"{'='*60}")
    print(f"  Ficheiros copiados: {ficheiros_copiados}")
    print(f"  Ficheiros ignorados (extensão não suportada): {ficheiros_ignorados}")
    print(f"  Ficheiros com erro: {ficheiros_erro}")
    print(f"{'='*60}\n")
    
    return ficheiros_copiados > 0


def listar_ficheiros_colab():
    """Lista os ficheiros agora disponíveis no Colab."""
    print(f"Ficheiros no Colab ({COLAB_PIPELINE_DIR}):\n")
    
    for passo, dirs in PASSOS_DIRS.items():
        if isinstance(dirs, list):
            for d in dirs:
                path = os.path.join(COLAB_PIPELINE_DIR, d)
                listar_dir(path, passo, d)
        else:
            path = os.path.join(COLAB_PIPELINE_DIR, dirs)
            listar_dir(path, passo, dirs)


def listar_dir(path, passo, subdir):
    """Lista ficheiros num directório específico."""
    if not os.path.exists(path):
        return
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if files:
        print(f"  {passo} ({subdir}/):")
        for f in sorted(files)[:10]:  # Mostrar máximo 10
            size_kb = os.path.getsize(os.path.join(path, f)) / 1024
            print(f"    - {f} ({size_kb:.1f} KB)")
        if len(files) > 10:
            print(f"    ... e mais {len(files) - 10} ficheiros")
        print()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SINCRONIZAÇÃO AUTOMÁTICA: GOOGLE DRIVE → COLAB")
    print("="*60 + "\n")
    
    # Passo 1: Criar directórios
    criar_diretorios()
    
    # Passo 2: Sincronizar ficheiros
    sucesso = sincronizar_ficheiros()
    
    # Passo 3: Listar ficheiros (se sucesso)
    if sucesso:
        listar_ficheiros_colab()
        print("✓ SINCRONIZAÇÃO CONCLUÍDA COM SUCESSO!\n")
    else:
        print("✗ SINCRONIZAÇÃO FALHOU. Verifique os erros acima.\n")
