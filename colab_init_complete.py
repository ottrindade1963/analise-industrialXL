"""
============================================================
INICIALIZAÇÃO COMPLETA DO PIPELINE NO COLAB
============================================================

Script para configurar correctamente todos os caminhos e executar
qualquer passo do pipeline no Colab.

Uso no Colab:

    # Célula 1: Montar o Drive e sincronizar ficheiros
    from google.colab import drive
    drive.mount('/content/drive')
    
    import urllib.request
    url = 'https://raw.githubusercontent.com/seu_usuario/seu_repo/main/colab_init_complete.py'
    exec(urllib.request.urlopen(url).read())
    
    # Célula 2: Executar um passo específico
    executar_passo6()  # ou qualquer outro passo

"""

import os
import sys

# ============================================================
# CONFIGURAÇÃO DE CAMINHOS
# ============================================================

# Directório raiz do pipeline no Colab
COLAB_REPO_DIR = '/content/repo/pipeline_africa_mo'

# Criar directórios se não existirem
DIRETORIOS = [
    'dados_brutos',
    'dados_limpos',
    'dados_agregados',      # ← IMPORTANTE: dados_agregados, não dados_engenharia
    'dados_sinteticos',
    'dados_engenharia',
    'modelos',
    'resultados_avaliacao',
    'analise_estrategias',
    'shap_analysis',
    'analise_geografica',
    'analise_avancada',
    'inovacoes_mestrado',
]

print("="*60)
print("INICIALIZAÇÃO DO PIPELINE NO COLAB")
print("="*60)

# Criar directórios
print("\n[1/3] Criando directórios...")
for d in DIRETORIOS:
    path = os.path.join(COLAB_REPO_DIR, d)
    os.makedirs(path, exist_ok=True)
print(f"✓ {len(DIRETORIOS)} directórios criados/verificados")

# Adicionar ao path do Python
print("\n[2/3] Configurando Python path...")
sys.path.insert(0, COLAB_REPO_DIR)
print(f"✓ {COLAB_REPO_DIR} adicionado ao sys.path")

# Importar e configurar config_global
print("\n[3/3] Configurando caminhos do pipeline...")

import config_global as config

# Sobrescrever caminhos para Colab
config.BASE_DIR = COLAB_REPO_DIR
config.DADOS_BRUTOS_DIR = os.path.join(COLAB_REPO_DIR, 'dados_brutos')
config.DADOS_LIMPOS_DIR = os.path.join(COLAB_REPO_DIR, 'dados_limpos')
config.DADOS_AGREGADOS_DIR = os.path.join(COLAB_REPO_DIR, 'dados_agregados')  # ← IMPORTANTE
config.DADOS_SINTETICOS_DIR = os.path.join(COLAB_REPO_DIR, 'dados_sinteticos')
config.DADOS_ENGENHARIA_DIR = os.path.join(COLAB_REPO_DIR, 'dados_engenharia')
config.MODELOS_DIR = os.path.join(COLAB_REPO_DIR, 'modelos')
config.RESULTADOS_DIR = os.path.join(COLAB_REPO_DIR, 'resultados_avaliacao')
config.ESTRATEGIAS_DIR = os.path.join(COLAB_REPO_DIR, 'analise_estrategias')
config.SHAP_DIR = os.path.join(COLAB_REPO_DIR, 'shap_analysis')
config.GEOGRAFICA_DIR = os.path.join(COLAB_REPO_DIR, 'analise_geografica')
config.AVANCADA_DIR = os.path.join(COLAB_REPO_DIR, 'analise_avancada')

# Passo 10
if not hasattr(config, 'INOVACOES_DIR'):
    config.INOVACOES_DIR = os.path.join(COLAB_REPO_DIR, 'inovacoes_mestrado')

print(f"✓ Caminhos configurados:")
print(f"  BASE_DIR: {config.BASE_DIR}")
print(f"  DADOS_AGREGADOS_DIR: {config.DADOS_AGREGADOS_DIR}")
print(f"  DADOS_ENGENHARIA_DIR: {config.DADOS_ENGENHARIA_DIR}")
print(f"  MODELOS_DIR: {config.MODELOS_DIR}")

# ============================================================
# FUNÇÕES DE EXECUÇÃO
# ============================================================

def verificar_ficheiros():
    """Verifica quais ficheiros estão disponíveis."""
    print("\n" + "="*60)
    print("FICHEIROS DISPONÍVEIS")
    print("="*60 + "\n")
    
    dirs_info = {
        'Dados Brutos': config.DADOS_BRUTOS_DIR,
        'Dados Limpos': config.DADOS_LIMPOS_DIR,
        'Dados Agregados': config.DADOS_AGREGADOS_DIR,
        'Dados Sintéticos': config.DADOS_SINTETICOS_DIR,
        'Engenharia de Features': config.DADOS_ENGENHARIA_DIR,
        'Modelos': config.MODELOS_DIR,
        'Resultados Avaliação': config.RESULTADOS_DIR,
    }
    
    for nome, path in dirs_info.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            if files:
                print(f"  {nome}/ ({len(files)} ficheiros)")
                for f in sorted(files)[:5]:
                    size_kb = os.path.getsize(os.path.join(path, f)) / 1024
                    print(f"    - {f} ({size_kb:.1f} KB)")
                if len(files) > 5:
                    print(f"    ... e mais {len(files) - 5}")
            else:
                print(f"  {nome}/ (vazio)")
        else:
            print(f"  {nome}/ (não existe)")
        print()


def executar_passo6():
    """Executa o Passo 6 com caminhos correctos."""
    print("\n" + "="*60)
    print("EXECUTANDO PASSO 6")
    print("="*60 + "\n")
    
    # Verificar se o dataset existe
    dataset_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    if not os.path.exists(dataset_path):
        print(f"✗ ERRO: Dataset não encontrado em {dataset_path}")
        print(f"\nFicheiros disponíveis em {config.DADOS_AGREGADOS_DIR}:")
        if os.path.exists(config.DADOS_AGREGADOS_DIR):
            for f in os.listdir(config.DADOS_AGREGADOS_DIR):
                print(f"  - {f}")
        else:
            print(f"  (directório não existe)")
        return
    
    print(f"✓ Dataset encontrado: {dataset_path}")
    
    # Importar e executar
    from passo6_estrategias import executar_passo6 as exec_p6
    exec_p6()


def executar_passo7():
    """Executa o Passo 7 com caminhos correctos."""
    print("\n" + "="*60)
    print("EXECUTANDO PASSO 7")
    print("="*60 + "\n")
    
    from passo7_shap import executar_passo7 as exec_p7
    exec_p7()


def executar_passo8():
    """Executa o Passo 8 com caminhos correctos."""
    print("\n" + "="*60)
    print("EXECUTANDO PASSO 8")
    print("="*60 + "\n")
    
    from passo8_geografica import executar_passo8 as exec_p8
    exec_p8()


def executar_passo9():
    """Executa o Passo 9 com caminhos correctos."""
    print("\n" + "="*60)
    print("EXECUTANDO PASSO 9")
    print("="*60 + "\n")
    
    from passo9_avancada import executar_passo9 as exec_p9
    exec_p9()


def executar_passo10():
    """Executa o Passo 10 com caminhos correctos."""
    print("\n" + "="*60)
    print("EXECUTANDO PASSO 10")
    print("="*60 + "\n")
    
    from passo10_inovacoes_mestrado import executar_passo10 as exec_p10
    exec_p10()


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "="*60)
print("✓ INICIALIZAÇÃO CONCLUÍDA")
print("="*60)

print("""
Comandos disponíveis:

  verificar_ficheiros()    # Listar ficheiros disponíveis
  executar_passo6()        # Executar Passo 6 (Estratégias)
  executar_passo7()        # Executar Passo 7 (SHAP)
  executar_passo8()        # Executar Passo 8 (Geográfica)
  executar_passo9()        # Executar Passo 9 (Avançada)
  executar_passo10()       # Executar Passo 10 (Inovações)

Exemplo:
  verificar_ficheiros()
  executar_passo6()
""")
