import os
import sys

# ============================================================
# CONFIGURAÇÕES DO PASSO 6: ANÁLISE DE ESTRATÉGIAS
# ============================================================

# Detecção automática de ambiente (Colab vs Local)
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    if _script_dir and os.path.exists(os.path.join(_script_dir, 'passo6_strategy_processor.py')):
        BASE_DIR = _script_dir
    else:
        _dirs = [d for d in os.listdir('/content') 
                 if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data', 'drive']]
        BASE_DIR = os.path.join('/content', _dirs[0]) if _dirs else os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')  # Lê os resultados do Passo 5
OUTPUT_DIR = os.path.join(BASE_DIR, 'analise_estrategias')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nomes das Estratégias
STRATEGIES = {
    'A1_Direta': 'A1: Inclusão Direta',
    'A2_PCA': 'A2: Fator Latente (PCA)',
    'A3_Interacao': 'A3: Termos de Interação'
}

# Baselines para comparação
BASELINE_DATASET = 'nao_agregado'
BASELINE_STRATEGY = 'A1_Direta'
