import os
import sys

# ============================================================
# CONFIGURAÇÕES DO PASSO 5: AVALIAÇÃO DE PERFORMANCE
# ============================================================

# Detecção automática de ambiente (Colab vs Local)
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    # No Colab: detectar o diretório do repositório dinamicamente
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    if _script_dir and os.path.exists(os.path.join(_script_dir, 'passo5_eval_processor.py')):
        BASE_DIR = _script_dir
    else:
        # Fallback: procurar o repositório em /content
        _dirs = [d for d in os.listdir('/content') 
                 if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data', 'drive']]
        BASE_DIR = os.path.join('/content', _dirs[0]) if _dirs else os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos
DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Modelos (incluindo Bayesianos)
MODELS = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
          'Bayes_PartialPooling', 'Bayes_CompletePooling']

# Datasets e Estratégias
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']
