import os
import sys

# ============================================================
# CONFIGURAÇÕES DO PASSO 7: INTERPRETABILIDADE SHAP
# ============================================================

# Detecção automática de ambiente (Colab vs Local)
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    if _script_dir and os.path.exists(os.path.join(_script_dir, 'passo7_shap_processor.py')):
        BASE_DIR = _script_dir
    else:
        _dirs = [d for d in os.listdir('/content') 
                 if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data', 'drive']]
        BASE_DIR = os.path.join('/content', _dirs[0]) if _dirs else os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos
DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'interpretabilidade_shap')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Modelos Suportados pelo SHAP (TreeExplainer)
MODELS_FOR_SHAP = ['RandomForest', 'XGBoost']

# Datasets e Estratégias
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Número de features a mostrar nos gráficos
TOP_N_FEATURES = 15
