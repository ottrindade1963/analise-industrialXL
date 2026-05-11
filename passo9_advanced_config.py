import os
import sys

# ============================================================
# CONFIGURAÇÕES DO PASSO 9: ANÁLISES AVANÇADAS
# ============================================================

# Detecção automática de ambiente (Colab vs Local)
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    if _script_dir and os.path.exists(os.path.join(_script_dir, 'passo9_advanced_processor.py')):
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
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analises_avancadas')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Análise de Sensibilidade
SENSITIVITY_VARS = [
    'wgi_control_corruption', 
    'wgi_rule_law', 
    'wgi_government_effectiveness',
    'wgi_political_stability',
    'fbc_percent_pib',
    'investimento_estrangeiro_direto_percent_pib',
    'comercio_percent_pib'
]
SENSITIVITY_STEPS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# Datasets e Modelos a analisar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']
MODELS = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
          'Bayes_PartialPooling', 'Bayes_CompletePooling']
