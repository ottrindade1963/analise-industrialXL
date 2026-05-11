import os

# ============================================================
# CONFIGURAÇÕES DO PASSO 5: AVALIAÇÃO DE PERFORMANCE
# ============================================================

# Detecção automática de ambiente (Colab vs Local)
try:
    import google.colab
    BASE_DIR = '/content/analise-industrialX'
except ImportError:
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
