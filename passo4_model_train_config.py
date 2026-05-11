"""
Passo 4 - Configuração de Treinamento de Modelos
==================================================
Configurações robustas para trabalho de mestrado.
Todos os hiperparâmetros são extensivos e justificáveis academicamente.
"""
import os
import sys

# Detectar se está no Colab
IN_COLAB = 'google.colab' in sys.modules

# Detectar o diretório raiz dinamicamente
if IN_COLAB:
    import glob
    dirs = [d for d in os.listdir('/content') if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data']]
    REPO_DIR = dirs[0] if dirs else os.getcwd()
    os.chdir(REPO_DIR)
    BASE_DIR = REPO_DIR
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configurações de Caminhos
DATA_DIR = 'dados_engenharia'
OUTPUT_DIR = 'modelos_treinados'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Divisão Temporal
# Dados cobrem 1996-2024 (~28 anos)
TRAIN_END_YEAR = 2016   # Treino: 1996-2016 (~21 anos = 75%)
VAL_END_YEAR = 2019     # Validação: 2017-2019 (3 anos = ~11%)
# Teste: 2020-2024 (5 anos = ~14%)

# Datasets e Estratégias a processar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Seed para reprodutibilidade
RANDOM_STATE = 42

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - RANDOM FOREST
# ============================================================
# Grid extensivo: 576 combinações possíveis
# RandomizedSearchCV com n_iter=100 para explorar espaço amplo
RF_N_ITER = 50  # Iterações de busca aleatória
RF_CV_FOLDS = 5  # Folds de validação cruzada temporal
RF_GRID = {
    'n_estimators': [100, 200, 300, 500, 700, 1000],
    'max_depth': [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'bootstrap': [True, False]
}

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - XGBOOST
# ============================================================
# Grid extensivo com early stopping
# RandomizedSearchCV com n_iter=100
XGB_N_ITER = 100
XGB_CV_FOLDS = 5
XGB_EARLY_STOPPING_ROUNDS = 50  # Parar se não melhorar em 50 rounds
XGB_GRID = {
    'n_estimators': [100, 200, 300, 500, 700, 1000],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - TFT (GradientBoosting)
# ============================================================
# Grid extensivo para GradientBoosting com tuning completo
TFT_N_ITER = 50
TFT_CV_FOLDS = 5
TFT_GRID = {
    'n_estimators': [100, 200, 300, 500, 700, 1000],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
    'loss': ['squared_error', 'huber']
}

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - SARIMAX (auto_arima via pmdarima)
# ============================================================
# Busca automática de (p,d,q)(P,D,Q,s) via critério AIC/BIC
SARIMAX_MAX_P = 3        # Ordem máxima AR
SARIMAX_MAX_D = 2        # Ordem máxima de diferenciação
SARIMAX_MAX_Q = 3        # Ordem máxima MA
SARIMAX_SEASONAL = False  # Dados anuais - sem sazonalidade sub-anual
SARIMAX_STEPWISE = True   # Busca stepwise (mais rápido que grid)
SARIMAX_INFORMATION_CRITERION = 'aic'  # Critério de seleção
SARIMAX_N_EXOG = 5       # Número de exógenas a incluir (top por correlação)
SARIMAX_MAXITER = 500    # Iterações máximas de convergência
SARIMAX_SUPPRESS_WARNINGS = True

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - LSTM (Keras/TensorFlow REAL)
# ============================================================
# LSTM real com camadas recorrentes, dropout, early stopping
LSTM_UNITS_LAYER1 = 128       # Neurónios na 1ª camada LSTM
LSTM_UNITS_LAYER2 = 64        # Neurónios na 2ª camada LSTM
LSTM_DENSE_UNITS = 32         # Neurónios na camada densa
LSTM_DROPOUT = 0.2            # Dropout entre camadas
LSTM_RECURRENT_DROPOUT = 0.1  # Dropout recorrente
LSTM_EPOCHS = 200             # Épocas máximas
LSTM_BATCH_SIZE = 16          # Batch size (dados pequenos por país)
LSTM_PATIENCE = 30            # Early stopping patience
LSTM_LEARNING_RATE = 0.001    # Learning rate Adam
LSTM_LOOKBACK = 5             # Janela temporal (anos anteriores como input)
LSTM_MIN_TRAIN_SAMPLES = 10   # Mínimo de amostras de treino por país
LSTM_VALIDATION_SPLIT = 0.15  # Fração para validação interna

# ============================================================
# CONFIGURAÇÕES ROBUSTAS - BAYESIANO HIERÁRQUICO (PyMC)
# ============================================================
BAYESIAN_N_SAMPLES = 2000       # Amostras MCMC após burn-in
BAYESIAN_N_TUNE = 1000          # Amostras de burn-in (tuning)
BAYESIAN_TARGET_ACCEPT = 0.9    # Taxa de aceitação NUTS
BAYESIAN_MAX_FEATURES = 5       # Máximo de features (estabilidade MCMC)
BAYESIAN_CHAINS = 2             # Número de cadeias MCMC
BAYESIAN_RHAT_THRESHOLD = 1.05  # Limiar de convergência R-hat

# ============================================================
# CONFIGURAÇÕES GERAIS DE VALIDAÇÃO
# ============================================================
SCORING = 'neg_mean_squared_error'
MIN_COUNTRIES_FOR_PER_COUNTRY = 5  # Mínimo de países para análise por país
