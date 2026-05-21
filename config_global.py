"""
============================================================
CONFIGURAÇÃO GLOBAL DO PIPELINE
Análise Industrial - África e Médio Oriente
============================================================
"""
import os
import sys

# ============================================================
# DETECÇÃO AUTOMÁTICA DE AMBIENTE
# ============================================================
_IN_COLAB = 'google.colab' in sys.modules
if _IN_COLAB:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    if _script_dir and os.path.exists(os.path.join(_script_dir, 'config_global.py')):
        BASE_DIR = _script_dir
    else:
        _dirs = [d for d in os.listdir('/content')
                 if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data', 'drive']]
        BASE_DIR = os.path.join('/content', _dirs[0]) if _dirs else os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DIRETÓRIOS
# ============================================================
DADOS_BRUTOS_DIR = os.path.join(BASE_DIR, 'dados_brutos')
DADOS_LIMPOS_DIR = os.path.join(BASE_DIR, 'dados_limpos')
DADOS_AGREGADOS_DIR = os.path.join(BASE_DIR, 'dados_agregados')
DADOS_SINTETICOS_DIR = os.path.join(BASE_DIR, 'dados_sinteticos')
DADOS_ENGENHARIA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODELOS_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
RESULTADOS_AVALIACAO_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')
ANALISE_ESTRATEGIAS_DIR = os.path.join(BASE_DIR, 'analise_estrategias')
SHAP_DIR = os.path.join(BASE_DIR, 'shap_analysis')
GEO_DIR = os.path.join(BASE_DIR, 'analise_geografica')
AVANCADA_DIR = os.path.join(BASE_DIR, 'analise_avancada')
METADADOS_DIR = os.path.join(BASE_DIR, 'metadados')

# Aliases para compatibilidade
RESULTADOS_DIR = RESULTADOS_AVALIACAO_DIR
ESTRATEGIAS_DIR = ANALISE_ESTRATEGIAS_DIR
GEOGRAFICA_DIR = GEO_DIR

# Criar todos os diretórios
for d in [DADOS_BRUTOS_DIR, DADOS_LIMPOS_DIR, DADOS_AGREGADOS_DIR,
          DADOS_SINTETICOS_DIR, DADOS_ENGENHARIA_DIR, MODELOS_DIR,
          RESULTADOS_AVALIACAO_DIR, ANALISE_ESTRATEGIAS_DIR,
          SHAP_DIR, GEO_DIR, AVANCADA_DIR, METADADOS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# GOOGLE DRIVE AUTO-SAVE
# ============================================================
if _IN_COLAB:
    DRIVE_SAVE_DIR = '/content/drive/MyDrive/analise_industrial_africa_mo/'
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
else:
    DRIVE_SAVE_DIR = None

# ============================================================
# PERÍODO DE ANÁLISE
# ============================================================
ANO_INICIO = 1996
ANO_FIM = 2023
TRAIN_END_YEAR = 2017  # Treino: 1996-2017
VAL_END_YEAR = 2020    # Validação: 2018-2020
# Teste: 2021-2023

# ============================================================
# VARIÁVEL ALVO
# ============================================================
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# ============================================================
# PAÍSES - ÁFRICA E MÉDIO ORIENTE
# ============================================================
# África Subsaariana + Norte de África + Médio Oriente
PAISES = {
    # Norte de África
    'DZA': 'Argélia',
    'EGY': 'Egito',
    'LBY': 'Líbia',
    'MAR': 'Marrocos',
    'TUN': 'Tunísia',
    # Médio Oriente
    'BHR': 'Bahrein',
    'IRN': 'Irão',
    'IRQ': 'Iraque',
    'JOR': 'Jordânia',
    'KWT': 'Kuwait',
    'LBN': 'Líbano',
    'OMN': 'Omã',
    'QAT': 'Qatar',
    'SAU': 'Arábia Saudita',
    'ARE': 'Emirados Árabes Unidos',
    'YEM': 'Iémen',
    'TUR': 'Turquia',
    # África Subsaariana (principais economias emergentes)
    'ZAF': 'África do Sul',
    'NGA': 'Nigéria',
    'KEN': 'Quénia',
    'GHA': 'Gana',
    'ETH': 'Etiópia',
    'TZA': 'Tanzânia',
    'CIV': "Costa do Marfim",
    'SEN': 'Senegal',
    'CMR': 'Camarões',
    'AGO': 'Angola',
    'MOZ': 'Moçambique',
    'UGA': 'Uganda',
    'RWA': 'Ruanda',
    'COD': 'RD Congo',
    'ZMB': 'Zâmbia',
    'BWA': 'Botsuana',
    'MUS': 'Maurícia',
    'NAM': 'Namíbia',
    'GAB': 'Gabão',
    'MDG': 'Madagáscar',
}

PAISES_CODIGOS = list(PAISES.keys())

# ============================================================
# INDICADORES WDI (Quantitativos)
# ============================================================
INDICADORES_WDI = {
    # Variável Alvo
    'NV.IND.TOTL.ZS': 'valor_agregado_industrial_percent_pib',
    # Crescimento e Renda
    'NY.GDP.PCAP.PP.KD': 'pib_per_capita_ppc',
    'NY.GDP.MKTP.KD.ZG': 'crescimento_pib_anual',
    # Investimento e Capital
    'NE.GDI.FTOT.ZS': 'formacao_bruta_capital_fixo_percent_pib',
    'BX.KLT.DINV.WD.GD.ZS': 'ied_percent_pib',
    # Comércio e Abertura
    'NE.TRD.GNFS.ZS': 'comercio_percent_pib',
    'NE.EXP.GNFS.ZS': 'exportacoes_percent_pib',
    'NE.IMP.GNFS.ZS': 'importacoes_percent_pib',
    # Infraestrutura e Tecnologia
    'EG.USE.ELEC.KH.PC': 'consumo_eletricidade_per_capita',
    'IT.NET.USER.ZS': 'utilizadores_internet_percent',
    'GB.XPD.RSDV.GD.ZS': 'despesa_id_percent_pib',
    # Capital Humano
    'SE.XPD.TOTL.GD.ZS': 'despesa_educacao_percent_pib',
    'SL.TLF.TOTL.IN': 'forca_trabalho_total',
    'SE.TER.ENRR': 'taxa_matricula_terciario',
    # Demografia e Urbanização
    'SP.URB.TOTL.IN.ZS': 'populacao_urbana_percent',
    'SP.POP.GROW': 'crescimento_populacional',
    # Setor Financeiro
    'FS.AST.PRVT.GD.ZS': 'credito_privado_percent_pib',
    # Manufatura
    'NV.IND.MANF.ZS': 'valor_agregado_manufatura_percent_pib',
    # Recursos Naturais
    'NY.GDP.TOTL.RT.ZS': 'rendas_recursos_naturais_percent_pib',
}

# ============================================================
# INDICADORES WGI (Qualitativos - Governança)
# ============================================================
INDICADORES_WGI = {
    'CC.EST': 'wgi_controle_corrupcao',
    'GE.EST': 'wgi_eficacia_governo',
    'PV.EST': 'wgi_estabilidade_politica',
    'RQ.EST': 'wgi_qualidade_regulatoria',
    'RL.EST': 'wgi_estado_direito',
    'VA.EST': 'wgi_voz_responsabilizacao',
}

# ============================================================
# CONFIGURAÇÃO DE ENGENHARIA DE FEATURES
# ============================================================
# Defasagens (lags) para variáveis qualitativas
LAGS_QUALITATIVOS = [1, 2, 3]
# Defasagens para variáveis quantitativas
LAGS_QUANTITATIVOS = [1]
# Defasagens da variável alvo
LAGS_TARGET = [1, 2]
# Janela para média móvel
JANELA_MEDIA_MOVEL = 3

# Interações a criar (conforme hipóteses H2)
INTERACOES = [
    ('wgi_qualidade_regulatoria', 'ied_percent_pib'),
    ('wgi_controle_corrupcao', 'formacao_bruta_capital_fixo_percent_pib'),
    ('wgi_estabilidade_politica', 'comercio_percent_pib'),
    ('wgi_estado_direito', 'pib_per_capita_ppc'),
]

# ============================================================
# CONFIGURAÇÃO DE DADOS SINTÉTICOS
# ============================================================
ANOS_SINTETICOS = 500  # Extrapolação para 500 anos

# ============================================================
# CONFIGURAÇÃO DE MODELOS
# ============================================================
MODELOS = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
           'Bayes_PartialPooling', 'Bayes_CompletePooling']

# Hiperparâmetros
RF_N_ITER = 50
RF_CV_FOLDS = 5
XGB_EARLY_STOPPING = 50
TFT_N_ITER = 100
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 32
LSTM_LOOKBACK = 5  # Janela sequencial de 5 anos
BAYESIAN_SAMPLES = 2000
BAYESIAN_TUNE = 1000
BAYESIAN_CHAINS = 2

# ============================================================
# DATASETS PARA TREINO
# ============================================================
DATASETS = ['agregado', 'sintetico']
STRATEGIES = ['A1_Direta']  # Apenas INNER JOIN, sem left/outer

# ============================================================
# METADADOS
# ============================================================
PROJETO_NOME = "Análise Industrial - África e Médio Oriente"
PROJETO_VERSAO = "2.0"
PROJETO_AUTOR = "Dissertação de Mestrado"
