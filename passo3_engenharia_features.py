"""
============================================================
PASSO 3: ENGENHARIA DE FEATURES - 4 DATASETS INDEPENDENTES
- Dataset 1: WDI limpo (não agregado, sem WGI)
- Dataset 2: Agregado (WDI + WGI limpos, INNER JOIN)
- Dataset 3: Sintético Agregado (WDI + WGI, 500 anos)
- Dataset 4: WDI Sintético (apenas WDI, sem WGI, 500 anos)

Cada dataset recebe as mesmas transformações:
- Lags (1,2,3 para WGI; 1 para WDI; 1,2 para target)
- Médias móveis (3 anos para WGI)
- Deltas (diferenças para WGI)
- Log-retornos (4 variáveis principais)
- Interações (4 pares H2)
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def criar_lags(df, colunas, lags):
    """Cria variáveis defasadas por país."""
    novas_cols = []
    for col in colunas:
        if col not in df.columns:
            continue
        for lag in lags:
            nome_lag = f'{col}_lag{lag}'
            df[nome_lag] = df.groupby('country_code')[col].shift(lag)
            novas_cols.append(nome_lag)
    return df, novas_cols


def criar_medias_moveis(df, colunas, janela=3):
    """Cria médias móveis por país."""
    novas_cols = []
    for col in colunas:
        if col not in df.columns:
            continue
        nome_ma = f'{col}_ma{janela}'
        df[nome_ma] = df.groupby('country_code')[col].transform(
            lambda x: x.rolling(janela, min_periods=1).mean()
        )
        novas_cols.append(nome_ma)
    return df, novas_cols


def criar_deltas(df, colunas):
    """Cria diferenças de primeira ordem por país."""
    novas_cols = []
    for col in colunas:
        if col not in df.columns:
            continue
        nome_delta = f'{col}_delta'
        df[nome_delta] = df.groupby('country_code')[col].diff(1)
        novas_cols.append(nome_delta)
    return df, novas_cols


def criar_log_retornos(df, colunas):
    """Cria log-retornos para variáveis quantitativas."""
    novas_cols = []
    for col in colunas:
        if col not in df.columns:
            continue
        nome_lr = f'{col}_log_ret'
        df[nome_lr] = df.groupby('country_code')[col].apply(
            lambda x: np.log(x.clip(lower=0.01)).diff()
        ).reset_index(level=0, drop=True)
        novas_cols.append(nome_lr)
    return df, novas_cols


def criar_interacoes(df, pares_interacao):
    """Cria termos de interação defasados."""
    novas_cols = []
    for var_qual, var_quant in pares_interacao:
        col_qual = f'{var_qual}_lag1' if f'{var_qual}_lag1' in df.columns else var_qual
        col_quant = f'{var_quant}_lag1' if f'{var_quant}_lag1' in df.columns else var_quant
        
        if col_qual not in df.columns or col_quant not in df.columns:
            continue
        
        nome_inter = f'inter_{var_qual.replace("wgi_", "")}_{var_quant.split("_")[0]}'
        
        std_qual = df[col_qual].std()
        std_quant = df[col_quant].std()
        
        if std_qual > 0 and std_quant > 0:
            df[nome_inter] = (df[col_qual] / std_qual) * (df[col_quant] / std_quant)
        else:
            df[nome_inter] = df[col_qual] * df[col_quant]
        
        novas_cols.append(nome_inter)
    
    return df, novas_cols


def engenharia_features(df, nome_dataset):
    """Aplica engenharia de features a um dataset."""
    print(f"\n  === Engenharia de Features: {nome_dataset} ===")
    print(f"  Shape original: {df.shape}")
    
    # Identificar tipos de variáveis
    todas_colunas = df.columns.tolist()
    colunas_wgi = [c for c in todas_colunas if 'wgi_' in c and '_lag' not in c 
                   and '_ma' not in c and '_delta' not in c]
    colunas_quant = [c for c in df.select_dtypes(include=[np.number]).columns 
                     if c not in colunas_wgi and c != 'year' and c != config.TARGET_VAR
                     and '_lag' not in c and '_ma' not in c and '_delta' not in c 
                     and '_log_ret' not in c and 'inter_' not in c]
    
    features_criadas = {}
    
    # 1. Lags das variáveis qualitativas (1, 2, 3)
    df, lags_wgi = criar_lags(df, colunas_wgi, config.LAGS_QUALITATIVOS)
    features_criadas['lags_wgi'] = lags_wgi
    print(f"  [1] Lags WGI (1,2,3): {len(lags_wgi)} features")
    
    # 2. Lags das variáveis quantitativas (1)
    df, lags_quant = criar_lags(df, colunas_quant, config.LAGS_QUANTITATIVOS)
    features_criadas['lags_quant'] = lags_quant
    print(f"  [2] Lags quantitativos (1): {len(lags_quant)} features")
    
    # 3. Lags da variável alvo (1, 2)
    df, lags_target = criar_lags(df, [config.TARGET_VAR], config.LAGS_TARGET)
    features_criadas['lags_target'] = lags_target
    print(f"  [3] Lags target (1,2): {len(lags_target)} features")
    
    # 4. Médias móveis das variáveis qualitativas (3 anos)
    df, ma_wgi = criar_medias_moveis(df, colunas_wgi, config.JANELA_MEDIA_MOVEL)
    features_criadas['ma_wgi'] = ma_wgi
    print(f"  [4] Médias móveis WGI (3 anos): {len(ma_wgi)} features")
    
    # 5. Deltas das variáveis qualitativas
    df, deltas_wgi = criar_deltas(df, colunas_wgi)
    features_criadas['deltas_wgi'] = deltas_wgi
    print(f"  [5] Deltas WGI: {len(deltas_wgi)} features")
    
    # 6. Log-retornos
    vars_log_ret = ['pib_per_capita_ppc', 'formacao_bruta_capital_fixo_percent_pib',
                    'comercio_percent_pib', 'ied_percent_pib']
    vars_log_ret = [v for v in vars_log_ret if v in df.columns]
    df, log_rets = criar_log_retornos(df, vars_log_ret)
    features_criadas['log_retornos'] = log_rets
    print(f"  [6] Log-retornos: {len(log_rets)} features")
    
    # 7. Interações
    df, interacoes = criar_interacoes(df, config.INTERACOES)
    features_criadas['interacoes'] = interacoes
    print(f"  [7] Interações (H2): {len(interacoes)} features")
    
    # 8. Remover linhas com NaN dos primeiros lags
    max_lag = max(config.LAGS_QUALITATIVOS)
    df_clean = df[df['year'] >= config.ANO_INICIO + max_lag].copy()
    
    # Imputar NaN residuais
    feature_cols = []
    for lista in features_criadas.values():
        feature_cols.extend(lista)
    
    for col in feature_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    total_features = sum(len(v) for v in features_criadas.values())
    print(f"\n  Total features criadas: {total_features}")
    print(f"  Shape final: {df_clean.shape}")
    
    return df_clean, features_criadas


def executar_passo3():
    """Executa o Passo 3 completo - Engenharia em 4 datasets."""
    print("\n" + "=" * 70)
    print("PASSO 3: ENGENHARIA DE FEATURES - 4 DATASETS INDEPENDENTES")
    print("=" * 70)
    
    os.makedirs(config.DADOS_ENGENHARIA_DIR, exist_ok=True)
    
    # ============================================================
    # DATASET 1: WDI LIMPO (NÃO AGREGADO)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 1: WDI LIMPO (NÃO AGREGADO)")
    print("-" * 70)
    
    wdi_limpo_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wdi_limpo.csv')
    df_wdi_limpo = pd.read_csv(wdi_limpo_path)
    
    df_wdi_eng, features_wdi = engenharia_features(df_wdi_limpo.copy(), 'WDI Limpo')
    
    wdi_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv')
    df_wdi_eng.to_csv(wdi_eng_path, index=False)
    print(f"  ✓ WDI com features: {wdi_eng_path}")
    
    # ============================================================
    # DATASET 2: AGREGADO (WDI + WGI LIMPOS)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 2: AGREGADO (WDI + WGI LIMPOS, INNER JOIN)")
    print("-" * 70)
    
    agregado_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    df_agregado = pd.read_csv(agregado_path)
    
    df_agregado_eng, features_agregado = engenharia_features(df_agregado.copy(), 'Agregado')
    
    agregado_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv')
    df_agregado_eng.to_csv(agregado_eng_path, index=False)
    print(f"  ✓ Agregado com features: {agregado_eng_path}")
    
    # ============================================================
    # DATASET 3: SINTÉTICO (500 ANOS)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 3: SINTÉTICO (500 ANOS)")
    print("-" * 70)
    
    sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'sintetico_500anos.csv')
    df_sintetico = pd.read_csv(sintetico_path)
    
    df_sintetico_eng, features_sintetico = engenharia_features(df_sintetico.copy(), 'Sintético')
    
    sintetico_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv')
    df_sintetico_eng.to_csv(sintetico_eng_path, index=False)
    print(f"  ✓ Sintético com features: {sintetico_eng_path}")
    
    # ============================================================
    # DATASET 4: WDI SINTÉTICO (500 ANOS, SEM WGI)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 4: WDI SINTÉTICO (500 ANOS, SEM WGI)")
    print("-" * 70)
    
    wdi_sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'wdi_sintetico_500anos.csv')
    df_wdi_sintetico = pd.read_csv(wdi_sintetico_path)
    
    df_wdi_sint_eng, features_wdi_sint = engenharia_features(df_wdi_sintetico.copy(), 'WDI Sintético')
    
    wdi_sint_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv')
    df_wdi_sint_eng.to_csv(wdi_sint_eng_path, index=False)
    print(f"  ✓ WDI Sintético com features: {wdi_sint_eng_path}")
    
    # ============================================================
    # METADADOS
    # ============================================================
    
    ficheiros_saida = [wdi_eng_path, agregado_eng_path, sintetico_eng_path, wdi_sint_eng_path]
    
    gerar_metadados(
        passo='passo3_engenharia_features',
        descricao='Engenharia de features em 4 datasets independentes: WDI limpo, Agregado, Sintético Agregado, WDI Sintético. Mesmas transformações: lags, MA, deltas, log-retornos, interações.',
        config=config,
        dados_entrada=[wdi_limpo_path, agregado_path, sintetico_path, wdi_sintetico_path],
        dados_saida=ficheiros_saida,
        parametros={
            'lags_qualitativos': config.LAGS_QUALITATIVOS,
            'lags_quantitativos': config.LAGS_QUANTITATIVOS,
            'lags_target': config.LAGS_TARGET,
            'janela_media_movel': config.JANELA_MEDIA_MOVEL,
            'interacoes': [f'{a}×{b}' for a, b in config.INTERACOES],
        },
        metricas={
            'wdi_features': registar_dataframe_info(df_wdi_eng, 'WDI com Features'),
            'agregado_features': registar_dataframe_info(df_agregado_eng, 'Agregado com Features'),
            'sintetico_agregado_features': registar_dataframe_info(df_sintetico_eng, 'Sintético Agregado com Features'),
            'wdi_sintetico_features': registar_dataframe_info(df_wdi_sint_eng, 'WDI Sintético com Features'),
        }
    )
    
    auto_save_drive(ficheiros_saida, config)
    
    print("\n  ✓ PASSO 3 CONCLUÍDO")
    return df_wdi_eng, df_agregado_eng, df_sintetico_eng, df_wdi_sint_eng


if __name__ == '__main__':
    executar_passo3()
