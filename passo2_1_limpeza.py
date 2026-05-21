"""
============================================================
PASSO 2.1: LIMPEZA E EDA - DATASETS SEPARADOS E INDEPENDENTES
- Limpeza WDI (Quantitativo): Interpolação + Winsorização
- Limpeza WGI (Qualitativo): Tratamento de missings + Normalização
- EDA separada para cada dataset
- Ficheiros de saída separados e independentes
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def limpar_wdi(df_wdi):
    """
    Limpeza de WDI (Quantitativo):
    1. Interpolação linear por país para preencher gaps
    2. Winsorização (1% e 99%) para remover outliers
    3. Preenchimento de NaN residuais com média por país
    """
    print("\n  === LIMPEZA WDI (Quantitativo) ===")
    print(f"  Shape original: {df_wdi.shape}")
    print(f"  Missing: {df_wdi.isna().sum().sum()} valores")
    
    df_clean = df_wdi.copy()
    
    # Colunas numéricas (excluir year e country_code)
    colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']]
    
    # 1. Interpolação linear por país
    print(f"\n  [1] Interpolação linear por país...")
    for col in colunas_numericas:
        df_clean[col] = df_clean.groupby('country_code')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
    
    # 2. Winsorização (1% e 99%)
    print(f"  [2] Winsorização (1% e 99%)...")
    for col in colunas_numericas:
        p1 = df_clean[col].quantile(0.01)
        p99 = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(lower=p1, upper=p99)
    
    # 3. Preenchimento de NaN residuais com média por país
    print(f"  [3] Preenchimento de NaN residuais...")
    for col in colunas_numericas:
        df_clean[col] = df_clean.groupby('country_code')[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Preenchimento final com média global
    for col in colunas_numericas:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    print(f"  Shape final: {df_clean.shape}")
    print(f"  Missing residuais: {df_clean.isna().sum().sum()}")
    
    return df_clean


def limpar_wgi(df_wgi):
    """
    Limpeza de WGI (Qualitativo - Governança):
    1. Interpolação linear por país
    2. Normalização para escala -2.5 a +2.5 (padrão WGI)
    3. Preenchimento de NaN com mediana por país
    """
    print("\n  === LIMPEZA WGI (Qualitativo - Governança) ===")
    print(f"  Shape original: {df_wgi.shape}")
    print(f"  Missing: {df_wgi.isna().sum().sum()} valores")
    
    df_clean = df_wgi.copy()
    
    # Colunas numéricas (excluir year e country_code)
    colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']]
    
    # 1. Interpolação linear por país
    print(f"\n  [1] Interpolação linear por país...")
    for col in colunas_numericas:
        df_clean[col] = df_clean.groupby('country_code')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
    
    # 2. Normalização para escala -2.5 a +2.5
    print(f"  [2] Normalização para escala WGI (-2.5 a +2.5)...")
    for col in colunas_numericas:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        if std > 0:
            df_clean[col] = ((df_clean[col] - mean) / std) * 0.5  # Escala aproximada
            df_clean[col] = df_clean[col].clip(lower=-2.5, upper=2.5)
    
    # 3. Preenchimento de NaN residuais com mediana por país
    print(f"  [3] Preenchimento de NaN residuais...")
    for col in colunas_numericas:
        df_clean[col] = df_clean.groupby('country_code')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Preenchimento final com mediana global
    for col in colunas_numericas:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    print(f"  Shape final: {df_clean.shape}")
    print(f"  Missing residuais: {df_clean.isna().sum().sum()}")
    
    return df_clean


def eda_wdi(df_wdi, nome_saida):
    """EDA para WDI - Gráficos e estatísticas."""
    print(f"\n  === EDA WDI ===")
    
    # Criar diretório de EDA
    eda_dir = os.path.join(config.DADOS_LIMPOS_DIR, 'eda_wdi')
    os.makedirs(eda_dir, exist_ok=True)
    
    # Estatísticas descritivas
    stats_path = os.path.join(eda_dir, f'{nome_saida}_stats.csv')
    df_wdi.describe().to_csv(stats_path)
    print(f"  ✓ Estatísticas: {stats_path}")
    
    # Gráfico de distribuição
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    colunas_numericas = df_wdi.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']][:9]
    
    for idx, col in enumerate(colunas_numericas):
        ax = axes[idx // 3, idx % 3]
        df_wdi[col].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'{col}')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
    
    plt.tight_layout()
    dist_path = os.path.join(eda_dir, f'{nome_saida}_distribuicao.png')
    plt.savefig(dist_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Distribuição: {dist_path}")
    
    # Gráfico de série temporal
    fig, ax = plt.subplots(figsize=(14, 6))
    for pais in df_wdi['country_code'].unique()[:5]:
        df_pais = df_wdi[df_wdi['country_code'] == pais].sort_values('year')
        if config.TARGET_VAR in df_pais.columns:
            ax.plot(df_pais['year'], df_pais[config.TARGET_VAR], label=pais, marker='o')
    
    ax.set_xlabel('Ano')
    ax.set_ylabel(config.TARGET_VAR)
    ax.set_title(f'Série Temporal - {config.TARGET_VAR} (5 países)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ts_path = os.path.join(eda_dir, f'{nome_saida}_serie_temporal.png')
    plt.savefig(ts_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Série Temporal: {ts_path}")
    
    return eda_dir


def eda_wgi(df_wgi, nome_saida):
    """EDA para WGI - Gráficos e estatísticas."""
    print(f"\n  === EDA WGI ===")
    
    # Criar diretório de EDA
    eda_dir = os.path.join(config.DADOS_LIMPOS_DIR, 'eda_wgi')
    os.makedirs(eda_dir, exist_ok=True)
    
    # Estatísticas descritivas
    stats_path = os.path.join(eda_dir, f'{nome_saida}_stats.csv')
    df_wgi.describe().to_csv(stats_path)
    print(f"  ✓ Estatísticas: {stats_path}")
    
    # Gráfico de distribuição
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colunas_numericas = df_wgi.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']]
    
    for idx, col in enumerate(colunas_numericas):
        ax = axes[idx // 3, idx % 3]
        df_wgi[col].hist(bins=30, ax=ax, edgecolor='black', color='steelblue')
        ax.set_title(f'{col}')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
    
    plt.tight_layout()
    dist_path = os.path.join(eda_dir, f'{nome_saida}_distribuicao.png')
    plt.savefig(dist_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Distribuição: {dist_path}")
    
    # Heatmap de correlação
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_wgi[colunas_numericas].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlação entre Indicadores WGI')
    
    corr_path = os.path.join(eda_dir, f'{nome_saida}_correlacao.png')
    plt.savefig(corr_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Correlação: {corr_path}")
    
    return eda_dir


def executar_passo2_1():
    """Executa o Passo 2.1 completo - Limpeza separada e independente."""
    print("\n" + "=" * 70)
    print("PASSO 2.1: LIMPEZA E EDA - DATASETS SEPARADOS E INDEPENDENTES")
    print("=" * 70)
    
    # Carregar dados brutos (aceita ambos os nomes de ficheiro)
    wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_bruto.csv')
    if not os.path.exists(wdi_path):
        wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_africa_mo_bruto.csv')
    wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_bruto.csv')
    if not os.path.exists(wgi_path):
        wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_africa_mo_bruto.csv')
    
    df_wdi_bruto = pd.read_csv(wdi_path)
    df_wgi_bruto = pd.read_csv(wgi_path)
    
    print(f"\n  WDI bruto: {df_wdi_bruto.shape}")
    print(f"  WGI bruto: {df_wgi_bruto.shape}")
    
    # ============================================================
    # LIMPEZA SEPARADA E INDEPENDENTE
    # ============================================================
    
    # Limpeza WDI
    df_wdi_limpo = limpar_wdi(df_wdi_bruto)
    
    # Limpeza WGI
    df_wgi_limpo = limpar_wgi(df_wgi_bruto)
    
    # ============================================================
    # SALVAR DATASETS LIMPOS SEPARADOS
    # ============================================================
    
    os.makedirs(config.DADOS_LIMPOS_DIR, exist_ok=True)
    
    wdi_limpo_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wdi_limpo.csv')
    wgi_limpo_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wgi_limpo.csv')
    
    df_wdi_limpo.to_csv(wdi_limpo_path, index=False)
    df_wgi_limpo.to_csv(wgi_limpo_path, index=False)
    
    print(f"\n  ✓ WDI limpo salvo: {wdi_limpo_path}")
    print(f"  ✓ WGI limpo salvo: {wgi_limpo_path}")
    
    # ============================================================
    # EDA SEPARADA E INDEPENDENTE
    # ============================================================
    
    eda_wdi(df_wdi_limpo, 'wdi_limpo')
    eda_wgi(df_wgi_limpo, 'wgi_limpo')
    
    # ============================================================
    # METADADOS
    # ============================================================
    
    gerar_metadados(
        passo='passo2_1_limpeza',
        descricao='Limpeza separada e independente de WDI (interpolação + winsorização) e WGI (interpolação + normalização). EDA separada para cada dataset.',
        config=config,
        dados_entrada=[wdi_path, wgi_path],
        dados_saida=[wdi_limpo_path, wgi_limpo_path],
        parametros={
            'wdi_interpolacao': 'linear por país',
            'wdi_winsorização': '1% e 99%',
            'wgi_interpolacao': 'linear por país',
            'wgi_normalização': 'escala -2.5 a +2.5',
        },
        metricas={
            'wdi': registar_dataframe_info(df_wdi_limpo, 'WDI Limpo'),
            'wgi': registar_dataframe_info(df_wgi_limpo, 'WGI Limpo'),
        }
    )
    
    auto_save_drive([wdi_limpo_path, wgi_limpo_path], config)
    
    print("\n  ✓ PASSO 2.1 CONCLUÍDO")
    return df_wdi_limpo, df_wgi_limpo


if __name__ == '__main__':
    executar_passo2_1()
