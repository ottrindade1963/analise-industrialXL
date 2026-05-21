"""
============================================================
PASSO 2: ANÁLISE EXPLORATÓRIA DOS DADOS BRUTOS
África e Médio Oriente
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


def eda_dataset(df, nome, output_dir):
    """Realiza EDA completa de um dataset."""
    print(f"\n  === EDA: {nome} ===")
    print(f"  Shape: {df.shape}")
    print(f"  Países: {df['country_code'].nunique()}")
    print(f"  Anos: {df['year'].min()}-{df['year'].max()}")
    
    resultados = {}
    
    # 1. Estatísticas descritivas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    stats = df[numeric_cols].describe().T
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
    stats_path = os.path.join(output_dir, f'{nome}_estatisticas_descritivas.csv')
    stats.to_csv(stats_path)
    resultados['estatisticas'] = stats_path
    
    # 2. Matriz de correlação
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Matriz de Correlação - {nome}', fontsize=14)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, f'{nome}_correlacao.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()
    resultados['correlacao'] = corr_path
    
    # 3. Missing data por variável
    fig, ax = plt.subplots(figsize=(12, 6))
    missing = df[numeric_cols].isnull().sum().sort_values(ascending=False)
    missing_pct = missing / len(df) * 100
    missing_with_data = missing_pct[missing_pct > 0]
    if len(missing_with_data) > 0:
        missing_with_data.plot(kind='bar', ax=ax, color='coral')
    else:
        ax.text(0.5, 0.5, 'Sem dados faltantes', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'Dados Faltantes (%) - {nome}', fontsize=14)
    ax.set_ylabel('% Missing')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    missing_path = os.path.join(output_dir, f'{nome}_missing.png')
    plt.savefig(missing_path, dpi=150, bbox_inches='tight')
    plt.close()
    resultados['missing'] = missing_path
    
    # 4. Distribuição da variável alvo (se existir)
    if config.TARGET_VAR in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[config.TARGET_VAR].hist(bins=30, ax=axes[0], color='steelblue', edgecolor='white')
        axes[0].set_title(f'Distribuição - {config.TARGET_VAR}')
        axes[0].set_xlabel('Valor')
        
        # Evolução temporal por país (amostra)
        sample_countries = df['country_code'].unique()[:8]
        for cc in sample_countries:
            subset = df[df['country_code'] == cc]
            axes[1].plot(subset['year'], subset[config.TARGET_VAR], label=cc, alpha=0.7)
        axes[1].set_title('Evolução Temporal (amostra)')
        axes[1].legend(fontsize=8, ncol=2)
        axes[1].set_xlabel('Ano')
        plt.tight_layout()
        target_path = os.path.join(output_dir, f'{nome}_target_dist.png')
        plt.savefig(target_path, dpi=150, bbox_inches='tight')
        plt.close()
        resultados['target'] = target_path
    
    # 5. Cobertura temporal por país
    fig, ax = plt.subplots(figsize=(14, 8))
    coverage = df.groupby('country_code')[numeric_cols].apply(lambda x: x.notna().sum() / len(x) * 100)
    coverage_mean = coverage.mean(axis=1).sort_values()
    coverage_mean.plot(kind='barh', ax=ax, color='teal')
    ax.set_title(f'Cobertura Média de Dados por País - {nome}', fontsize=14)
    ax.set_xlabel('% Dados Disponíveis')
    ax.axvline(x=70, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    coverage_path = os.path.join(output_dir, f'{nome}_cobertura.png')
    plt.savefig(coverage_path, dpi=150, bbox_inches='tight')
    plt.close()
    resultados['cobertura'] = coverage_path
    
    return resultados


def executar_passo2():
    """Executa o Passo 2 completo."""
    print("\n" + "=" * 60)
    print("PASSO 2: ANÁLISE EXPLORATÓRIA DOS DADOS BRUTOS")
    print("=" * 60)
    
    output_dir = os.path.join(config.BASE_DIR, 'eda_brutos')
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dados brutos (aceita ambos os nomes)
    wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_bruto.csv')
    if not os.path.exists(wdi_path):
        wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_africa_mo_bruto.csv')
    wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_bruto.csv')
    if not os.path.exists(wgi_path):
        wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_africa_mo_bruto.csv')
    
    df_wdi = pd.read_csv(wdi_path)
    df_wgi = pd.read_csv(wgi_path)
    
    # EDA WDI
    res_wdi = eda_dataset(df_wdi, 'WDI', output_dir)
    
    # EDA WGI
    res_wgi = eda_dataset(df_wgi, 'WGI', output_dir)
    
    # Metadados
    ficheiros = list(res_wdi.values()) + list(res_wgi.values())
    gerar_metadados(
        passo='passo2_eda_brutos',
        descricao='Análise exploratória dos dados brutos WDI e WGI',
        config=config,
        dados_entrada=[wdi_path, wgi_path],
        dados_saida=ficheiros,
        metricas={
            'wdi': registar_dataframe_info(df_wdi, 'WDI Bruto'),
            'wgi': registar_dataframe_info(df_wgi, 'WGI Bruto'),
        }
    )
    
    auto_save_drive(ficheiros, config)
    
    print("\n  ✓ PASSO 2 CONCLUÍDO")
    return df_wdi, df_wgi


if __name__ == '__main__':
    executar_passo2()
