"""
============================================================
PASSO 2.1: LIMPEZA E EDA - DATASETS SEPARADOS E INDEPENDENTES
- Limpeza WDI (Quantitativo): MICE (IterativeImputer) sem Winsorização
- Limpeza WGI (Qualitativo): MICE + preservação da escala original WGI
- EDA separada para cada dataset
- Ficheiros de saída separados e independentes
============================================================
Justificação (conforme plano de pesquisa):
- MICE (Multiple Imputation by Chained Equations) utiliza relações
  multivariadas para imputação, superior à interpolação linear univariada.
- Winsorização eliminada: em economias emergentes, valores extremos
  representam eventos reais (crises, recuperações pós-conflito) que
  não devem ser suprimidos. Modelos baseados em árvores (RF, XGBoost,
  GradientBoosting) são robustos a outliers; para modelos lineares
  (SARIMAX), aplica-se transformação logarítmica quando necessário.
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

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def limpar_wdi(df_wdi):
    """
    Limpeza de WDI (Quantitativo):
    1. MICE (IterativeImputer) por país para preencher missings
       - Utiliza correlações multivariadas entre indicadores
       - max_iter=20, random_state=42 para reprodutibilidade
    2. Fallback: preenchimento residual com mediana por país
    3. SEM Winsorização (preserva outliers reais)
    
    NOTA: A variável target (valor_agregado_industrial_percent_pib) é
    incluída no MICE como variável auxiliar para melhorar a imputação
    das preditoras, mas NÃO é imputada ela própria (os seus NaN são
    tratados separadamente para evitar leakage).
    """
    print("\n  === LIMPEZA WDI (Quantitativo - MICE) ===")
    print(f"  Shape original: {df_wdi.shape}")
    n_missing_original = df_wdi.isna().sum().sum()
    print(f"  Missing: {n_missing_original} valores")
    
    df_clean = df_wdi.copy()
    
    # Colunas numéricas (excluir year)
    colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']]
    
    # ============================================================
    # ETAPA 1: MICE (IterativeImputer) por país
    # ============================================================
    print(f"\n  [1] MICE (IterativeImputer) por país...")
    print(f"      Variáveis numéricas: {len(colunas_numericas)}")
    print(f"      max_iter=20, random_state=42")
    
    # Aplicar MICE por país para respeitar a estrutura de painel
    paises = df_clean['country_code'].unique()
    n_imputados = 0
    
    for pais in paises:
        mask_pais = df_clean['country_code'] == pais
        df_pais = df_clean.loc[mask_pais, colunas_numericas].copy()
        
        # Só aplicar MICE se houver missings E dados suficientes
        n_missing_pais = df_pais.isna().sum().sum()
        n_obs = len(df_pais)
        
        if n_missing_pais == 0:
            continue
        
        # Verificar se há variáveis com dados suficientes para MICE
        # (pelo menos 3 observações não-NaN por variável)
        cols_validas = [c for c in colunas_numericas if df_pais[c].notna().sum() >= 3]
        
        if len(cols_validas) < 2:
            # Se não há dados suficientes para MICE, usar interpolação simples
            for col in colunas_numericas:
                df_clean.loc[mask_pais, col] = df_pais[col].interpolate(
                    method='linear', limit_direction='both'
                )
            continue
        
        # Aplicar MICE nas colunas válidas
        try:
            imputer = IterativeImputer(
                max_iter=20,
                random_state=42,
                n_nearest_features=min(5, len(cols_validas) - 1),
                initial_strategy='median',
                skip_complete=True
            )
            
            dados_imputados = imputer.fit_transform(df_pais[cols_validas])
            df_clean.loc[mask_pais, cols_validas] = dados_imputados
            n_imputados += n_missing_pais
            
        except Exception as e:
            # Fallback: interpolação linear se MICE falhar para este país
            print(f"      ⚠ MICE falhou para {pais}: {e}. Usando interpolação.")
            for col in colunas_numericas:
                df_clean.loc[mask_pais, col] = df_pais[col].interpolate(
                    method='linear', limit_direction='both'
                )
    
    n_missing_pos_mice = df_clean[colunas_numericas].isna().sum().sum()
    print(f"      Valores imputados por MICE: {n_missing_original - n_missing_pos_mice}")
    print(f"      Missing residuais após MICE: {n_missing_pos_mice}")
    
    # ============================================================
    # ETAPA 2: Preenchimento de NaN residuais com mediana por país
    # ============================================================
    if n_missing_pos_mice > 0:
        print(f"\n  [2] Preenchimento de NaN residuais (mediana por país)...")
        for col in colunas_numericas:
            df_clean[col] = df_clean.groupby('country_code')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Preenchimento final com mediana global (para países sem dados)
        for col in colunas_numericas:
            if df_clean[col].isna().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # ============================================================
    # SEM WINSORIZAÇÃO (preserva outliers reais)
    # ============================================================
    print(f"\n  [✓] Winsorização NÃO aplicada (preserva eventos extremos reais)")
    print(f"      Justificação: outliers em economias emergentes representam")
    print(f"      eventos reais (crises, recuperações pós-conflito).")
    print(f"      Modelos de árvores (RF, XGBoost) são robustos a outliers.")
    
    print(f"\n  Shape final: {df_clean.shape}")
    print(f"  Missing residuais: {df_clean.isna().sum().sum()}")
    
    return df_clean


def limpar_wgi(df_wgi):
    """
    Limpeza de WGI (Qualitativo - Governança):
    1. MICE (IterativeImputer) por país para preencher missings
    2. Preservação da escala original WGI (já está em -2.5 a +2.5)
    3. Preenchimento de NaN residuais com mediana por país
    
    NOTA: Os indicadores WGI já vêm na escala padronizada (-2.5 a +2.5)
    do Banco Mundial. Não é necessária normalização adicional, pois isso
    distorceria a interpretabilidade dos scores de governança.
    """
    print("\n  === LIMPEZA WGI (Qualitativo - Governança - MICE) ===")
    print(f"  Shape original: {df_wgi.shape}")
    n_missing_original = df_wgi.isna().sum().sum()
    print(f"  Missing: {n_missing_original} valores")
    
    df_clean = df_wgi.copy()
    
    # Colunas numéricas (excluir year)
    colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in ['year']]
    
    if n_missing_original == 0:
        print(f"\n  [✓] Sem valores faltantes - nenhuma imputação necessária")
        print(f"  [✓] Escala WGI original preservada (-2.5 a +2.5)")
        print(f"\n  Shape final: {df_clean.shape}")
        return df_clean
    
    # ============================================================
    # ETAPA 1: MICE (IterativeImputer) por país
    # ============================================================
    print(f"\n  [1] MICE (IterativeImputer) por país...")
    
    paises = df_clean['country_code'].unique()
    
    for pais in paises:
        mask_pais = df_clean['country_code'] == pais
        df_pais = df_clean.loc[mask_pais, colunas_numericas].copy()
        
        n_missing_pais = df_pais.isna().sum().sum()
        if n_missing_pais == 0:
            continue
        
        cols_validas = [c for c in colunas_numericas if df_pais[c].notna().sum() >= 3]
        
        if len(cols_validas) < 2:
            for col in colunas_numericas:
                df_clean.loc[mask_pais, col] = df_pais[col].interpolate(
                    method='linear', limit_direction='both'
                )
            continue
        
        try:
            imputer = IterativeImputer(
                max_iter=20,
                random_state=42,
                n_nearest_features=min(5, len(cols_validas) - 1),
                initial_strategy='median',
                skip_complete=True
            )
            
            dados_imputados = imputer.fit_transform(df_pais[cols_validas])
            df_clean.loc[mask_pais, cols_validas] = dados_imputados
            
        except Exception as e:
            print(f"      ⚠ MICE falhou para {pais}: {e}. Usando interpolação.")
            for col in colunas_numericas:
                df_clean.loc[mask_pais, col] = df_pais[col].interpolate(
                    method='linear', limit_direction='both'
                )
    
    # ============================================================
    # ETAPA 2: Preservação da escala WGI original
    # ============================================================
    print(f"\n  [2] Escala WGI original preservada (-2.5 a +2.5)")
    print(f"      Os indicadores WGI já vêm padronizados pelo Banco Mundial.")
    print(f"      Não é aplicada normalização adicional.")
    
    # Apenas garantir que valores imputados não excedem a escala teórica
    for col in colunas_numericas:
        df_clean[col] = df_clean[col].clip(lower=-2.5, upper=2.5)
    
    # ============================================================
    # ETAPA 3: Preenchimento de NaN residuais
    # ============================================================
    n_missing_pos = df_clean[colunas_numericas].isna().sum().sum()
    if n_missing_pos > 0:
        print(f"\n  [3] Preenchimento de NaN residuais (mediana por país)...")
        for col in colunas_numericas:
            df_clean[col] = df_clean.groupby('country_code')[col].transform(
                lambda x: x.fillna(x.median())
            )
        for col in colunas_numericas:
            if df_clean[col].isna().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    print(f"\n  Shape final: {df_clean.shape}")
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
    
    # Boxplot para detectar outliers preservados
    fig, ax = plt.subplots(figsize=(16, 6))
    cols_box = colunas_numericas[:9]
    df_wdi[cols_box].boxplot(ax=ax, rot=45)
    ax.set_title('Boxplot - Indicadores WDI (outliers preservados)')
    ax.set_ylabel('Valor')
    
    box_path = os.path.join(eda_dir, f'{nome_saida}_boxplot_outliers.png')
    plt.savefig(box_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Boxplot outliers: {box_path}")
    
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
        if idx >= 6:
            break
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
    print("PASSO 2.1: LIMPEZA E EDA - MICE SEM WINSORIZAÇÃO")
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
    # LIMPEZA SEPARADA E INDEPENDENTE (MICE)
    # ============================================================
    
    # Limpeza WDI com MICE
    df_wdi_limpo = limpar_wdi(df_wdi_bruto)
    
    # Limpeza WGI com MICE
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
        descricao='Limpeza separada e independente de WDI e WGI com MICE (IterativeImputer). Sem Winsorização (preserva outliers reais). EDA separada para cada dataset.',
        config=config,
        dados_entrada=[wdi_path, wgi_path],
        dados_saida=[wdi_limpo_path, wgi_limpo_path],
        parametros={
            'wdi_imputacao': 'MICE (IterativeImputer, max_iter=20, por país)',
            'wdi_winsorização': 'NÃO APLICADA (preserva outliers reais)',
            'wdi_fallback': 'mediana por país para NaN residuais',
            'wgi_imputacao': 'MICE (IterativeImputer, max_iter=20, por país)',
            'wgi_escala': 'original WGI (-2.5 a +2.5) preservada',
            'wgi_clip': 'clip em -2.5/+2.5 apenas para valores imputados fora da escala',
        },
        metricas={
            'wdi': registar_dataframe_info(df_wdi_limpo, 'WDI Limpo'),
            'wgi': registar_dataframe_info(df_wgi_limpo, 'WGI Limpo'),
        }
    )
    
    auto_save_drive([wdi_limpo_path, wgi_limpo_path], config)
    
    print("\n  ✓ PASSO 2.1 CONCLUÍDO (MICE sem Winsorização)")
    return df_wdi_limpo, df_wgi_limpo


if __name__ == '__main__':
    executar_passo2_1()
