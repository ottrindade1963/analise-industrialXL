"""
============================================================
PASSO 3: ENGENHARIA DE FEATURES - 4 DATASETS INDEPENDENTES
============================================================
Conforme plano de pesquisa (secção 6.4), são implementadas 3 estratégias
de agregação das variáveis qualitativas WGI:

  A1 - Inclusão directa: 6 variáveis WGI na escala original
  A2 - Fator latente (PCA): 1º componente principal dos 6 WGI
  A3 - Termos de interação: produtos WGI × variáveis quantitativas

Para cada dataset, são geradas TODAS as estratégias em paralelo,
permitindo comparação posterior nos modelos (secção 6.5 do plano).

Transformações aplicadas:
- Lags (1,2,3 para WGI/PCA; 1 para WDI; 1,2 para target)
- Médias móveis (3 anos para WGI/PCA)
- Deltas (diferenças para WGI/PCA)
- Log-retornos (4 variáveis principais)
- Interações A3 (4 pares conforme H2)
- PCA A2 (1º componente principal dos 6 WGI)

Visualizações geradas (relevantes ao plano de pesquisa):
- Variância explicada pelo PCA (justifica redução dimensional)
- Loadings do PCA (identifica quais WGI dominam o fator latente)
- Correlação features vs target (valida poder preditivo)
- Distribuição do componente PCA vs WGI originais
- Heatmap de multicolinearidade WGI (justifica necessidade de PCA)
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


# ============================================================
# FUNÇÕES DE ENGENHARIA DE FEATURES
# ============================================================

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
    """Cria termos de interação defasados (Estratégia A3)."""
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


def aplicar_pca_wgi(df, colunas_wgi, nome_dataset, viz_dir=None):
    """
    Aplica PCA aos 6 indicadores WGI (Estratégia A2 do plano de pesquisa).
    
    Extrai o 1º componente principal que representa a "qualidade institucional
    latente". Este componente substitui as 6 variáveis WGI originais para
    reduzir dimensionalidade e mitigar multicolinearidade (correlações >0.7).
    
    Returns:
        df: DataFrame com coluna 'wgi_pca1' adicionada
        pca_info: dicionário com informações do PCA (loadings, variância, etc.)
    """
    # Filtrar apenas colunas WGI presentes no dataset
    cols_presentes = [c for c in colunas_wgi if c in df.columns]
    
    if len(cols_presentes) < 2:
        print(f"    ⚠ PCA não aplicável: apenas {len(cols_presentes)} variáveis WGI presentes")
        return df, None
    
    # Extrair dados WGI (sem NaN)
    dados_wgi = df[cols_presentes].copy()
    mask_validos = dados_wgi.notna().all(axis=1)
    
    if mask_validos.sum() < 10:
        print(f"    ⚠ PCA não aplicável: apenas {mask_validos.sum()} observações válidas")
        return df, None
    
    # Padronizar antes do PCA
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados_wgi[mask_validos])
    
    # Aplicar PCA
    pca = PCA(n_components=min(len(cols_presentes), 3))
    componentes = pca.fit_transform(dados_padronizados)
    
    # Adicionar 1º componente principal ao DataFrame
    df['wgi_pca1'] = np.nan
    df.loc[mask_validos, 'wgi_pca1'] = componentes[:, 0]
    
    # Preencher NaN do PCA com interpolação por país
    df['wgi_pca1'] = df.groupby('country_code')['wgi_pca1'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    df['wgi_pca1'].fillna(0, inplace=True)
    
    # Informações do PCA
    pca_info = {
        'variancia_explicada': pca.explained_variance_ratio_.tolist(),
        'variancia_acumulada': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'loadings': pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=cols_presentes
        ),
        'n_componentes': pca.n_components_,
        'colunas_wgi': cols_presentes,
    }
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    print(f"    ✓ PCA aplicado: PC1 explica {var_pc1:.1f}% da variância")
    print(f"      Variância acumulada (PC1-PC3): {[f'{v*100:.1f}%' for v in pca_info['variancia_acumulada']]}")
    
    return df, pca_info


def criar_interacoes_pca(df):
    """
    Cria termos de interação usando o componente PCA (Estratégia A2+A3).
    Conforme recomendação: inter_pca1_ied, inter_pca1_formacao, etc.
    """
    novas_cols = []
    
    if 'wgi_pca1' not in df.columns:
        return df, novas_cols
    
    pca_lag = 'wgi_pca1_lag1' if 'wgi_pca1_lag1' in df.columns else 'wgi_pca1'
    
    pares_pca = [
        (pca_lag, 'ied_percent_pib'),
        (pca_lag, 'formacao_bruta_capital_fixo_percent_pib'),
        (pca_lag, 'comercio_percent_pib'),
        (pca_lag, 'pib_per_capita_ppc'),
    ]
    
    for col_pca, var_quant in pares_pca:
        col_quant = f'{var_quant}_lag1' if f'{var_quant}_lag1' in df.columns else var_quant
        
        if col_pca not in df.columns or col_quant not in df.columns:
            continue
        
        nome_inter = f'inter_pca1_{var_quant.split("_")[0]}'
        
        std_pca = df[col_pca].std()
        std_quant = df[col_quant].std()
        
        if std_pca > 0 and std_quant > 0:
            df[nome_inter] = (df[col_pca] / std_pca) * (df[col_quant] / std_quant)
        else:
            df[nome_inter] = df[col_pca] * df[col_quant]
        
        novas_cols.append(nome_inter)
    
    return df, novas_cols


# ============================================================
# VISUALIZAÇÕES (relevantes ao plano de pesquisa)
# ============================================================

def gerar_visualizacoes(df, pca_info, nome_dataset, viz_dir):
    """
    Gera visualizações relevantes para o plano de pesquisa:
    1. Multicolinearidade WGI (justifica PCA)
    2. Variância explicada pelo PCA
    3. Loadings do PCA (quais WGI dominam o fator latente)
    4. Distribuição PCA vs WGI originais
    5. Correlação features vs target (poder preditivo)
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    colunas_wgi = pca_info['colunas_wgi'] if pca_info else []
    
    # --------------------------------------------------------
    # 1. Heatmap de multicolinearidade WGI (justifica PCA)
    # --------------------------------------------------------
    if len(colunas_wgi) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        cols_presentes = [c for c in colunas_wgi if c in df.columns]
        corr_wgi = df[cols_presentes].corr()
        
        # Nomes simplificados para legibilidade
        nomes_curtos = [c.replace('wgi_', '').replace('_', ' ').title() for c in cols_presentes]
        
        mask = np.triu(np.ones_like(corr_wgi, dtype=bool), k=1)
        sns.heatmap(corr_wgi, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                    xticklabels=nomes_curtos, yticklabels=nomes_curtos,
                    mask=mask, ax=ax, vmin=-1, vmax=1,
                    square=True, linewidths=0.5)
        ax.set_title(f'Multicolinearidade WGI - {nome_dataset}\n'
                     f'(Correlações >0.7 justificam redução por PCA)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(viz_dir, f'{nome_dataset}_multicolinearidade_wgi.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Multicolinearidade WGI: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 2. Variância explicada pelo PCA (scree plot)
    # --------------------------------------------------------
    if pca_info:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        n_comp = len(pca_info['variancia_explicada'])
        x = range(1, n_comp + 1)
        
        # Barras de variância individual
        ax1.bar(x, [v*100 for v in pca_info['variancia_explicada']], 
                color='steelblue', edgecolor='black', alpha=0.8)
        ax1.plot(x, [v*100 for v in pca_info['variancia_acumulada']], 
                 'ro-', linewidth=2, markersize=8, label='Acumulada')
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Limiar 70%')
        ax1.set_xlabel('Componente Principal')
        ax1.set_ylabel('Variância Explicada (%)')
        ax1.set_title('Scree Plot - PCA dos Indicadores WGI')
        ax1.set_xticks(list(x))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loadings do PC1
        loadings = pca_info['loadings']['PC1'].sort_values(ascending=True)
        nomes_curtos = [c.replace('wgi_', '').replace('_', ' ').title() for c in loadings.index]
        
        colors = ['#d32f2f' if v < 0 else '#1976d2' for v in loadings.values]
        ax2.barh(nomes_curtos, loadings.values, color=colors, edgecolor='black', alpha=0.8)
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Loading no PC1')
        ax2.set_title('Contribuição de cada WGI ao Fator Latente (PC1)\n'
                      '(Identifica preditores qualitativos dominantes - H5)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        path = os.path.join(viz_dir, f'{nome_dataset}_pca_variancia_loadings.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ PCA variância + loadings: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 3. Distribuição PCA vs WGI originais
    # --------------------------------------------------------
    if pca_info and 'wgi_pca1' in df.columns and len(colunas_wgi) >= 2:
        n_wgi = min(len(colunas_wgi), 6)
        fig, axes = plt.subplots(1, n_wgi + 1, figsize=(3.5 * (n_wgi + 1), 4))
        
        # PCA
        axes[0].hist(df['wgi_pca1'].dropna(), bins=30, color='darkorange', 
                     edgecolor='black', alpha=0.8)
        axes[0].set_title('wgi_pca1\n(Fator Latente)', fontsize=9, fontweight='bold')
        axes[0].set_xlabel('Valor')
        axes[0].set_ylabel('Frequência')
        
        # WGI originais
        for i, col in enumerate(colunas_wgi[:n_wgi]):
            if col in df.columns:
                axes[i+1].hist(df[col].dropna(), bins=30, color='steelblue', 
                               edgecolor='black', alpha=0.7)
                nome_curto = col.replace('wgi_', '').replace('_', '\n')
                axes[i+1].set_title(f'{nome_curto}', fontsize=8)
                axes[i+1].set_xlabel('Valor')
        
        plt.suptitle(f'Distribuição: PCA vs WGI Originais - {nome_dataset}', 
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(viz_dir, f'{nome_dataset}_distribuicao_pca_vs_wgi.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Distribuição PCA vs WGI: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 4. Correlação features vs target (poder preditivo)
    # --------------------------------------------------------
    if config.TARGET_VAR in df.columns:
        # Seleccionar features mais relevantes para correlação
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != config.TARGET_VAR and c != 'year'
                        and df[c].notna().sum() > 10]
        
        if len(feature_cols) > 0:
            correlacoes = df[feature_cols].corrwith(df[config.TARGET_VAR]).dropna()
            correlacoes = correlacoes.abs().sort_values(ascending=False)
            
            # Top 25 features
            top_n = min(25, len(correlacoes))
            top_corr = correlacoes.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Colorir por tipo de feature
            cores = []
            for feat in top_corr.index:
                if 'pca' in feat:
                    cores.append('#ff6f00')  # Laranja para PCA
                elif 'wgi_' in feat:
                    cores.append('#1976d2')  # Azul para WGI directos
                elif 'inter_' in feat:
                    cores.append('#388e3c')  # Verde para interações
                else:
                    cores.append('#616161')  # Cinza para quantitativos
            
            ax.barh(range(top_n), top_corr.values, color=cores, edgecolor='black', alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f.replace('_', ' ')[:35] for f in top_corr.index], fontsize=8)
            ax.set_xlabel('|Correlação| com Target')
            ax.set_title(f'Top {top_n} Features por Correlação com {config.TARGET_VAR}\n'
                         f'(Laranja=PCA, Azul=WGI, Verde=Interação, Cinza=Quantitativo)',
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            plt.tight_layout()
            path = os.path.join(viz_dir, f'{nome_dataset}_correlacao_features_target.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Correlação features vs target: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 5. Comparação de dimensionalidade: A1 vs A2
    # --------------------------------------------------------
    if pca_info:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        n_wgi_originais = len(colunas_wgi)
        n_derivadas_a1 = n_wgi_originais * (1 + 3 + 1 + 1)  # original + 3 lags + ma + delta
        n_derivadas_a2 = 1 * (1 + 3 + 1 + 1)  # pca1 + 3 lags + ma + delta
        
        categorias = ['A1 - Inclusão\nDirecta', 'A2 - Fator\nLatente (PCA)']
        valores = [n_derivadas_a1, n_derivadas_a2]
        reducao = (1 - n_derivadas_a2 / n_derivadas_a1) * 100
        
        bars = ax.bar(categorias, valores, color=['#1976d2', '#ff6f00'], 
                      edgecolor='black', alpha=0.8, width=0.5)
        
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val} features', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Número de Features WGI-derivadas')
        ax.set_title(f'Redução de Dimensionalidade: A1 vs A2\n'
                     f'(Redução de {reducao:.0f}% - parcimónia para séries curtas de 25 anos)',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(valores) * 1.3)
        
        plt.tight_layout()
        path = os.path.join(viz_dir, f'{nome_dataset}_reducao_dimensionalidade_a1_vs_a2.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Redução dimensionalidade A1 vs A2: {os.path.basename(path)}")


# ============================================================
# FUNÇÃO PRINCIPAL DE ENGENHARIA DE FEATURES
# ============================================================

def engenharia_features(df, nome_dataset, viz_dir=None):
    """
    Aplica engenharia de features a um dataset, incluindo as 3 estratégias
    do plano de pesquisa (secção 6.4):
      A1 - Inclusão directa dos WGI + lags/MA/deltas
      A2 - PCA (fator latente) + lags/MA/deltas do PC1
      A3 - Termos de interação (WGI × quantitativos)
    """
    print(f"\n  === Engenharia de Features: {nome_dataset} ===")
    print(f"  Shape original: {df.shape}")
    
    # Identificar tipos de variáveis
    todas_colunas = df.columns.tolist()
    colunas_wgi = [c for c in todas_colunas if 'wgi_' in c and '_lag' not in c 
                   and '_ma' not in c and '_delta' not in c and '_pca' not in c]
    colunas_quant = [c for c in df.select_dtypes(include=[np.number]).columns 
                     if c not in colunas_wgi and c != 'year' and c != config.TARGET_VAR
                     and '_lag' not in c and '_ma' not in c and '_delta' not in c 
                     and '_log_ret' not in c and 'inter_' not in c and '_pca' not in c]
    
    features_criadas = {}
    pca_info = None
    
    # ============================================================
    # ESTRATÉGIA A2: PCA nos indicadores WGI (fator latente)
    # ============================================================
    print(f"\n  [PCA] Estratégia A2 - Fator Latente...")
    if len(colunas_wgi) >= 2:
        df, pca_info = aplicar_pca_wgi(df, colunas_wgi, nome_dataset, viz_dir)
        if pca_info:
            features_criadas['pca'] = ['wgi_pca1']
    else:
        print(f"    (Sem variáveis WGI - PCA não aplicável a este dataset)")
    
    # ============================================================
    # ESTRATÉGIA A1: Lags, MA, Deltas dos WGI originais
    # ============================================================
    
    # 1. Lags das variáveis qualitativas (1, 2, 3)
    df, lags_wgi = criar_lags(df, colunas_wgi, config.LAGS_QUALITATIVOS)
    features_criadas['lags_wgi'] = lags_wgi
    print(f"  [1] Lags WGI A1 (1,2,3): {len(lags_wgi)} features")
    
    # 2. Lags do PCA (1, 2, 3) - Estratégia A2
    if 'wgi_pca1' in df.columns:
        df, lags_pca = criar_lags(df, ['wgi_pca1'], config.LAGS_QUALITATIVOS)
        features_criadas['lags_pca'] = lags_pca
        print(f"  [2] Lags PCA A2 (1,2,3): {len(lags_pca)} features")
    
    # 3. Lags das variáveis quantitativas (1)
    df, lags_quant = criar_lags(df, colunas_quant, config.LAGS_QUANTITATIVOS)
    features_criadas['lags_quant'] = lags_quant
    print(f"  [3] Lags quantitativos (1): {len(lags_quant)} features")
    
    # 4. Lags da variável alvo (1, 2)
    df, lags_target = criar_lags(df, [config.TARGET_VAR], config.LAGS_TARGET)
    features_criadas['lags_target'] = lags_target
    print(f"  [4] Lags target (1,2): {len(lags_target)} features")
    
    # 5. Médias móveis das variáveis qualitativas (3 anos) - A1
    df, ma_wgi = criar_medias_moveis(df, colunas_wgi, config.JANELA_MEDIA_MOVEL)
    features_criadas['ma_wgi'] = ma_wgi
    print(f"  [5] Médias móveis WGI A1 (3 anos): {len(ma_wgi)} features")
    
    # 6. Média móvel do PCA (3 anos) - A2
    if 'wgi_pca1' in df.columns:
        df, ma_pca = criar_medias_moveis(df, ['wgi_pca1'], config.JANELA_MEDIA_MOVEL)
        features_criadas['ma_pca'] = ma_pca
        print(f"  [6] Média móvel PCA A2 (3 anos): {len(ma_pca)} features")
    
    # 7. Deltas das variáveis qualitativas - A1
    df, deltas_wgi = criar_deltas(df, colunas_wgi)
    features_criadas['deltas_wgi'] = deltas_wgi
    print(f"  [7] Deltas WGI A1: {len(deltas_wgi)} features")
    
    # 8. Delta do PCA - A2
    if 'wgi_pca1' in df.columns:
        df, deltas_pca = criar_deltas(df, ['wgi_pca1'])
        features_criadas['deltas_pca'] = deltas_pca
        print(f"  [8] Delta PCA A2: {len(deltas_pca)} features")
    
    # 9. Log-retornos
    vars_log_ret = ['pib_per_capita_ppc', 'formacao_bruta_capital_fixo_percent_pib',
                    'comercio_percent_pib', 'ied_percent_pib']
    vars_log_ret = [v for v in vars_log_ret if v in df.columns]
    df, log_rets = criar_log_retornos(df, vars_log_ret)
    features_criadas['log_retornos'] = log_rets
    print(f"  [9] Log-retornos: {len(log_rets)} features")
    
    # 10. Interações A3 (WGI × quantitativos)
    df, interacoes = criar_interacoes(df, config.INTERACOES)
    features_criadas['interacoes_a3'] = interacoes
    print(f"  [10] Interações A3 (H2): {len(interacoes)} features")
    
    # 11. Interações PCA (A2+A3 combinada)
    if 'wgi_pca1' in df.columns:
        df, interacoes_pca = criar_interacoes_pca(df)
        features_criadas['interacoes_pca'] = interacoes_pca
        print(f"  [11] Interações PCA (A2+A3): {len(interacoes_pca)} features")
    
    # ============================================================
    # LIMPEZA FINAL
    # ============================================================
    
    # Remover linhas com NaN dos primeiros lags
    # Usa o ano mínimo do próprio dataset (não o global ANO_INICIO)
    # para funcionar tanto com dados reais como sintéticos
    max_lag = max(config.LAGS_QUALITATIVOS)
    ano_min_dataset = int(df['year'].min())
    df_clean = df[df['year'] >= ano_min_dataset + max_lag].copy()
    
    # Imputar NaN residuais com 0
    feature_cols = []
    for lista in features_criadas.values():
        feature_cols.extend(lista)
    
    for col in feature_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    total_features = sum(len(v) for v in features_criadas.values())
    print(f"\n  Total features criadas: {total_features}")
    print(f"  Shape final: {df_clean.shape}")
    
    # Resumo por estratégia
    n_a1 = len(features_criadas.get('lags_wgi', [])) + len(features_criadas.get('ma_wgi', [])) + len(features_criadas.get('deltas_wgi', []))
    n_a2 = len(features_criadas.get('pca', [])) + len(features_criadas.get('lags_pca', [])) + len(features_criadas.get('ma_pca', [])) + len(features_criadas.get('deltas_pca', [])) + len(features_criadas.get('interacoes_pca', []))
    n_a3 = len(features_criadas.get('interacoes_a3', []))
    print(f"  Estratégia A1 (inclusão directa): {n_a1} features WGI-derivadas")
    print(f"  Estratégia A2 (fator latente PCA): {n_a2} features PCA-derivadas")
    print(f"  Estratégia A3 (interações): {n_a3} features de interação")
    
    # ============================================================
    # VISUALIZAÇÕES
    # ============================================================
    if viz_dir and pca_info:
        print(f"\n  Gerando visualizações...")
        gerar_visualizacoes(df_clean, pca_info, nome_dataset, viz_dir)
    
    return df_clean, features_criadas, pca_info


def executar_passo3():
    """Executa o Passo 3 completo - Engenharia em 4 datasets."""
    print("\n" + "=" * 70)
    print("PASSO 3: ENGENHARIA DE FEATURES - 3 ESTRATÉGIAS (A1, A2, A3)")
    print("=" * 70)
    
    os.makedirs(config.DADOS_ENGENHARIA_DIR, exist_ok=True)
    
    # Directório de visualizações do Passo 3
    viz_dir = os.path.join(config.BASE_DIR, 'eda_engenharia')
    os.makedirs(viz_dir, exist_ok=True)
    
    pca_infos = {}
    
    # ============================================================
    # DATASET 1: WDI LIMPO (NÃO AGREGADO)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 1: WDI LIMPO (NÃO AGREGADO)")
    print("-" * 70)
    
    wdi_limpo_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wdi_limpo.csv')
    df_wdi_limpo = pd.read_csv(wdi_limpo_path)
    
    df_wdi_eng, features_wdi, _ = engenharia_features(df_wdi_limpo.copy(), 'wdi_limpo')
    
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
    
    df_agregado_eng, features_agregado, pca_info_agreg = engenharia_features(
        df_agregado.copy(), 'agregado', viz_dir=viz_dir
    )
    pca_infos['agregado'] = pca_info_agreg
    
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
    
    df_sintetico_eng, features_sintetico, pca_info_sint = engenharia_features(
        df_sintetico.copy(), 'sintetico', viz_dir=viz_dir
    )
    pca_infos['sintetico'] = pca_info_sint
    
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
    
    df_wdi_sint_eng, features_wdi_sint, _ = engenharia_features(
        df_wdi_sintetico.copy(), 'wdi_sintetico'
    )
    
    wdi_sint_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv')
    df_wdi_sint_eng.to_csv(wdi_sint_eng_path, index=False)
    print(f"  ✓ WDI Sintético com features: {wdi_sint_eng_path}")
    
    # ============================================================
    # SALVAR INFORMAÇÕES DO PCA
    # ============================================================
    
    # Salvar loadings do PCA como CSV para referência
    for nome, info in pca_infos.items():
        if info:
            loadings_path = os.path.join(config.DADOS_ENGENHARIA_DIR, f'{nome}_pca_loadings.csv')
            info['loadings'].to_csv(loadings_path)
            print(f"\n  ✓ PCA loadings ({nome}): {loadings_path}")
    
    # ============================================================
    # METADADOS
    # ============================================================
    
    ficheiros_saida = [wdi_eng_path, agregado_eng_path, sintetico_eng_path, wdi_sint_eng_path]
    
    # Informações do PCA para metadados
    pca_params = {}
    for nome, info in pca_infos.items():
        if info:
            pca_params[f'pca_{nome}_var_pc1'] = f"{info['variancia_explicada'][0]*100:.1f}%"
            pca_params[f'pca_{nome}_var_acumulada'] = [f"{v*100:.1f}%" for v in info['variancia_acumulada']]
    
    gerar_metadados(
        passo='passo3_engenharia_features',
        descricao='Engenharia de features com 3 estratégias (A1: inclusão directa WGI, A2: fator latente PCA, A3: interações). PCA aplicado aos 6 indicadores WGI para redução dimensional. Lags, MA, deltas, log-retornos e interações para todas as estratégias.',
        config=config,
        dados_entrada=[wdi_limpo_path, agregado_path, sintetico_path, wdi_sintetico_path],
        dados_saida=ficheiros_saida,
        parametros={
            'estrategias': ['A1 - Inclusão directa', 'A2 - Fator latente (PCA)', 'A3 - Termos de interação'],
            'lags_qualitativos': config.LAGS_QUALITATIVOS,
            'lags_quantitativos': config.LAGS_QUANTITATIVOS,
            'lags_target': config.LAGS_TARGET,
            'janela_media_movel': config.JANELA_MEDIA_MOVEL,
            'interacoes_a3': [f'{a}×{b}' for a, b in config.INTERACOES],
            'pca_n_componentes': 1,
            **pca_params,
        },
        metricas={
            'wdi_features': registar_dataframe_info(df_wdi_eng, 'WDI com Features'),
            'agregado_features': registar_dataframe_info(df_agregado_eng, 'Agregado com Features'),
            'sintetico_agregado_features': registar_dataframe_info(df_sintetico_eng, 'Sintético Agregado com Features'),
            'wdi_sintetico_features': registar_dataframe_info(df_wdi_sint_eng, 'WDI Sintético com Features'),
        }
    )
    
    auto_save_drive(ficheiros_saida, config)
    
    print("\n  ✓ PASSO 3 CONCLUÍDO (com PCA e visualizações)")
    return df_wdi_eng, df_agregado_eng, df_sintetico_eng, df_wdi_sint_eng


if __name__ == '__main__':
    executar_passo3()
