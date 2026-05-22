"""
============================================================
PASSO 3: ENGENHARIA DE FEATURES - ESTRATÉGIA PARCIMONIOSA
============================================================
Conforme plano de pesquisa (secção 6.4) e recomendações de parcimónia
para séries curtas (25 anos por país):

  ESTRATÉGIA PRINCIPAL: A2 - Fator Latente (PCA)
    - 1º componente principal dos 6 WGI ("qualidade institucional latente")
    - Substitui completamente os 6 WGI originais (não coexistem)
    - Redução de ~83% na dimensionalidade WGI-derivada

  COMPLEMENTO: Interações PCA × Quantitativos
    - Termos de interação entre PC1 e variáveis económicas (H2)

  REMOVIDO (A1): Inclusão directa dos 6 WGI + lags/MA/deltas
    - Causava 30+ features para apenas 25 anos → overfitting certo

Princípios de design:
  1. PCA ajustado APENAS nos dados de treino (80% primeiros anos por país)
     → transformação aplicada ao teste → sem data leakage
  2. Lags reduzidos: máximo lag=2 (com 25 anos, lag3 é ruído)
  3. Parcimónia: ~25-30 features totais (ratio observações/features > 3)
  4. Sintético marcado como "apenas robustez" (não para avaliação preditiva)

Visualizações geradas:
  - Multicolinearidade WGI (justifica PCA)
  - Scree Plot + Loadings (variância explicada)
  - Correlação features vs target (poder preditivo)
  - Distribuição PCA vs WGI originais
  - Resumo de parcimónia (dimensionalidade final)
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
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


def criar_interacoes_pca(df):
    """
    Cria termos de interação usando o componente PCA (Estratégia A2 + H2).
    Interações: PC1_lag1 × variáveis económicas (IED, Formação Capital, Comércio, PIB).
    Padronizados para evitar problemas de escala.
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
# PCA COM SEPARAÇÃO TREINO/TESTE (SEM DATA LEAKAGE)
# ============================================================

def aplicar_pca_wgi_train_only(df, colunas_wgi, nome_dataset, pca_dir):
    """
    Aplica PCA aos 6 indicadores WGI com separação treino/teste:
      - FIT (ajuste) apenas nos 80% primeiros anos de cada país
      - TRANSFORM aplicado a todo o dataset
    
    Isto evita data leakage: o PC1 não "conhece" a variância de anos futuros.
    
    O scaler e o PCA são salvos em disco para reutilização no passo 4 (previsão).
    
    Returns:
        df: DataFrame com coluna 'wgi_pca1' adicionada
        pca_info: dicionário com informações do PCA
    """
    cols_presentes = [c for c in colunas_wgi if c in df.columns]
    
    if len(cols_presentes) < 2:
        print(f"    (Sem variáveis WGI suficientes - PCA não aplicável)")
        return df, None
    
    # Extrair dados WGI (sem NaN)
    dados_wgi = df[cols_presentes].copy()
    mask_validos = dados_wgi.notna().all(axis=1)
    
    if mask_validos.sum() < 10:
        print(f"    (Apenas {mask_validos.sum()} observações válidas - PCA não aplicável)")
        return df, None
    
    # -------------------------------------------------------
    # SEPARAÇÃO TREINO/TESTE POR PAÍS (80/20 temporal)
    # -------------------------------------------------------
    df_validos = df[mask_validos].copy()
    
    # Para cada país, os primeiros 80% dos anos são treino
    train_mask = pd.Series(False, index=df_validos.index)
    for pais in df_validos['country_code'].unique():
        idx_pais = df_validos[df_validos['country_code'] == pais].index
        n_treino = int(len(idx_pais) * 0.8)
        train_mask.loc[idx_pais[:n_treino]] = True
    
    dados_treino = df_validos.loc[train_mask, cols_presentes]
    
    print(f"    PCA ajustado em {len(dados_treino)} obs. de treino "
          f"({len(df_validos) - len(dados_treino)} obs. de teste)")
    
    # -------------------------------------------------------
    # FIT no treino, TRANSFORM em todo o dataset
    # -------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(dados_treino)
    
    # Transformar TODOS os dados válidos (treino + teste)
    dados_padronizados = scaler.transform(df_validos[cols_presentes])
    
    pca = PCA(n_components=min(len(cols_presentes), 3))
    pca.fit(scaler.transform(dados_treino))  # FIT apenas no treino
    
    # TRANSFORM em todo o dataset
    componentes = pca.transform(dados_padronizados)
    
    # Adicionar PC1 ao DataFrame
    df['wgi_pca1'] = np.nan
    df.loc[mask_validos, 'wgi_pca1'] = componentes[:, 0]
    
    # Preencher NaN residuais com interpolação por país
    df['wgi_pca1'] = df.groupby('country_code')['wgi_pca1'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    df['wgi_pca1'].fillna(0, inplace=True)
    
    # -------------------------------------------------------
    # SALVAR SCALER + PCA PARA REUTILIZAÇÃO NO PASSO 4
    # -------------------------------------------------------
    os.makedirs(pca_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(pca_dir, f'{nome_dataset}_pca_scaler.pkl'))
    joblib.dump(pca, os.path.join(pca_dir, f'{nome_dataset}_pca_model.pkl'))
    
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
        'n_treino': len(dados_treino),
        'n_total': mask_validos.sum(),
    }
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    print(f"    PC1 explica {var_pc1:.1f}% da variância")
    print(f"      Variância acumulada (PC1-PC3): "
          f"{[f'{v*100:.1f}%' for v in pca_info['variancia_acumulada']]}")
    
    return df, pca_info


# ============================================================
# VISUALIZAÇÕES (relevantes ao plano de pesquisa)
# ============================================================

def gerar_visualizacoes(df, pca_info, nome_dataset, viz_dir):
    """
    Gera visualizações relevantes para o plano de pesquisa:
    1. Multicolinearidade WGI (justifica PCA)
    2. Variância explicada pelo PCA (scree plot + loadings)
    3. Distribuição PCA vs WGI originais
    4. Correlação features vs target (poder preditivo)
    5. Resumo de parcimónia (dimensionalidade)
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    colunas_wgi = pca_info['colunas_wgi'] if pca_info else []
    
    # --------------------------------------------------------
    # 1. Heatmap de multicolinearidade WGI (justifica PCA)
    # --------------------------------------------------------
    if len(colunas_wgi) >= 2:
        cols_presentes = [c for c in colunas_wgi if c in df.columns]
        if cols_presentes:
            fig, ax = plt.subplots(figsize=(9, 7))
            corr_wgi = df[cols_presentes].corr()
            
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
            print(f"    Multicolinearidade WGI: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 2. Variância explicada pelo PCA (scree plot + loadings)
    # --------------------------------------------------------
    if pca_info:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        var_exp = [v * 100 for v in pca_info['variancia_explicada']]
        var_acum = [v * 100 for v in pca_info['variancia_acumulada']]
        x = range(1, len(var_exp) + 1)
        
        ax1.bar(x, var_exp, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.plot(x, var_acum, 'ro-', linewidth=2, markersize=8, label='Acumulada')
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
        print(f"    PCA variância + loadings: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 3. Distribuição PCA vs WGI originais
    # --------------------------------------------------------
    if pca_info and 'wgi_pca1' in df.columns:
        cols_presentes = [c for c in colunas_wgi if c in df.columns]
        n_cols = len(cols_presentes) + 1
        
        fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(16, 4))
        if not hasattr(axes, '__len__'):
            axes = [axes]
        
        # PC1
        axes[0].hist(df['wgi_pca1'].dropna(), bins=30, color='#ff6f00', 
                     edgecolor='black', alpha=0.8)
        axes[0].set_title('PC1 (Fator Latente)', fontweight='bold')
        axes[0].set_xlabel('Score')
        
        # Primeiros 3 WGI originais
        for i, col in enumerate(cols_presentes[:3]):
            nome_curto = col.replace('wgi_', '').replace('_', ' ').title()
            axes[i+1].hist(df[col].dropna(), bins=30, color='#1976d2', 
                          edgecolor='black', alpha=0.8)
            axes[i+1].set_title(f'{nome_curto} (original)')
            axes[i+1].set_xlabel('Score')
        
        plt.suptitle(f'Distribuição: PCA vs WGI Originais - {nome_dataset}', 
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(viz_dir, f'{nome_dataset}_distribuicao_pca_vs_wgi.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Distribuição PCA vs WGI: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 4. Correlação features vs target (poder preditivo)
    # --------------------------------------------------------
    if config.TARGET_VAR in df.columns:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in ['year', config.TARGET_VAR] and 'country' not in c]
        
        if feature_cols:
            correlacoes = df[feature_cols].corrwith(df[config.TARGET_VAR]).abs().sort_values(ascending=False)
            top_n = min(20, len(correlacoes))
            top_corr = correlacoes.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Cores por tipo de feature
            cores = []
            for feat in top_corr.index:
                if 'pca' in feat or 'inter_pca' in feat:
                    cores.append('#ff6f00')  # Laranja para PCA/interações PCA
                elif 'inter_' in feat:
                    cores.append('#388e3c')  # Verde para outras interações
                elif 'wgi_' in feat:
                    cores.append('#1976d2')  # Azul para WGI (se ainda presentes)
                else:
                    cores.append('#616161')  # Cinza para quantitativos
            
            ax.barh(range(top_n), top_corr.values, color=cores, edgecolor='black', alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f.replace('_', ' ')[:35] for f in top_corr.index], fontsize=8)
            ax.set_xlabel('|Correlação| com Target')
            ax.set_title(f'Top {top_n} Features por Correlação com {config.TARGET_VAR}\n'
                         f'(Laranja=PCA, Verde=Interação, Cinza=Quantitativo)',
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            plt.tight_layout()
            path = os.path.join(viz_dir, f'{nome_dataset}_correlacao_features_target.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Correlação features vs target: {os.path.basename(path)}")
    
    # --------------------------------------------------------
    # 5. Resumo de parcimónia (dimensionalidade)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Contar features por categoria
    todas_features = [c for c in df.columns if c not in ['country_code', 'year', config.TARGET_VAR]]
    n_pca = len([c for c in todas_features if 'pca' in c])
    n_quant = len([c for c in todas_features if 'pca' not in c and 'inter_' not in c and 'wgi_' not in c])
    n_inter = len([c for c in todas_features if 'inter_' in c])
    n_wgi_orig = len([c for c in todas_features if 'wgi_' in c and 'pca' not in c])
    
    categorias = ['PCA-derivadas\n(A2)', 'Quantitativas\n(lags, MA, delta)', 
                  'Interações\n(PCA×Quant)', 'WGI originais\n(removidas)']
    valores = [n_pca, n_quant, n_inter, n_wgi_orig]
    cores = ['#ff6f00', '#616161', '#388e3c', '#d32f2f']
    
    bars = ax.bar(categorias, valores, color=cores, edgecolor='black', alpha=0.85)
    
    for bar, val in zip(bars, valores):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    total = sum(valores)
    n_obs = len(df)
    ratio = n_obs / max(total, 1)
    
    ax.set_ylabel('Número de Features')
    ax.set_title(f'Parcimónia: {total} features para {n_obs} observações\n'
                 f'(Ratio obs/features = {ratio:.1f} — recomendado >3)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(valores) * 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(viz_dir, f'{nome_dataset}_parcimonia_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Parcimónia: {os.path.basename(path)}")


# ============================================================
# FUNÇÃO PRINCIPAL DE ENGENHARIA DE FEATURES (PARCIMONIOSA)
# ============================================================

def engenharia_features(df, nome_dataset, viz_dir=None, pca_dir=None):
    """
    Aplica engenharia de features PARCIMONIOSA a um dataset.
    
    Estratégia A2 EXCLUSIVA (fator latente PCA):
      - PC1 substitui os 6 WGI originais
      - Lags máximo 2 (para séries de 25 anos)
      - Interações PCA × quantitativos (H2)
      - WGI originais REMOVIDOS do dataset final
    
    Princípio: manter ratio observações/features > 3
    """
    print(f"\n  === Engenharia de Features (Parcimoniosa): {nome_dataset} ===")
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
    # ESTRATÉGIA A2: PCA nos indicadores WGI (TRAIN-ONLY)
    # ============================================================
    print(f"\n  [PCA] Estratégia A2 - Fator Latente (train-only)...")
    if len(colunas_wgi) >= 2:
        if pca_dir is None:
            pca_dir = os.path.join(config.DADOS_ENGENHARIA_DIR, 'pca_models')
        df, pca_info = aplicar_pca_wgi_train_only(df, colunas_wgi, nome_dataset, pca_dir)
        if pca_info:
            features_criadas['pca'] = ['wgi_pca1']
    else:
        print(f"    (Sem variáveis WGI - PCA não aplicável a este dataset)")
    
    # ============================================================
    # LAGS DO PCA (máximo 2 - parcimónia para 25 anos)
    # ============================================================
    LAGS_PCA = [1, 2]  # Reduzido de [1,2,3] para [1,2]
    
    if 'wgi_pca1' in df.columns:
        df, lags_pca = criar_lags(df, ['wgi_pca1'], LAGS_PCA)
        features_criadas['lags_pca'] = lags_pca
        print(f"  [1] Lags PCA (1,2): {len(lags_pca)} features")
    
    # ============================================================
    # LAGS DAS VARIÁVEIS QUANTITATIVAS (lag=1 apenas)
    # ============================================================
    df, lags_quant = criar_lags(df, colunas_quant, [1])
    features_criadas['lags_quant'] = lags_quant
    print(f"  [2] Lags quantitativos (1): {len(lags_quant)} features")
    
    # ============================================================
    # LAGS DA VARIÁVEL ALVO (1, 2)
    # ============================================================
    df, lags_target = criar_lags(df, [config.TARGET_VAR], [1, 2])
    features_criadas['lags_target'] = lags_target
    print(f"  [3] Lags target (1,2): {len(lags_target)} features")
    
    # ============================================================
    # MÉDIA MÓVEL DO PCA (3 anos)
    # ============================================================
    if 'wgi_pca1' in df.columns:
        df, ma_pca = criar_medias_moveis(df, ['wgi_pca1'], 3)
        features_criadas['ma_pca'] = ma_pca
        print(f"  [4] Média móvel PCA (3 anos): {len(ma_pca)} features")
    
    # ============================================================
    # DELTA DO PCA (variação anual)
    # ============================================================
    if 'wgi_pca1' in df.columns:
        df, deltas_pca = criar_deltas(df, ['wgi_pca1'])
        features_criadas['deltas_pca'] = deltas_pca
        print(f"  [5] Delta PCA: {len(deltas_pca)} features")
    
    # ============================================================
    # LOG-RETORNOS (variáveis económicas principais)
    # ============================================================
    vars_log_ret = ['pib_per_capita_ppc', 'formacao_bruta_capital_fixo_percent_pib',
                    'comercio_percent_pib', 'ied_percent_pib']
    vars_log_ret = [v for v in vars_log_ret if v in df.columns]
    df, log_rets = criar_log_retornos(df, vars_log_ret)
    features_criadas['log_retornos'] = log_rets
    print(f"  [6] Log-retornos: {len(log_rets)} features")
    
    # ============================================================
    # INTERAÇÕES PCA × QUANTITATIVOS (H2)
    # ============================================================
    if 'wgi_pca1' in df.columns:
        df, interacoes_pca = criar_interacoes_pca(df)
        features_criadas['interacoes_pca'] = interacoes_pca
        print(f"  [7] Interações PCA×Quant (H2): {len(interacoes_pca)} features")
    
    # ============================================================
    # REMOVER WGI ORIGINAIS (substituídos pelo PCA)
    # ============================================================
    if pca_info and colunas_wgi:
        cols_a_remover = [c for c in colunas_wgi if c in df.columns]
        df = df.drop(columns=cols_a_remover)
        print(f"\n  [PARCIMÓNIA] Removidos {len(cols_a_remover)} WGI originais "
              f"(substituídos por PC1)")
    
    # ============================================================
    # LIMPEZA FINAL
    # ============================================================
    
    # Remover linhas com NaN dos primeiros lags
    max_lag = 2  # Máximo lag usado
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
    n_obs = len(df_clean)
    ratio = n_obs / max(total_features, 1)
    
    print(f"\n  Total features criadas: {total_features}")
    print(f"  Shape final: {df_clean.shape}")
    print(f"  Ratio observações/features: {ratio:.1f} (recomendado >3)")
    
    if ratio < 3:
        print(f"  ⚠ AVISO: ratio < 3 — considerar redução adicional de features")
    
    # ============================================================
    # VISUALIZAÇÕES
    # ============================================================
    if viz_dir and pca_info:
        print(f"\n  Gerando visualizações...")
        gerar_visualizacoes(df_clean, pca_info, nome_dataset, viz_dir)
    
    return df_clean, features_criadas, pca_info


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def executar_passo3():
    """Executa o Passo 3 completo - Engenharia Parcimoniosa em 4 datasets."""
    print("\n" + "=" * 70)
    print("PASSO 3: ENGENHARIA DE FEATURES - ESTRATÉGIA PARCIMONIOSA (A2)")
    print("=" * 70)
    print("  Princípios: PCA train-only | Lags max=2 | WGI removidos | Ratio >3")
    
    os.makedirs(config.DADOS_ENGENHARIA_DIR, exist_ok=True)
    
    # Directório de visualizações do Passo 3
    viz_dir = os.path.join(config.BASE_DIR, 'eda_engenharia')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Directório para modelos PCA (reutilizáveis no passo 4)
    pca_dir = os.path.join(config.DADOS_ENGENHARIA_DIR, 'pca_models')
    os.makedirs(pca_dir, exist_ok=True)
    
    pca_infos = {}
    
    # ============================================================
    # DATASET 1: WDI LIMPO (SEM WGI - apenas features quantitativas)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 1: WDI LIMPO (SEM WGI)")
    print("-" * 70)
    
    wdi_limpo_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wdi_limpo.csv')
    df_wdi_limpo = pd.read_csv(wdi_limpo_path)
    
    df_wdi_eng, features_wdi, _ = engenharia_features(
        df_wdi_limpo.copy(), 'wdi_limpo', pca_dir=pca_dir
    )
    
    wdi_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv')
    df_wdi_eng.to_csv(wdi_eng_path, index=False)
    print(f"  Salvo: {wdi_eng_path}")
    
    # ============================================================
    # DATASET 2: AGREGADO (WDI + WGI) - DATASET PRINCIPAL
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 2: AGREGADO (WDI + WGI) — DATASET PRINCIPAL")
    print("-" * 70)
    
    agregado_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    df_agregado = pd.read_csv(agregado_path)
    
    df_agregado_eng, features_agregado, pca_info_agreg = engenharia_features(
        df_agregado.copy(), 'agregado', viz_dir=viz_dir, pca_dir=pca_dir
    )
    pca_infos['agregado'] = pca_info_agreg
    
    agregado_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv')
    df_agregado_eng.to_csv(agregado_eng_path, index=False)
    print(f"  Salvo: {agregado_eng_path}")
    
    # ============================================================
    # DATASET 3: SINTÉTICO (500 ANOS) - APENAS ROBUSTEZ
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 3: SINTÉTICO (500 ANOS) — APENAS TESTES DE ROBUSTEZ")
    print("-" * 70)
    print("  ⚠ NOTA: Correlações com target são fracas no sintético.")
    print("    Não usar para avaliação de desempenho preditivo.")
    
    sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'sintetico_500anos.csv')
    df_sintetico = pd.read_csv(sintetico_path)
    
    df_sintetico_eng, features_sintetico, pca_info_sint = engenharia_features(
        df_sintetico.copy(), 'sintetico', viz_dir=viz_dir, pca_dir=pca_dir
    )
    pca_infos['sintetico'] = pca_info_sint
    
    sintetico_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv')
    df_sintetico_eng.to_csv(sintetico_eng_path, index=False)
    print(f"  Salvo: {sintetico_eng_path}")
    
    # ============================================================
    # DATASET 4: WDI SINTÉTICO (500 ANOS, SEM WGI)
    # ============================================================
    
    print("\n" + "-" * 70)
    print("DATASET 4: WDI SINTÉTICO (500 ANOS, SEM WGI)")
    print("-" * 70)
    
    wdi_sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'wdi_sintetico_500anos.csv')
    df_wdi_sintetico = pd.read_csv(wdi_sintetico_path)
    
    df_wdi_sint_eng, features_wdi_sint, _ = engenharia_features(
        df_wdi_sintetico.copy(), 'wdi_sintetico', pca_dir=pca_dir
    )
    
    wdi_sint_eng_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv')
    df_wdi_sint_eng.to_csv(wdi_sint_eng_path, index=False)
    print(f"  Salvo: {wdi_sint_eng_path}")
    
    # ============================================================
    # SALVAR INFORMAÇÕES DO PCA (loadings)
    # ============================================================
    
    for nome, info in pca_infos.items():
        if info:
            loadings_path = os.path.join(config.DADOS_ENGENHARIA_DIR, f'{nome}_pca_loadings.csv')
            info['loadings'].to_csv(loadings_path)
            print(f"\n  PCA loadings ({nome}): {loadings_path}")
    
    # ============================================================
    # METADADOS
    # ============================================================
    
    ficheiros_saida = [wdi_eng_path, agregado_eng_path, sintetico_eng_path, wdi_sint_eng_path]
    
    pca_params = {}
    for nome, info in pca_infos.items():
        if info:
            pca_params[f'pca_{nome}_var_pc1'] = f"{info['variancia_explicada'][0]*100:.1f}%"
            pca_params[f'pca_{nome}_var_acumulada'] = [f"{v*100:.1f}%" for v in info['variancia_acumulada']]
            pca_params[f'pca_{nome}_n_treino'] = info['n_treino']
    
    gerar_metadados(
        passo='passo3_engenharia_features',
        descricao=('Engenharia de features PARCIMONIOSA. Estratégia A2 exclusiva: '
                   'PCA (fator latente) substitui os 6 WGI originais. '
                   'PCA ajustado apenas nos dados de treino (80% temporal por país) → sem data leakage. '
                   'Lags reduzidos a máximo 2. Interações PCA×Quantitativos para H2. '
                   'WGI originais removidos do dataset final.'),
        config=config,
        dados_entrada=[wdi_limpo_path, agregado_path, sintetico_path, wdi_sintetico_path],
        dados_saida=ficheiros_saida,
        parametros={
            'estrategia': 'A2 - Fator Latente (PCA) EXCLUSIVA',
            'pca_train_only': True,
            'pca_split': '80% temporal por país',
            'lags_pca': [1, 2],
            'lags_quantitativos': [1],
            'lags_target': [1, 2],
            'janela_media_movel': 3,
            'interacoes_pca': ['PC1×IED', 'PC1×Formação Capital', 'PC1×Comércio', 'PC1×PIB'],
            'wgi_originais_removidos': True,
            'sintetico_uso': 'apenas testes de robustez (não avaliação preditiva)',
            **pca_params,
        },
        metricas={
            'wdi_features': registar_dataframe_info(df_wdi_eng, 'WDI com Features'),
            'agregado_features': registar_dataframe_info(df_agregado_eng, 'Agregado com Features (A2)'),
            'sintetico_features': registar_dataframe_info(df_sintetico_eng, 'Sintético com Features'),
            'wdi_sintetico_features': registar_dataframe_info(df_wdi_sint_eng, 'WDI Sintético com Features'),
        }
    )
    
    auto_save_drive(ficheiros_saida, config)
    
    print("\n  PASSO 3 CONCLUÍDO — Estratégia Parcimoniosa (A2 exclusiva)")
    return df_wdi_eng, df_agregado_eng, df_sintetico_eng, df_wdi_sint_eng


if __name__ == '__main__':
    executar_passo3()
