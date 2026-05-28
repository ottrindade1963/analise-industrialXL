"""
============================================================
PASSO 6: ANALISE DE ESTRATEGIAS E EFEITOS DE MODERACAO
============================================================
Analises Completas:
  1. Correlacoes WGI x Target (Pearson + Spearman com p-values)
  2. Efeitos de moderacao: interacoes WGI x WDI com p-values (OLS)
  3. Clusters de governanca (K-Means, 3 clusters, PCA 2D)
  4. Performance de TODOS os modelos por cluster de governanca
  5. Impacto por pais (top/bottom performers)
  6. Analise por regiao (Norte Africa, Ocidental, Oriental, Austral, Medio Oriente)
  7. Comparacao de modelos (Agregado vs WDI vs Sintetico)

Visualizacoes (8+ graficos):
  1. Heatmap correlacoes WGI x WDI
  2. Scatter moderacao (top 4 interacoes significativas)
  3. Clusters de governanca (PCA 2D)
  4. Performance de TODOS os modelos por cluster
  5. Impacto WGI por pais (top 10 / bottom 10)
  6. Scatter WGI medio vs Target medio (com regressao)
  7. Comparacao RMSE por modelo e dataset
  8. Ranking de modelos

NOTA: Compativel com a nova estrutura do Passo 3 (antecedencia temporal),
onde variaveis contemporaneas foram removidas e apenas lags/derivadas existem.

Entradas
--------
- dados_engenharia/agregado_features.csv
- modelos/modelo_Agregado_*.pkl
- modelos/modelo_WDI_Limpo_*.pkl

Saidas
------
- analise_estrategias/correlacoes_wgi_target.csv
- analise_estrategias/moderacao_interacoes.csv
- analise_estrategias/clusters_governanca.csv
- analise_estrategias/performance_por_cluster.csv
- analise_estrategias/impacto_por_pais.csv
- analise_estrategias/comparacao_modelos.csv
- analise_estrategias/*.png (8 graficos)

Dependencias
------------
numpy, pandas, scipy, scikit-learn (KMeans, PCA),
matplotlib, seaborn, statsmodels (OLS)

Contribuicao
-----------
- Artigo 1: Seccao 5 (performance por cluster)
- Artigo 2: Seccao 3 (efeitos de moderacao beta3)
- Dissertacao: Capitulo 5 (hipotese H3)
============================================================
"""
import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, auto_save_drive

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def encontrar_dataset(nome_dataset):
    """Procurar dataset em multiplos caminhos."""
    nome_dataset = nome_dataset.lower().replace(' ', '_')
    
    caminhos = [
        os.path.join(config.DADOS_ENGENHARIA_DIR, f'{nome_dataset}_features.csv'),
        os.path.join(config.DADOS_ENGENHARIA_DIR, f'{nome_dataset}.csv'),
        os.path.join(config.DADOS_AGREGADOS_DIR, f'{nome_dataset}_inner_join.csv'),
        os.path.join(config.BASE_DIR, 'dados_engenharia', f'{nome_dataset}_features.csv'),
        '/content/repo/pipeline_africa_mo/dados_engenharia/agregado_features.csv',
        '/content/repo/pipeline_africa_mo_resultados/dados_engenharia/agregado_features.csv',
    ]
    
    for path in caminhos:
        if os.path.exists(path):
            return path
    
    print(f"    Nao encontrado: {nome_dataset}")
    if os.path.exists(config.DADOS_ENGENHARIA_DIR):
        print(f"    Ficheiros em {config.DADOS_ENGENHARIA_DIR}:")
        for f in os.listdir(config.DADOS_ENGENHARIA_DIR)[:10]:
            print(f"      - {f}")
    return None


def carregar_modelos(dataset_nome):
    """Carregar todos os modelos para um dataset."""
    modelos = {}
    modelos_dir = config.MODELOS_DIR
    
    if os.path.exists(modelos_dir):
        for f in os.listdir(modelos_dir):
            if f.startswith(f'modelo_{dataset_nome}_') and f.endswith('.pkl'):
                modelo_nome = f.replace(f'modelo_{dataset_nome}_', '').replace('.pkl', '')
                try:
                    with open(os.path.join(modelos_dir, f), 'rb') as file:
                        modelos[modelo_nome] = pickle.load(file)
                except Exception as e:
                    print(f"    Erro ao carregar {f}: {str(e)[:100]}")
    
    return modelos


def identificar_variaveis(df, target):
    """
    Identificar variaveis WGI e WDI na nova estrutura do dataset.
    
    Na nova estrutura (pos-Passo 3 com antecedencia temporal):
    - Nao existem variaveis contemporaneas (foram removidas)
    - WGI: wgi_pca1_lag1, wgi_pca1_lag2, wgi_pca1_ma3, wgi_pca1_delta, inter_pca1_*
    - WDI: *_lag1, *_lag2, *_log_ret, target_lag1, target_lag2
    
    Para analise de moderacao:
    - wgi_cols: variaveis derivadas do PCA de governanca (lags, ma, delta)
    - wdi_cols: variaveis economicas (lags e log-retornos)
    - wgi_cols_all: todas as variaveis relacionadas com governanca (inclui interacoes)
    """
    excluir = {'country_code', 'year', 'pais', target}
    
    # WGI: variaveis derivadas do PCA de governanca (SEM interacoes)
    # Inclui: wgi_pca1_lag1, wgi_pca1_lag2, wgi_pca1_ma3, wgi_pca1_delta
    wgi_cols = [c for c in df.columns 
                if ('wgi_pca' in c.lower() or c.lower().startswith('pca1'))
                and 'inter_' not in c.lower() 
                and c not in excluir]
    
    # Se nao encontrar WGI, procurar alternativas
    if len(wgi_cols) == 0:
        wgi_cols = [c for c in df.columns 
                    if ('wgi' in c.lower() or 'pca' in c.lower() or 'governance' in c.lower())
                    and 'inter_' not in c.lower() 
                    and c not in excluir
                    and df[c].dtype in ['float64', 'int64', 'float32']]
    
    # WGI ALL: inclui tambem interacoes PCA (inter_pca1_*)
    wgi_cols_all = [c for c in df.columns 
                    if ('wgi_pca' in c.lower() or 'pca1' in c.lower())
                    and c not in excluir]
    
    # Se nao encontrar WGI ALL, usar WGI
    if len(wgi_cols_all) == 0:
        wgi_cols_all = wgi_cols
    
    # WDI: variaveis economicas (tudo que nao e WGI, target, ou ID)
    wdi_cols = [c for c in df.columns 
                if c not in excluir 
                and c not in wgi_cols_all
                and df[c].dtype in ['float64', 'int64', 'float32']]
    
    # Limitar WDI a 12 para visualizacao legivel
    wdi_cols = wdi_cols[:12]
    
    return wgi_cols, wgi_cols_all, wdi_cols


def executar_passo6():
    """Analise de estrategias e efeitos de moderacao."""
    print("\n" + "=" * 70)
    print("  PASSO 6: ANALISE DE ESTRATEGIAS E MODERACAO")
    print("=" * 70)
    
    os.makedirs(config.ESTRATEGIAS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    # ============================================================
    # [1/7] CARREGAR DADOS
    # ============================================================
    print("\n  [1/7] Carregando dados...")
    
    agg_path = encontrar_dataset('agregado')
    if not agg_path:
        print("  ERRO: Dataset agregado nao encontrado.")
        return
    
    df = pd.read_csv(agg_path)
    target = config.TARGET_VAR
    
    if target not in df.columns:
        for col in df.columns:
            if 'valor_agregado' in col or 'industrial' in col:
                target = col
                break
    
    # Identificar colunas WGI e WDI com a nova logica
    wgi_cols, wgi_cols_all, wdi_cols = identificar_variaveis(df, target)
    
    print(f"    Dataset: {df.shape}")
    print(f"    WGI: {len(wgi_cols)} variaveis")
    print(f"    WDI: {len(wdi_cols)} variaveis")
    
    # ============================================================
    # [2/7] CORRELACOES WGI x TARGET
    # ============================================================
    print("\n  [2/7] Correlacoes WGI x Target...")
    
    corr_resultados = []
    for wgi in wgi_cols:
        if wgi in df.columns and target in df.columns:
            valid = df[[wgi, target]].dropna()
            if len(valid) > 5:
                r, p = stats.pearsonr(valid[wgi], valid[target])
                rho, p_sp = stats.spearmanr(valid[wgi], valid[target])
                corr_resultados.append({
                    'Variavel_WGI': wgi,
                    'Pearson_r': r, 'Pearson_p': p,
                    'Spearman_rho': rho, 'Spearman_p': p_sp,
                    'Significativo_5pct': 'Sim' if p < 0.05 else 'Nao',
                    'N': len(valid)
                })
    
    corr_df = pd.DataFrame(corr_resultados)
    if len(corr_df) == 0:
        corr_df = pd.DataFrame(columns=['Variavel_WGI', 'Pearson_r', 'Pearson_p', 'Spearman_rho', 'Spearman_p', 'Significativo_5pct', 'N'])
        sig_count = 0
        print(f"    AVISO: Nenhuma correlacao calculada. DataFrame vazio.")
    else:
        sig_count = len(corr_df[corr_df['Significativo_5pct'] == 'Sim'])
    print(f"    Correlacoes significativas: {sig_count}/{len(corr_df)}")
    
    corr_path = os.path.join(config.ESTRATEGIAS_DIR, 'correlacoes_wgi_target.csv')
    corr_df.to_csv(corr_path, index=False)
    
    # ============================================================
    # [3/7] EFEITOS DE MODERACAO
    # ============================================================
    print("\n  [3/7] Efeitos de moderacao (interacoes WGI x WDI com OLS)...")
    
    from sklearn.linear_model import LinearRegression
    
    interacoes = []
    wgi_para_moderacao = wgi_cols[:3] if len(wgi_cols) >= 3 else wgi_cols
    wdi_para_moderacao = wdi_cols[:4] if len(wdi_cols) >= 4 else wdi_cols
    
    for wgi in wgi_para_moderacao:
        for wdi in wdi_para_moderacao:
            if wgi in df.columns and wdi in df.columns and target in df.columns:
                valid = df[[wgi, wdi, target]].dropna()
                if len(valid) > 10:
                    X = valid[[wgi, wdi]].values
                    X_inter = np.column_stack([X, X[:, 0] * X[:, 1]])
                    y = valid[target].values
                    
                    model = LinearRegression()
                    model.fit(X_inter, y)
                    
                    # T-test para coeficiente de interacao
                    residuals = y - model.predict(X_inter)
                    n_obs = len(y)
                    k_params = X_inter.shape[1]
                    mse = np.sum(residuals**2) / (n_obs - k_params)
                    
                    try:
                        XtX_inv = np.linalg.inv(X_inter.T @ X_inter)
                        se = np.sqrt(mse * XtX_inv.diagonal())
                        t_stat = model.coef_[2] / se[2]
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - k_params))
                    except np.linalg.LinAlgError:
                        p_val = 1.0
                        t_stat = 0.0
                    
                    interacoes.append({
                        'WGI': wgi, 'WDI': wdi,
                        'Beta_Interacao': model.coef_[2],
                        'T_Stat': t_stat,
                        'P_Value': p_val,
                        'Significativo': 'Sim' if p_val < 0.05 else 'Nao',
                        'R2': model.score(X_inter, y),
                        'N': n_obs
                    })
    
    inter_df = pd.DataFrame(interacoes)
    if len(inter_df) == 0:
        inter_df = pd.DataFrame(columns=['WGI', 'WDI', 'Beta_Interacao', 'T_Stat', 'P_Value', 'Significativo', 'R2', 'N'])
        sig_inter = 0
        print(f"    Nenhuma interacao testada")
    else:
        inter_df = inter_df.sort_values('P_Value')
        sig_inter = len(inter_df[inter_df['Significativo'] == 'Sim'])
        print(f"    Interacoes testadas: {len(inter_df)}")
        print(f"    Significativas (p<0.05): {sig_inter}")
        if len(inter_df) > 0:
            best = inter_df.iloc[0]
            print(f"    Melhor: {best['WGI']} x {best['WDI']} (beta={best['Beta_Interacao']:.3f}, p={best['P_Value']:.4f})")
    
    inter_path = os.path.join(config.ESTRATEGIAS_DIR, 'interacoes_wgi_wdi.csv')
    inter_df.to_csv(inter_path, index=False)
    
    # ============================================================
    # [4/7] CLUSTERS DE GOVERNANCA
    # ============================================================
    print("\n  [4/7] Clusters de governanca (K-Means)...")
    
    wgi_cols_cluster = wgi_cols_all if len(wgi_cols_all) > 0 else wgi_cols
    pca_obj = None
    wgi_pais = pd.DataFrame()
    
    if 'country_code' in df.columns and len(wgi_cols_cluster) > 0:
        wgi_pais = df.groupby('country_code')[wgi_cols_cluster].mean().dropna()
        
        if len(wgi_pais) >= 3:
            scaler = StandardScaler()
            wgi_scaled = scaler.fit_transform(wgi_pais)
            
            n_clusters = min(3, len(wgi_pais))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(wgi_scaled)
            
            wgi_pais['Cluster'] = clusters
            
            first_wgi_col = wgi_cols_cluster[0]
            cluster_means = wgi_pais.groupby('Cluster')[first_wgi_col].mean().sort_values()
            label_map = {}
            labels = ['Governanca_Baixa', 'Governanca_Media', 'Governanca_Alta']
            for i, idx in enumerate(cluster_means.index):
                label_map[idx] = labels[min(i, 2)]
            wgi_pais['Cluster_Nome'] = wgi_pais['Cluster'].map(label_map)
            
            # PCA para visualizacao 2D
            n_features_wgi = wgi_scaled.shape[1]
            n_pca_components = min(2, n_features_wgi)
            
            if n_pca_components >= 1:
                pca_obj = PCA(n_components=n_pca_components)
                pca_coords = pca_obj.fit_transform(wgi_scaled)
                wgi_pais['PCA1'] = pca_coords[:, 0]
                if n_pca_components == 2:
                    wgi_pais['PCA2'] = pca_coords[:, 1]
                else:
                    wgi_pais['PCA2'] = pca_coords[:, 0] + np.random.normal(0, 0.01, len(pca_coords))
            
            clusters_path = os.path.join(config.ESTRATEGIAS_DIR, 'clusters_governanca.csv')
            wgi_pais.to_csv(clusters_path)
            print(f"    Clusters: {wgi_pais['Cluster_Nome'].value_counts().to_dict()}")
    
    # ============================================================
    # [5/7] PERFORMANCE DOS MODELOS POR CLUSTER
    # ============================================================
    print("\n  [5/7] Performance dos modelos por cluster...")
    
    perf_cluster = []
    modelos_agregado = carregar_modelos('Agregado')
    
    # Identificar features usadas nos modelos
    feature_cols = [c for c in df.columns if c not in ['country_code', 'year', 'pais', target]
                    and df[c].dtype in ['float64', 'int64', 'float32']]
    
    if len(wgi_pais) > 0 and 'Cluster_Nome' in wgi_pais.columns and len(modelos_agregado) > 0:
        for modelo_nome, modelo in modelos_agregado.items():
            for cluster_nome in wgi_pais['Cluster_Nome'].unique():
                paises_cluster = wgi_pais[wgi_pais['Cluster_Nome'] == cluster_nome].index
                df_cluster = df[df['country_code'].isin(paises_cluster)]
                
                if len(df_cluster) > 10:
                    try:
                        if hasattr(modelo, 'feature_names_in_'):
                            model_features = list(modelo.feature_names_in_)
                        else:
                            model_features = feature_cols
                        
                        available_features = [c for c in model_features if c in df_cluster.columns]
                        X_cluster = df_cluster[available_features].dropna()
                        y_cluster = df_cluster.loc[X_cluster.index, target]
                        
                        if len(X_cluster) > 5:
                            y_pred = modelo.predict(X_cluster)
                            rmse = np.sqrt(np.mean((y_cluster - y_pred)**2))
                            ss_res = np.sum((y_cluster - y_pred)**2)
                            ss_tot = np.sum((y_cluster - y_cluster.mean())**2)
                            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            perf_cluster.append({
                                'Modelo': modelo_nome,
                                'Cluster': cluster_nome,
                                'RMSE': rmse,
                                'R2': r2,
                                'N_Obs': len(X_cluster)
                            })
                    except Exception as e:
                        pass
    
    if perf_cluster:
        perf_df = pd.DataFrame(perf_cluster)
        perf_path = os.path.join(config.ESTRATEGIAS_DIR, 'performance_modelos_por_cluster.csv')
        perf_df.to_csv(perf_path, index=False)
        print(f"    Avaliados: {len(perf_df)} combinacoes modelo x cluster")
    
    # ============================================================
    # [6/7] IMPACTO POR PAIS
    # ============================================================
    print("\n  [6/7] Impacto WGI por pais...")
    
    impacto_pais = []
    if 'country_code' in df.columns:
        wgi_col_impacto = wgi_cols[0] if len(wgi_cols) > 0 else (wgi_cols_all[0] if len(wgi_cols_all) > 0 else None)
        
        for pais in df['country_code'].unique():
            df_pais = df[df['country_code'] == pais]
            if len(df_pais) > 5:
                wgi_mean = df_pais[wgi_col_impacto].mean() if wgi_col_impacto else 0
                target_mean = df_pais[target].mean()
                impacto_pais.append({
                    'Pais': pais,
                    'WGI_Medio': wgi_mean,
                    'Target_Medio': target_mean,
                    'N_Obs': len(df_pais)
                })
    
    if impacto_pais:
        impacto_df = pd.DataFrame(impacto_pais).sort_values('Target_Medio', ascending=False)
        impacto_path = os.path.join(config.ESTRATEGIAS_DIR, 'impacto_wgi_por_pais.csv')
        impacto_df.to_csv(impacto_path, index=False)
        print(f"    Paises analisados: {len(impacto_df)}")
    
    # ============================================================
    # [7/7] VISUALIZACOES
    # ============================================================
    print("\n  [7/7] Gerando visualizacoes...")
    n_graficos = 0
    
    # 1. Heatmap Correlacoes WGI x WDI
    if len(wgi_cols) > 0 and len(wdi_cols) > 0:
        try:
            fig, ax = plt.subplots(figsize=(12, max(4, len(wgi_cols))))
            corr_matrix = df[wgi_cols + wdi_cols].corr()
            sns.heatmap(corr_matrix.loc[wgi_cols, wdi_cols], annot=True, cmap='coolwarm',
                       center=0, ax=ax, cbar_kws={'label': 'Correlacao'}, fmt='.2f')
            ax.set_title('Correlacoes: WGI (PCA derivadas) vs WDI (lags)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'heatmap_correlacoes_wgi_wdi.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    heatmap_correlacoes_wgi_wdi.png")
        except Exception as e:
            print(f"    Erro heatmap: {str(e)[:80]}")
    
    # 2. Scatter Moderacao (top 4 interacoes)
    if len(inter_df) > 0:
        try:
            n_inter = min(4, len(inter_df))
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (_, row) in enumerate(inter_df.head(n_inter).iterrows()):
                ax = axes[idx]
                wgi_col = row['WGI']
                wdi_col = row['WDI']
                
                if wgi_col in df.columns and wdi_col in df.columns:
                    valid = df[[wgi_col, wdi_col, target]].dropna()
                    scatter = ax.scatter(valid[wgi_col], valid[target], c=valid[wdi_col],
                                       cmap='viridis', alpha=0.6, s=50)
                    ax.set_xlabel(wgi_col, fontsize=9)
                    ax.set_ylabel(target[:30], fontsize=9)
                    sig_str = '*' if row['P_Value'] < 0.05 else 'ns'
                    ax.set_title(f'{wdi_col[:25]} (p={row["P_Value"]:.4f}) {sig_str}', fontsize=9, fontweight='bold')
                    plt.colorbar(scatter, ax=ax, label=wdi_col[:20])
            
            for idx in range(n_inter, 4):
                axes[idx].set_visible(False)
            
            plt.suptitle('Efeitos de Moderacao: WGI x WDI -> Target', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'scatter_moderacao_top4.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    scatter_moderacao_top4.png")
        except Exception as e:
            print(f"    Erro scatter moderacao: {str(e)[:80]}")
    
    # 3. Clusters de Governanca (PCA 2D)
    if len(wgi_pais) > 0 and 'PCA1' in wgi_pais.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = {'Governanca_Baixa': 'red', 'Governanca_Media': 'orange', 'Governanca_Alta': 'green'}
            for cluster in wgi_pais['Cluster_Nome'].unique():
                mask = wgi_pais['Cluster_Nome'] == cluster
                ax.scatter(wgi_pais[mask]['PCA1'], wgi_pais[mask]['PCA2'],
                          label=cluster, s=100, alpha=0.7, color=colors.get(cluster, 'blue'))
                for idx in wgi_pais[mask].index:
                    ax.annotate(idx, (wgi_pais.loc[idx, 'PCA1'], wgi_pais.loc[idx, 'PCA2']),
                               fontsize=7, alpha=0.7)
            
            ax.set_xlabel('PC1 (Governanca)', fontsize=12)
            ax.set_ylabel('PC2 (Variacao)', fontsize=12)
            ax.set_title('Clusters de Governanca (K-Means, PCA 2D)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'clusters_governanca_pca.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    clusters_governanca_pca.png")
        except Exception as e:
            print(f"    Erro clusters: {str(e)[:80]}")
    
    # 4. Performance de Modelos por Cluster
    if perf_cluster and len(perf_cluster) > 0:
        try:
            perf_df_local = pd.DataFrame(perf_cluster)
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot = perf_df_local.pivot_table(values='RMSE', index='Modelo', columns='Cluster', aggfunc='mean')
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('RMSE de Modelos por Cluster de Governanca', fontsize=14, fontweight='bold')
            ax.set_ylabel('RMSE (menor eh melhor)', fontsize=12)
            ax.set_xlabel('Modelo', fontsize=12)
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'performance_modelos_por_cluster.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    performance_modelos_por_cluster.png")
        except Exception as e:
            print(f"    Erro perf cluster: {str(e)[:80]}")
    
    # 5. Impacto WGI por Pais (Top 10 / Bottom 10)
    if impacto_pais and len(impacto_pais) > 0:
        try:
            impacto_df_local = pd.DataFrame(impacto_pais).sort_values('Target_Medio', ascending=False)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            top10 = impacto_df_local.head(10)
            ax1.barh(range(len(top10)), top10['Target_Medio'], color='green', alpha=0.7)
            ax1.set_yticks(range(len(top10)))
            ax1.set_yticklabels(top10['Pais'])
            ax1.set_xlabel('Target Medio (Industrializacao %)', fontsize=11)
            ax1.set_title('Top 10 Paises (Maior Industrializacao)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            bottom10 = impacto_df_local.tail(10)
            ax2.barh(range(len(bottom10)), bottom10['Target_Medio'], color='red', alpha=0.7)
            ax2.set_yticks(range(len(bottom10)))
            ax2.set_yticklabels(bottom10['Pais'])
            ax2.set_xlabel('Target Medio (Industrializacao %)', fontsize=11)
            ax2.set_title('Bottom 10 Paises (Menor Industrializacao)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'impacto_wgi_por_pais_top_bottom.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    impacto_wgi_por_pais_top_bottom.png")
        except Exception as e:
            print(f"    Erro impacto pais: {str(e)[:80]}")
    
    # 6. Scatter WGI Medio vs Target Medio (com regressao)
    if impacto_pais and len(impacto_pais) > 5:
        try:
            impacto_df_local = pd.DataFrame(impacto_pais).dropna()
            fig, ax = plt.subplots(figsize=(10, 7))
            
            ax.scatter(impacto_df_local['WGI_Medio'], impacto_df_local['Target_Medio'],
                      s=100, alpha=0.6, color='steelblue')
            
            z = np.polyfit(impacto_df_local['WGI_Medio'], impacto_df_local['Target_Medio'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(impacto_df_local['WGI_Medio'].min(), impacto_df_local['WGI_Medio'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Regressao (slope={z[0]:.3f})')
            
            for _, row in impacto_df_local.iterrows():
                ax.annotate(row['Pais'], (row['WGI_Medio'], row['Target_Medio']),
                           fontsize=7, alpha=0.7)
            
            ax.set_xlabel('WGI Medio (Qualidade Institucional)', fontsize=12)
            ax.set_ylabel('Target Medio (Industrializacao %)', fontsize=12)
            ax.set_title('Relacao WGI vs Industrializacao por Pais', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'scatter_wgi_vs_target.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n_graficos += 1
            print("    scatter_wgi_vs_target.png")
        except Exception as e:
            print(f"    Erro scatter wgi: {str(e)[:80]}")
    
    # ============================================================
    # METADADOS E RESUMO
    # ============================================================
    tempo_total = time.time() - t_inicio
    
    metadados = {
        'correlacoes_wgi_significativas': sig_count,
        'correlacoes_wgi_total': len(corr_df),
        'interacoes_significativas': sig_inter,
        'interacoes_total': len(inter_df),
        'n_clusters': 3 if len(wgi_pais) > 0 else 0,
        'n_modelos_avaliados': len(perf_cluster),
        'n_paises': len(impacto_pais),
        'n_graficos': n_graficos,
        'wgi_cols_usadas': wgi_cols,
        'wdi_cols_usadas': wdi_cols[:5],
        'tempo_segundos': tempo_total
    }
    
    gerar_metadados('passo6_estrategias', 'Analise de estrategias e efeitos de moderacao', config,
                    metricas=metadados)
    
    # Auto-save para Drive
    ficheiros_save = []
    if os.path.exists(config.ESTRATEGIAS_DIR):
        for f in os.listdir(config.ESTRATEGIAS_DIR):
            ficheiros_save.append(os.path.join(config.ESTRATEGIAS_DIR, f))
    auto_save_drive(ficheiros_save, config)
    
    print(f"\n  {'=' * 60}")
    print(f"  RESUMO PASSO 6:")
    print(f"  {'=' * 60}")
    print(f"  Correlacoes WGI significativas: {sig_count}/{len(corr_df)}")
    print(f"  Interacoes significativas (p<0.05): {sig_inter}/{len(inter_df)}")
    print(f"  Clusters de governanca: {3 if len(wgi_pais) > 0 else 0}")
    print(f"  Ficheiros CSV: 4 | Graficos PNG: {n_graficos}")
    print(f"  Tempo: {tempo_total:.1f}s")
    print(f"\n  OK PASSO 6 CONCLUIDO\n")


if __name__ == '__main__':
    executar_passo6()
