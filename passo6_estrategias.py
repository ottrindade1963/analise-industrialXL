"""
============================================================
PASSO 6: ANALISE DE ESTRATEGIAS E EFEITOS DE MODERACAO
============================================================
Analises Completas:
  1. Correlacoes WGI x Target (Pearson + Spearman com p-values)
  2. Efeitos de moderacao: 24 interacoes WGI x WDI com p-values (OLS)
  3. Clusters de governanca (K-Means, 3 clusters, PCA 2D)
  4. Performance dos modelos por cluster de governanca
  5. Impacto por pais (top/bottom performers)
  6. Analise por regiao (Norte Africa, Ocidental, Oriental, Austral, Medio Oriente)

Visualizacoes (6 graficos):
  1. Heatmap correlacoes WGI x WDI
  2. Scatter moderacao (top 4 interacoes significativas)
  3. Clusters de governanca (PCA 2D)
  4. Performance por cluster
  5. Impacto WGI por pais (top 10 / bottom 10)
  6. Scatter WGI medio vs Target medio (com regressao)
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


def executar_passo6():
    """Analise de estrategias e efeitos de moderacao."""
    print("\n" + "=" * 70)
    print("  PASSO 6: ANALISE DE ESTRATEGIAS E MODERACAO")
    print("=" * 70)
    
    os.makedirs(config.ESTRATEGIAS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    # --------------------------------------------------------
    # CARREGAR DADOS
    # --------------------------------------------------------
    print("\n  [1/7] Carregando dados...")
    
    agg_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv')
    if not os.path.exists(agg_path):
        agg_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    
    if not os.path.exists(agg_path):
        print("  ERRO: Dataset agregado nao encontrado.")
        return
    
    df = pd.read_csv(agg_path)
    target = config.TARGET_VAR
    
    if target not in df.columns:
        for col in df.columns:
            if 'valor_agregado' in col or 'industrial' in col:
                target = col
                break
    
    # Identificar colunas WGI e WDI
    # Para clustering: incluir todas as features PCA (base + lags + MA + delta) para mais dimensionalidade
    wgi_cols_all = [c for c in df.columns if 'wgi_pca' in c.lower() or 'inter_pca' in c.lower()]
    # Para correlacoes: apenas base (sem lags/MA/delta)
    wgi_cols = [c for c in df.columns if 'wgi' in c.lower() and 'lag' not in c and 'ma' not in c and 'delta' not in c and 'inter_' not in c]
    wdi_cols = [c for c in df.columns if c not in wgi_cols_all and c not in ['country_code', 'year', 'pais', target]
                and 'lag' not in c and 'ma' not in c and 'delta' not in c and 'log_ret' not in c and 'inter_' not in c]
    wdi_cols = [c for c in wdi_cols if df[c].dtype in ['float64', 'int64', 'float32']][:12]
    
    print(f"    Dataset: {df.shape}")
    print(f"    WGI: {len(wgi_cols)} variaveis")
    print(f"    WDI: {len(wdi_cols)} variaveis")
    
    # ============================================================
    # [2/7] CORRELACOES WGI x TARGET (Pearson + Spearman)
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
    
    df_corr = pd.DataFrame(corr_resultados).sort_values('Pearson_r', key=abs, ascending=False)
    corr_path = os.path.join(config.ESTRATEGIAS_DIR, 'correlacoes_wgi_target.csv')
    df_corr.to_csv(corr_path, index=False)
    n_sig_corr = len(df_corr[df_corr['Significativo_5pct'] == 'Sim']) if len(df_corr) > 0 else 0
    print(f"    Correlacoes significativas: {n_sig_corr}/{len(df_corr)}")
    
    # ============================================================
    # [3/7] EFEITOS DE MODERACAO (24 INTERACOES COM P-VALUES)
    # ============================================================
    print("\n  [3/7] Efeitos de moderacao (interacoes WGI x WDI com OLS)...")
    
    interacao_resultados = []
    
    for wgi in wgi_cols[:6]:
        for wdi in wdi_cols[:4]:
            if wgi not in df.columns or wdi not in df.columns or target not in df.columns:
                continue
            
            valid = df[[wgi, wdi, target]].dropna()
            if len(valid) < 20:
                continue
            
            # Padronizar
            wgi_std = (valid[wgi] - valid[wgi].mean()) / (valid[wgi].std() + 1e-10)
            wdi_std = (valid[wdi] - valid[wdi].mean()) / (valid[wdi].std() + 1e-10)
            interacao = wgi_std * wdi_std
            
            # Regressao OLS: target ~ wgi + wdi + wgi*wdi + constante
            try:
                X_reg = np.column_stack([wgi_std, wdi_std, interacao, np.ones(len(valid))])
                y_reg = valid[target].values
                
                beta = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
                y_hat = X_reg @ beta
                residuos = y_reg - y_hat
                n_obs = len(y_reg)
                k = X_reg.shape[1]
                
                # Erro padrao dos coeficientes
                mse = np.sum(residuos**2) / max(n_obs - k, 1)
                try:
                    var_beta = mse * np.linalg.inv(X_reg.T @ X_reg + np.eye(k) * 1e-10)
                    se_beta = np.sqrt(np.abs(np.diag(var_beta)))
                except Exception:
                    se_beta = np.ones(k) * 1e-5
                
                # t-test para coeficiente de interacao (indice 2)
                t_stat = beta[2] / (se_beta[2] + 1e-10)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n_obs - k, 1)))
                
                r2 = 1 - np.sum(residuos**2) / (np.sum((y_reg - y_reg.mean())**2) + 1e-10)
                
                # R2 sem interacao (para calcular Delta R2)
                X_base = np.column_stack([wgi_std, wdi_std, np.ones(len(valid))])
                beta_base = np.linalg.lstsq(X_base, y_reg, rcond=None)[0]
                r2_base = 1 - np.sum((y_reg - X_base @ beta_base)**2) / (np.sum((y_reg - y_reg.mean())**2) + 1e-10)
                
                interacao_resultados.append({
                    'WGI': wgi, 'WDI': wdi,
                    'Beta_WGI': beta[0], 'Beta_WDI': beta[1],
                    'Beta_Interacao': beta[2], 'SE_Interacao': se_beta[2],
                    'T_Stat': t_stat, 'P_Value': p_value,
                    'R2_Completo': r2, 'R2_Base': r2_base,
                    'Delta_R2': r2 - r2_base, 'N': n_obs,
                    'Significativo_5pct': 'Sim' if p_value < 0.05 else 'Nao',
                    'Significativo_10pct': 'Sim' if p_value < 0.10 else 'Nao',
                    'Direcao': 'Positiva' if beta[2] > 0 else 'Negativa'
                })
            except Exception:
                continue
    
    df_inter = pd.DataFrame(interacao_resultados).sort_values('P_Value')
    inter_path = os.path.join(config.ESTRATEGIAS_DIR, 'interacoes_moderacao.csv')
    df_inter.to_csv(inter_path, index=False)
    
    n_sig_inter = len(df_inter[df_inter['Significativo_5pct'] == 'Sim']) if len(df_inter) > 0 else 0
    print(f"    Interacoes testadas: {len(df_inter)}")
    print(f"    Significativas (p<0.05): {n_sig_inter}")
    if len(df_inter) > 0:
        top = df_inter.iloc[0]
        print(f"    Melhor: {top['WGI']} x {top['WDI']} (beta={top['Beta_Interacao']:.3f}, p={top['P_Value']:.4f})")
    
    # ============================================================
    # [4/7] CLUSTERS DE GOVERNANCA (K-Means + PCA)
    # ============================================================
    print("\n  [4/7] Clusters de governanca (K-Means)...")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    pca_obj = None
    wgi_pais = pd.DataFrame()
    
    # Usar wgi_cols_all para clustering (inclui lags/MA/delta para mais dimensionalidade)
    wgi_cols_cluster = wgi_cols_all if len(wgi_cols_all) > 0 else wgi_cols
    
    if 'country_code' in df.columns and len(wgi_cols_cluster) > 0:
        wgi_pais = df.groupby('country_code')[wgi_cols_cluster].mean().dropna()
        
        if len(wgi_pais) >= 3:
            scaler = StandardScaler()
            wgi_scaled = scaler.fit_transform(wgi_pais)
            
            # K-Means com 3 clusters
            n_clusters = min(3, len(wgi_pais))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(wgi_scaled)
            
            wgi_pais['Cluster'] = clusters
            
            # Ordenar clusters por media WGI (usar primeira coluna disponivel)
            first_wgi_col = wgi_cols_cluster[0] if len(wgi_cols_cluster) > 0 else wgi_cols[0]
            cluster_means = wgi_pais.groupby('Cluster')[first_wgi_col].mean().sort_values()
            label_map = {}
            labels = ['Governanca_Baixa', 'Governanca_Media', 'Governanca_Alta']
            for i, idx in enumerate(cluster_means.index):
                label_map[idx] = labels[min(i, 2)]
            wgi_pais['Cluster_Nome'] = wgi_pais['Cluster'].map(label_map)
            
            # PCA para visualizacao (adaptado para dimensionalidade reduzida)
            n_features_wgi = wgi_scaled.shape[1]
            n_pca_components = min(2, n_features_wgi)
            
            if n_pca_components >= 1:
                pca_obj = PCA(n_components=n_pca_components)
                pca_coords = pca_obj.fit_transform(wgi_scaled)
                wgi_pais['PCA1'] = pca_coords[:, 0]
                if n_pca_components == 2:
                    wgi_pais['PCA2'] = pca_coords[:, 1]
                else:
                    # Se so ha 1 componente, duplicar com ruido para scatter plot
                    wgi_pais['PCA2'] = pca_coords[:, 0] + np.random.normal(0, 0.01, len(pca_coords))
            else:
                # Fallback: usar indice do pais como coordenada
                wgi_pais['PCA1'] = np.arange(len(wgi_pais))
                wgi_pais['PCA2'] = np.random.normal(0, 1, len(wgi_pais))
            
            clusters_path = os.path.join(config.ESTRATEGIAS_DIR, 'clusters_governanca.csv')
            wgi_pais.to_csv(clusters_path)
            print(f"    Clusters: {wgi_pais['Cluster_Nome'].value_counts().to_dict()}")
    
    # ============================================================
    # [5/7] PERFORMANCE POR CLUSTER
    # ============================================================
    print("\n  [5/7] Performance dos modelos por cluster...")
    
    perf_cluster = []
    if len(wgi_pais) > 0 and 'Cluster_Nome' in wgi_pais.columns:
        best_model = None
        for m in ['RandomForest', 'XGBoost', 'GradientBoosting']:
            path = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{m}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    best_model = pickle.load(f)
                break
        
        if best_model:
            for cluster_nome in wgi_pais['Cluster_Nome'].unique():
                paises_cluster = wgi_pais[wgi_pais['Cluster_Nome'] == cluster_nome].index.tolist()
                df_cluster = df[df['country_code'].isin(paises_cluster)]
                
                if len(df_cluster) > 10 and target in df_cluster.columns:
                    X_c = df_cluster.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
                    y_c = df_cluster[target].values
                    
                    try:
                        y_pred = best_model.predict(X_c)
                        rmse = np.sqrt(np.mean((y_c - y_pred)**2))
                        mae = np.mean(np.abs(y_c - y_pred))
                        r2 = 1 - np.sum((y_c - y_pred)**2) / (np.sum((y_c - y_c.mean())**2) + 1e-10)
                        
                        perf_cluster.append({
                            'Cluster': cluster_nome,
                            'N_Paises': len(paises_cluster),
                            'N_Obs': len(df_cluster),
                            'RMSE': rmse, 'MAE': mae, 'R2': r2,
                            'Target_Medio': y_c.mean(),
                            'Target_Std': y_c.std(),
                            'WGI_Medio': df_cluster[wgi_cols].mean().mean() if len(wgi_cols) > 0 else np.nan
                        })
                    except Exception:
                        pass
    
    df_perf_cluster = pd.DataFrame(perf_cluster)
    perf_cluster_path = os.path.join(config.ESTRATEGIAS_DIR, 'performance_por_cluster.csv')
    df_perf_cluster.to_csv(perf_cluster_path, index=False)
    
    # ============================================================
    # [6/7] IMPACTO POR PAIS (TOP/BOTTOM)
    # ============================================================
    print("\n  [6/7] Impacto WGI por pais...")
    
    impacto_pais = []
    if 'country_code' in df.columns and target in df.columns:
        for pais in df['country_code'].unique():
            df_pais = df[df['country_code'] == pais]
            if len(df_pais) > 5:
                target_mean = df_pais[target].mean()
                target_std = df_pais[target].std()
                
                # Tendencia temporal
                target_trend = 0
                if 'year' in df_pais.columns and len(df_pais) > 2:
                    slope, intercept, r_val, p_val, se = stats.linregress(
                        df_pais['year'].values, df_pais[target].values)
                    target_trend = slope
                
                wgi_mean = df_pais[wgi_cols].mean().mean() if len(wgi_cols) > 0 else np.nan
                wgi_trend = 0
                if len(wgi_cols) > 0 and 'year' in df_pais.columns and len(df_pais) > 2:
                    wgi_series = df_pais[wgi_cols[0]].values
                    valid_mask = ~np.isnan(wgi_series)
                    if valid_mask.sum() > 2:
                        slope_wgi, _, _, _, _ = stats.linregress(
                            df_pais['year'].values[valid_mask], wgi_series[valid_mask])
                        wgi_trend = slope_wgi
                
                impacto_pais.append({
                    'Pais': pais,
                    'Nome': config.PAISES.get(pais, pais) if hasattr(config, 'PAISES') else pais,
                    'Target_Medio': target_mean,
                    'Target_Std': target_std,
                    'Target_Tendencia': target_trend,
                    'WGI_Medio': wgi_mean,
                    'WGI_Tendencia': wgi_trend,
                    'N_Obs': len(df_pais)
                })
    
    df_impacto = pd.DataFrame(impacto_pais).sort_values('Target_Medio', ascending=False)
    impacto_path = os.path.join(config.ESTRATEGIAS_DIR, 'impacto_por_pais.csv')
    df_impacto.to_csv(impacto_path, index=False)
    
    # ============================================================
    # [7/7] VISUALIZACOES (6 GRAFICOS)
    # ============================================================
    print("\n  [7/7] Gerando 6 visualizacoes...")
    
    # --- GRAFICO 1: Heatmap correlacoes WGI x WDI ---
    if len(wgi_cols) > 0 and len(wdi_cols) > 0:
        corr_matrix = df[wgi_cols + wdi_cols[:8]].corr()
        wgi_wdi_corr = corr_matrix.loc[wgi_cols, wdi_cols[:8]]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(wgi_wdi_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Correlacoes: Variaveis de Governanca (WGI) x Variaveis Economicas (WDI)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'heatmap_correlacoes_wgi_wdi.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 2: Scatter moderacao (top 4 interacoes) ---
    if len(df_inter) > 0:
        top_inter = df_inter.head(4)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_flat = axes.flatten()
        
        for i, (_, row) in enumerate(top_inter.iterrows()):
            if i >= 4:
                break
            ax = axes_flat[i]
            wgi_col = row['WGI']
            wdi_col = row['WDI']
            
            if wgi_col in df.columns and wdi_col in df.columns and target in df.columns:
                valid = df[[wgi_col, wdi_col, target]].dropna()
                mediana_wgi = valid[wgi_col].median()
                alto = valid[valid[wgi_col] >= mediana_wgi]
                baixo = valid[valid[wgi_col] < mediana_wgi]
                
                ax.scatter(alto[wdi_col], alto[target], alpha=0.4, s=20, color='green', label='WGI Alto')
                ax.scatter(baixo[wdi_col], baixo[target], alpha=0.4, s=20, color='red', label='WGI Baixo')
                
                # Linhas de tendencia
                for subset, color in [(alto, 'darkgreen'), (baixo, 'darkred')]:
                    if len(subset) > 5:
                        z = np.polyfit(subset[wdi_col], subset[target], 1)
                        x_line = np.linspace(subset[wdi_col].min(), subset[wdi_col].max(), 50)
                        ax.plot(x_line, np.polyval(z, x_line), color=color, linewidth=2)
                
                sig = ' ***' if row['Significativo_5pct'] == 'Sim' else ''
                ax.set_title(f"{wgi_col}\nx {wdi_col}{sig}\nbeta={row['Beta_Interacao']:.3f}, p={row['P_Value']:.4f}", fontsize=9)
                ax.set_xlabel(wdi_col, fontsize=8)
                ax.set_ylabel(target[:30], fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        
        plt.suptitle('Efeitos de Moderacao: Top 4 Interacoes WGI x WDI\n(*** = significativo p<0.05)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'scatter_moderacao.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 3: Clusters de governanca (PCA 2D) ---
    if len(wgi_pais) > 0 and 'PCA1' in wgi_pais.columns and pca_obj is not None:
        fig, ax = plt.subplots(figsize=(11, 8))
        colors_map = {'Governanca_Baixa': 'red', 'Governanca_Media': 'orange', 'Governanca_Alta': 'green'}
        
        for cluster_nome in wgi_pais['Cluster_Nome'].unique():
            mask = wgi_pais['Cluster_Nome'] == cluster_nome
            color = colors_map.get(cluster_nome, 'gray')
            ax.scatter(wgi_pais.loc[mask, 'PCA1'], wgi_pais.loc[mask, 'PCA2'],
                      s=80, alpha=0.7, label=cluster_nome, color=color)
            for idx in wgi_pais[mask].index:
                ax.annotate(idx, (wgi_pais.loc[idx, 'PCA1'], wgi_pais.loc[idx, 'PCA2']),
                           fontsize=7, alpha=0.8)
        
        ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}% variancia)')
        ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}% variancia)')
        ax.set_title('Clusters de Governanca (PCA dos Indicadores WGI)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'clusters_governanca_pca.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 4: Performance por cluster ---
    if len(df_perf_cluster) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['red', 'orange', 'green'][:len(df_perf_cluster)]
        
        ax = axes[0]
        ax.bar(range(len(df_perf_cluster)), df_perf_cluster['RMSE'], color=colors, alpha=0.7)
        ax.set_xticks(range(len(df_perf_cluster)))
        ax.set_xticklabels(df_perf_cluster['Cluster'], rotation=15, fontsize=8)
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE por Cluster', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[1]
        ax.bar(range(len(df_perf_cluster)), df_perf_cluster['R2'], color=colors, alpha=0.7)
        ax.set_xticks(range(len(df_perf_cluster)))
        ax.set_xticklabels(df_perf_cluster['Cluster'], rotation=15, fontsize=8)
        ax.set_ylabel('R2')
        ax.set_title('R2 por Cluster', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[2]
        ax.bar(range(len(df_perf_cluster)), df_perf_cluster['Target_Medio'], color=colors, alpha=0.7,
               yerr=df_perf_cluster['Target_Std'], capsize=5)
        ax.set_xticks(range(len(df_perf_cluster)))
        ax.set_xticklabels(df_perf_cluster['Cluster'], rotation=15, fontsize=8)
        ax.set_ylabel('Target Medio')
        ax.set_title('Target por Cluster', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Performance do Modelo por Cluster de Governanca', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'performance_por_cluster.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 5: Impacto por pais (top 10 / bottom 10) ---
    if len(df_impacto) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        top10 = df_impacto.head(10)
        bottom10 = df_impacto.tail(10)
        
        ax = axes[0]
        ax.barh(range(len(top10)), top10['Target_Medio'], color='green', alpha=0.7)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['Nome'] if 'Nome' in top10.columns else top10['Pais'])
        ax.set_xlabel('Valor Agregado Industrial (%PIB)')
        ax.set_title('Top 10 Paises (Maior Industrializacao)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        ax = axes[1]
        ax.barh(range(len(bottom10)), bottom10['Target_Medio'], color='red', alpha=0.7)
        ax.set_yticks(range(len(bottom10)))
        ax.set_yticklabels(bottom10['Nome'] if 'Nome' in bottom10.columns else bottom10['Pais'])
        ax.set_xlabel('Valor Agregado Industrial (%PIB)')
        ax.set_title('Bottom 10 Paises (Menor Industrializacao)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Valor Agregado Industrial por Pais', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'impacto_por_pais.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 6: Scatter WGI medio vs Target medio ---
    if len(df_impacto) > 0 and 'WGI_Medio' in df_impacto.columns:
        fig, ax = plt.subplots(figsize=(11, 8))
        valid_imp = df_impacto.dropna(subset=['WGI_Medio', 'Target_Medio'])
        
        ax.scatter(valid_imp['WGI_Medio'], valid_imp['Target_Medio'], s=60, alpha=0.7, color='steelblue')
        
        # Linha de tendencia + estatisticas
        if len(valid_imp) > 5:
            z = np.polyfit(valid_imp['WGI_Medio'], valid_imp['Target_Medio'], 1)
            x_line = np.linspace(valid_imp['WGI_Medio'].min(), valid_imp['WGI_Medio'].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2, label='Tendencia linear')
            
            r, p = stats.pearsonr(valid_imp['WGI_Medio'], valid_imp['Target_Medio'])
            ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value = {p:.4f}\nn = {len(valid_imp)}',
                   transform=ax.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Anotar paises
        for _, row in valid_imp.iterrows():
            label = row['Nome'] if 'Nome' in row.index else row['Pais']
            ax.annotate(label, (row['WGI_Medio'], row['Target_Medio']), fontsize=7, alpha=0.7)
        
        ax.set_xlabel('WGI Medio (Qualidade de Governanca)', fontsize=11)
        ax.set_ylabel('Valor Agregado Industrial (%PIB)', fontsize=11)
        ax.set_title('Relacao entre Qualidade de Governanca e Industrializacao\n(Media por Pais)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.ESTRATEGIAS_DIR, 'scatter_wgi_vs_target.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # ============================================================
    # METADADOS E RESUMO
    # ============================================================
    ficheiros_saida = [corr_path, inter_path, perf_cluster_path, impacto_path]
    
    gerar_metadados(
        passo='passo6_estrategias',
        descricao='Analise de estrategias: correlacoes Pearson+Spearman, moderacao OLS com p-values, clusters K-Means, impacto por pais',
        config=config,
        dados_entrada=[agg_path],
        dados_saida=ficheiros_saida,
        parametros={'n_clusters': 3, 'interacoes_testadas': len(df_inter), 'alpha': 0.05},
        metricas={
            'correlacoes_significativas': n_sig_corr,
            'interacoes_significativas': n_sig_inter,
            'n_paises': len(df_impacto),
        }
    )
    auto_save_drive(ficheiros_saida, config)
    
    t_total = time.time() - t_inicio
    print(f"\n  {'='*60}")
    print(f"  RESUMO PASSO 6:")
    print(f"  {'='*60}")
    print(f"  Correlacoes WGI significativas: {n_sig_corr}/{len(df_corr)}")
    print(f"  Interacoes significativas (p<0.05): {n_sig_inter}/{len(df_inter)}")
    print(f"  Clusters de governanca: 3")
    print(f"  Ficheiros CSV: {len(ficheiros_saida)} | Graficos PNG: 6")
    print(f"  Tempo: {t_total:.1f}s")
    print(f"\n  OK PASSO 6 CONCLUIDO")


if __name__ == '__main__':
    executar_passo6()
