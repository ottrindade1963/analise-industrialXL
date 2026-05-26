"""
============================================================
PASSO 8: ANALISE GEOGRAFICA AVANCADA
============================================================
Analises Completas:
  1. Serie temporal por pais (real vs previsto, top 6 paises)
  2. Convergencia beta (regressao crescimento vs nivel inicial)
  3. Convergencia beta com IC 95% (bootstrap)
  4. Quebras estruturais (Chow test por pais)
  5. Clusters geograficos (K-Means, 3 clusters)
  6. Analise por regiao (5 regioes)
  7. Mapa de performance (heatmap pais x ano)
============================================================
"""
import os, sys, pickle, time
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


def encontrar_dataset(nome_dataset):
    """Procurar dataset em multiplos caminhos (compatível com Colab)."""
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
    
    return None


def executar_passo8():
    print("\n" + "=" * 70)
    print("  PASSO 8: ANALISE GEOGRAFICA AVANCADA")
    print("=" * 70)
    os.makedirs(config.GEOGRAFICA_DIR, exist_ok=True)
    t_inicio = time.time()

    # CARREGAR DADOS
    print("\n  [1/7] Carregando dados e modelos...")
    agg_path = encontrar_dataset('agregado')
    if not agg_path:
        print("  ERRO: Dataset agregado nao encontrado.")
        print(f"  Procurou em: {config.DADOS_ENGENHARIA_DIR}")
        return
    
    df = pd.read_csv(agg_path)
    target = config.TARGET_VAR
    if target not in df.columns:
        for col in df.columns:
            if 'valor_agregado' in col or 'industrial' in col:
                target = col
                break

    # Carregar melhor modelo
    best_model = None
    for m in ['RandomForest', 'XGBoost', 'GradientBoosting']:
        mp = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{m}.pkl')
        if os.path.exists(mp):
            with open(mp, 'rb') as f:
                best_model = pickle.load(f)
            break

    # Definir regioes
    regioes = {
        'Norte de Africa': ['DZA', 'EGY', 'LBY', 'MAR', 'TUN'],
        'Africa Ocidental': ['GHA', 'NGA', 'SEN', 'CIV', 'CMR'],
        'Africa Oriental': ['ETH', 'KEN', 'TZA', 'UGA', 'RWA', 'MOZ'],
        'Africa Austral': ['ZAF', 'BWA', 'NAM', 'ZMB', 'ZWE'],
        'Medio Oriente': ['SAU', 'ARE', 'QAT', 'KWT', 'BHR', 'OMN', 'JOR', 'LBN', 'IRQ']
    }
    pais_regiao = {}
    for reg, paises in regioes.items():
        for p in paises:
            pais_regiao[p] = reg

    # PREVISOES POR PAIS
    print("\n  [2/7] Previsoes por pais...")
    previsoes_pais = {}
    erros_pais = {}
    if best_model and 'country_code' in df.columns and target in df.columns:
        for pais in df['country_code'].unique():
            df_p = df[df['country_code'] == pais].sort_values('year')
            if len(df_p) < 5:
                continue
            X_p = df_p.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
            y_p = df_p[target].values
            try:
                y_pred = best_model.predict(X_p)
                previsoes_pais[pais] = {'year': df_p['year'].values, 'real': y_p, 'previsto': y_pred}
                rmse = np.sqrt(np.mean((y_p - y_pred)**2))
                mae = np.mean(np.abs(y_p - y_pred))
                r2 = 1 - np.sum((y_p - y_pred)**2) / (np.sum((y_p - y_p.mean())**2) + 1e-10)
                erros_pais[pais] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'N': len(y_p), 'Regiao': pais_regiao.get(pais, 'Outro')}
            except Exception:
                pass
    df_erros = pd.DataFrame(erros_pais).T
    df_erros.index.name = 'country_code'
    erros_path = os.path.join(config.GEOGRAFICA_DIR, 'erros_por_pais.csv')
    df_erros.to_csv(erros_path)

    # CONVERGENCIA BETA
    print("\n  [3/7] Convergencia beta...")
    conv_data = []
    if 'country_code' in df.columns and 'year' in df.columns and target in df.columns:
        for pais in df['country_code'].unique():
            df_p = df[df['country_code'] == pais].sort_values('year')
            if len(df_p) < 5:
                continue
            y_vals = df_p[target].values
            y_inicial = y_vals[0]
            y_final = y_vals[-1]
            n_anos = len(y_vals)
            crescimento = (y_final - y_inicial) / max(n_anos, 1)
            conv_data.append({'Pais': pais, 'Nivel_Inicial': y_inicial, 'Nivel_Final': y_final, 'Crescimento_Anual': crescimento, 'N_Anos': n_anos, 'Regiao': pais_regiao.get(pais, 'Outro')})
    df_conv = pd.DataFrame(conv_data)
    conv_path = os.path.join(config.GEOGRAFICA_DIR, 'convergencia_beta.csv')
    df_conv.to_csv(conv_path, index=False)

    # Regressao beta-convergencia com IC 95% bootstrap
    beta_coef = None
    beta_p = None
    beta_ic = [None, None]
    if len(df_conv) > 5:
        x_conv = df_conv['Nivel_Inicial'].values
        y_conv = df_conv['Crescimento_Anual'].values
        slope, intercept, r_val, p_val, se = stats.linregress(x_conv, y_conv)
        beta_coef = slope
        beta_p = p_val
        # Bootstrap IC 95%
        n_boot = 1000
        slopes_boot = []
        for _ in range(n_boot):
            idx = np.random.choice(len(x_conv), len(x_conv), replace=True)
            s, _, _, _, _ = stats.linregress(x_conv[idx], y_conv[idx])
            slopes_boot.append(s)
        beta_ic = [np.percentile(slopes_boot, 2.5), np.percentile(slopes_boot, 97.5)]
        print(f"    Beta = {beta_coef:.4f} (p={beta_p:.4f}), IC95%=[{beta_ic[0]:.4f}, {beta_ic[1]:.4f}]")
        convergencia = 'Sim' if beta_coef < 0 and beta_p < 0.05 else 'Nao'
        print(f"    Convergencia: {convergencia}")

    # QUEBRAS ESTRUTURAIS (CHOW TEST)
    print("\n  [4/7] Quebras estruturais (Chow test)...")
    quebras = []
    if 'country_code' in df.columns and 'year' in df.columns and target in df.columns:
        for pais in df['country_code'].unique():
            df_p = df[df['country_code'] == pais].sort_values('year')
            if len(df_p) < 10:
                continue
            years = df_p['year'].values
            y_vals = df_p[target].values
            best_f = 0
            best_year = None
            best_p = 1.0
            for t in range(3, len(years) - 3):
                y1 = y_vals[:t]
                y2 = y_vals[t:]
                x1 = np.arange(t).reshape(-1, 1)
                x2 = np.arange(len(y2)).reshape(-1, 1)
                x_all = np.arange(len(y_vals)).reshape(-1, 1)
                slope_all, int_all, _, _, _ = stats.linregress(x_all.flatten(), y_vals)
                rss_all = np.sum((y_vals - (slope_all * x_all.flatten() + int_all))**2)
                s1, i1, _, _, _ = stats.linregress(x1.flatten(), y1)
                rss1 = np.sum((y1 - (s1 * x1.flatten() + i1))**2)
                s2, i2, _, _, _ = stats.linregress(x2.flatten(), y2)
                rss2 = np.sum((y2 - (s2 * x2.flatten() + i2))**2)
                k = 2
                n = len(y_vals)
                f_stat = ((rss_all - rss1 - rss2) / k) / ((rss1 + rss2) / max(n - 2*k, 1))
                if f_stat > best_f:
                    best_f = f_stat
                    best_year = years[t]
                    best_p = 1 - stats.f.cdf(f_stat, k, max(n - 2*k, 1))
            if best_year is not None:
                quebras.append({'Pais': pais, 'Ano_Quebra': best_year, 'F_Stat': best_f, 'P_Value': best_p, 'Significativo': 'Sim' if best_p < 0.05 else 'Nao', 'Regiao': pais_regiao.get(pais, 'Outro')})
    df_quebras = pd.DataFrame(quebras).sort_values('F_Stat', ascending=False)
    quebras_path = os.path.join(config.GEOGRAFICA_DIR, 'quebras_estruturais.csv')
    df_quebras.to_csv(quebras_path, index=False)
    n_sig_quebras = len(df_quebras[df_quebras['Significativo'] == 'Sim']) if len(df_quebras) > 0 else 0
    print(f"    Quebras significativas: {n_sig_quebras}/{len(df_quebras)}")

    # CLUSTERS GEOGRAFICOS
    print("\n  [5/7] Clusters geograficos...")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    cluster_data = []
    if 'country_code' in df.columns and target in df.columns:
        for pais in df['country_code'].unique():
            df_p = df[df['country_code'] == pais]
            if len(df_p) < 3:
                continue
            feats = {'target_mean': df_p[target].mean(), 'target_std': df_p[target].std(), 'target_trend': 0, 'n_obs': len(df_p)}
            if 'year' in df_p.columns and len(df_p) > 2:
                s, _, _, _, _ = stats.linregress(df_p['year'].values, df_p[target].values)
                feats['target_trend'] = s
            wgi_cols = [c for c in df_p.columns if 'wgi' in c.lower() and 'lag' not in c and 'ma' not in c]
            if wgi_cols:
                feats['wgi_mean'] = df_p[wgi_cols].mean().mean()
            else:
                feats['wgi_mean'] = 0
            feats['Pais'] = pais
            feats['Regiao'] = pais_regiao.get(pais, 'Outro')
            cluster_data.append(feats)
    df_cluster = pd.DataFrame(cluster_data)
    pca_obj = None
    if len(df_cluster) >= 3:
        feat_cols = ['target_mean', 'target_std', 'target_trend', 'wgi_mean']
        X_cl = df_cluster[feat_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cl)
        n_cl = min(3, len(df_cluster))
        km = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
        df_cluster['Cluster'] = km.fit_predict(X_scaled)
        pca_obj = PCA(n_components=2)
        pca_coords = pca_obj.fit_transform(X_scaled)
        df_cluster['PC1'] = pca_coords[:, 0]
        df_cluster['PC2'] = pca_coords[:, 1]
    clusters_path = os.path.join(config.GEOGRAFICA_DIR, 'clusters_geograficos.csv')
    df_cluster.to_csv(clusters_path, index=False)

    # ANALISE POR REGIAO
    print("\n  [6/7] Analise por regiao...")
    regiao_stats = []
    for reg, codigos in regioes.items():
        df_reg = df[df['country_code'].isin(codigos)]
        if len(df_reg) > 0 and target in df_reg.columns:
            regiao_stats.append({'Regiao': reg, 'Media': df_reg[target].mean(), 'Mediana': df_reg[target].median(), 'Std': df_reg[target].std(), 'Min': df_reg[target].min(), 'Max': df_reg[target].max(), 'N_Paises': df_reg['country_code'].nunique(), 'N_Obs': len(df_reg)})
    df_regioes = pd.DataFrame(regiao_stats)
    regioes_path = os.path.join(config.GEOGRAFICA_DIR, 'analise_por_regiao.csv')
    df_regioes.to_csv(regioes_path, index=False)

    # VISUALIZACOES (7 GRAFICOS)
    print("\n  [7/7] Gerando 7 visualizacoes...")

    # GRAFICO 1: Serie temporal top 6 paises
    if previsoes_pais:
        top6 = sorted(erros_pais.keys(), key=lambda p: erros_pais[p]['R2'], reverse=True)[:6]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for i, pais in enumerate(top6):
            ax = axes.flat[i]
            dados = previsoes_pais[pais]
            ax.plot(dados['year'], dados['real'], 'b-o', markersize=3, label='Real', linewidth=1.5)
            ax.plot(dados['year'], dados['previsto'], 'r--s', markersize=3, label='Previsto', linewidth=1.5)
            ax.fill_between(dados['year'], dados['real'], dados['previsto'], alpha=0.2, color='gray')
            ax.set_title(f'{pais} (R2={erros_pais[pais]["R2"]:.3f})', fontsize=9, fontweight='bold')
            ax.set_xlabel('Ano', fontsize=8)
            ax.set_ylabel(target[:20], fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        plt.suptitle('Serie Temporal: Top 6 Paises (Melhor Performance)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'serie_temporal_top6.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 2: Convergencia beta
    if len(df_conv) > 5 and beta_coef is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_conv['Nivel_Inicial'], df_conv['Crescimento_Anual'], s=50, alpha=0.7, color='steelblue')
        x_line = np.linspace(df_conv['Nivel_Inicial'].min(), df_conv['Nivel_Inicial'].max(), 50)
        ax.plot(x_line, beta_coef * x_line + (df_conv['Crescimento_Anual'].mean() - beta_coef * df_conv['Nivel_Inicial'].mean()), 'r-', linewidth=2, label=f'beta={beta_coef:.4f} (p={beta_p:.4f})')
        for _, row in df_conv.iterrows():
            ax.annotate(row['Pais'], (row['Nivel_Inicial'], row['Crescimento_Anual']), fontsize=7, alpha=0.7)
        ax.set_xlabel('Nivel Inicial (Valor Agregado Industrial)', fontsize=11)
        ax.set_ylabel('Crescimento Anual Medio', fontsize=11)
        ax.set_title(f'Convergencia Beta\nbeta={beta_coef:.4f}, IC95%=[{beta_ic[0]:.4f}, {beta_ic[1]:.4f}]', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'convergencia_beta.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 3: Quebras estruturais
    if len(df_quebras) > 0:
        sig_quebras = df_quebras[df_quebras['Significativo'] == 'Sim'].head(15)
        if len(sig_quebras) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.barh(range(len(sig_quebras)), sig_quebras['F_Stat'], color='red', alpha=0.7)
            ax.set_yticks(range(len(sig_quebras)))
            labels = [f"{row['Pais']} ({int(row['Ano_Quebra'])})" for _, row in sig_quebras.iterrows()]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('F-Statistic (Chow Test)')
            ax.set_title(f'Quebras Estruturais Significativas ({n_sig_quebras} paises)', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'quebras_estruturais.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # GRAFICO 4: Clusters geograficos PCA
    if pca_obj is not None and 'PC1' in df_cluster.columns:
        fig, ax = plt.subplots(figsize=(11, 8))
        for cl in df_cluster['Cluster'].unique():
            mask = df_cluster['Cluster'] == cl
            ax.scatter(df_cluster.loc[mask, 'PC1'], df_cluster.loc[mask, 'PC2'], s=60, alpha=0.7, label=f'Cluster {cl}')
            for _, row in df_cluster[mask].iterrows():
                ax.annotate(row['Pais'], (row['PC1'], row['PC2']), fontsize=7, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Clusters Geograficos (PCA)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'clusters_geograficos_pca.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 5: Heatmap regiao x ano
    if 'country_code' in df.columns and 'year' in df.columns and target in df.columns:
        df['Regiao'] = df['country_code'].map(pais_regiao)
        heatmap_data = df.groupby(['Regiao', 'year'])[target].mean().unstack()
        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, linewidths=0.5)
            ax.set_title('Valor Agregado Industrial: Regiao x Ano', fontsize=12, fontweight='bold')
            ax.set_xlabel('Ano')
            ax.set_ylabel('Regiao')
            plt.tight_layout()
            plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'heatmap_regiao_ano.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # GRAFICO 6: Boxplot por regiao
    if 'Regiao' in df.columns and target in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df[df['Regiao'].notna()]
        if len(df_plot) > 0:
            df_plot.boxplot(column=target, by='Regiao', ax=ax)
            ax.set_title('Distribuicao do Target por Regiao', fontsize=12, fontweight='bold')
            ax.set_xlabel('Regiao')
            ax.set_ylabel(target[:30])
            plt.suptitle('')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'boxplot_regioes.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # GRAFICO 7: Mapa de erro por pais
    if len(df_erros) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        df_erros_sorted = df_erros.sort_values('RMSE', ascending=True)
        colors = ['green' if r2 > 0.7 else 'orange' if r2 > 0.4 else 'red' for r2 in df_erros_sorted['R2']]
        ax.barh(range(len(df_erros_sorted)), df_erros_sorted['RMSE'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_erros_sorted)))
        ax.set_yticklabels(df_erros_sorted.index, fontsize=7)
        ax.set_xlabel('RMSE')
        ax.set_title('Erro de Previsao por Pais (Verde:R2>0.7, Laranja:R2>0.4, Vermelho:R2<0.4)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.GEOGRAFICA_DIR, 'mapa_erro_pais.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # METADADOS
    ficheiros_saida = [erros_path, conv_path, quebras_path, clusters_path, regioes_path]
    gerar_metadados(passo='passo8_geografica', descricao='Analise geografica: convergencia beta com IC95%, Chow test, clusters, serie temporal por pais', config=config, dados_entrada=[agg_path], dados_saida=ficheiros_saida, parametros={'n_paises': len(previsoes_pais), 'n_clusters': 3, 'n_bootstrap': 1000}, metricas={'beta_coef': beta_coef, 'beta_p': beta_p, 'convergencia': 'Sim' if (beta_coef and beta_coef < 0 and beta_p < 0.05) else 'Nao', 'quebras_significativas': n_sig_quebras})
    auto_save_drive(ficheiros_saida, config)
    t_total = time.time() - t_inicio
    print(f"\n  RESUMO PASSO 8: {len(previsoes_pais)} paises | beta={beta_coef} | {n_sig_quebras} quebras | 7 graficos | {t_total:.1f}s")
    print("  OK PASSO 8 CONCLUIDO")


if __name__ == '__main__':
    executar_passo8()
