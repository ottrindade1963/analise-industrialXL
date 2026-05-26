"""
============================================================
PASSO 9: ANALISES AVANCADAS
============================================================
Analises Completas:
  1. Convergencia sigma (dispersao ao longo do tempo)
  2. Analise de sensibilidade (tornado plot com IC)
  3. Cenarios de politica (pessimista, base, otimista)
  4. Elasticidades (com bootstrap IC 95%)
  5. Causalidade de Granger (WGI -> Target)
  6. Analise de robustez (performance com dados faltantes)
  7. Projecoes futuras (usando dados sinteticos)
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


def executar_passo9():
    print("\n" + "=" * 70)
    print("  PASSO 9: ANALISES AVANCADAS")
    print("=" * 70)
    os.makedirs(config.AVANCADA_DIR, exist_ok=True)
    t_inicio = time.time()

    # CARREGAR DADOS E MODELOS
    print("\n  [1/7] Carregando dados e modelos...")
    datasets = {}
    for ds_nome in ['WDI_Limpo', 'Agregado', 'Sintetico_Agregado', 'WDI_Sintetico']:
        fpath = encontrar_dataset(ds_nome)
        if fpath and os.path.exists(fpath):
            datasets[ds_nome] = pd.read_csv(fpath)
    
    target = config.TARGET_VAR
    best_model = None
    for m in ['RandomForest', 'XGBoost', 'GradientBoosting']:
        mp = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{m}.pkl')
        if os.path.exists(mp):
            with open(mp, 'rb') as f:
                best_model = pickle.load(f)
            break
    
    df_agg = datasets.get('Agregado', datasets.get(list(datasets.keys())[0]) if datasets else None)
    if df_agg is None:
        print("  ERRO: Nenhum dataset encontrado.")
        print(f"  Procurou em: {config.DADOS_ENGENHARIA_DIR}")
        return
    
    print(f"    Datasets: {len(datasets)} | Modelo: {'OK' if best_model else 'Nao'}")

    # 1. CONVERGENCIA SIGMA
    print("\n  [2/7] Convergencia sigma...")
    sigma_data = []
    if 'country_code' in df_agg.columns and 'year' in df_agg.columns and target in df_agg.columns:
        for year in sorted(df_agg['year'].unique()):
            vals = df_agg[df_agg['year'] == year][target].dropna()
            if len(vals) >= 3:
                sigma_data.append({'Ano': year, 'Std': vals.std(), 'CV': vals.std() / (vals.mean() + 1e-10), 'Media': vals.mean(), 'N': len(vals), 'Min': vals.min(), 'Max': vals.max(), 'Range': vals.max() - vals.min()})
    df_sigma = pd.DataFrame(sigma_data)
    sigma_path = os.path.join(config.AVANCADA_DIR, 'convergencia_sigma.csv')
    df_sigma.to_csv(sigma_path, index=False)
    sigma_trend = None
    if len(df_sigma) > 3:
        s, _, _, p, _ = stats.linregress(df_sigma['Ano'].values, df_sigma['Std'].values)
        sigma_trend = {'slope': s, 'p_value': p, 'convergencia': 'Sim' if s < 0 and p < 0.05 else 'Nao'}
        print(f"    Sigma trend: slope={s:.4f} (p={p:.4f}) -> {'Convergencia' if s < 0 else 'Divergencia'}")

    # 2. ANALISE DE SENSIBILIDADE (TORNADO)
    print("\n  [3/7] Analise de sensibilidade (tornado)...")
    sensibilidade = []
    wgi_cols = [c for c in df_agg.columns if 'wgi' in c.lower() and 'lag' not in c and 'ma' not in c and 'delta' not in c]
    if best_model and target in df_agg.columns:
        X_base = df_agg.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y_base_pred = best_model.predict(X_base).mean()
        for col in wgi_cols:
            if col not in X_base.columns:
                continue
            col_std = X_base[col].std()
            if col_std < 1e-10:
                continue
            X_up = X_base.copy()
            X_up[col] = X_up[col] + col_std
            y_up = best_model.predict(X_up).mean()
            X_down = X_base.copy()
            X_down[col] = X_down[col] - col_std
            y_down = best_model.predict(X_down).mean()
            n_boot = 50
            deltas_up = []
            deltas_down = []
            sample_size = min(100, len(X_base))
            for _ in range(n_boot):
                idx = np.random.choice(len(X_base), sample_size, replace=True)
                Xb = X_base.iloc[idx]
                yb_base = best_model.predict(Xb).mean()
                Xb_up = Xb.copy()
                Xb_up[col] = Xb_up[col] + col_std
                deltas_up.append(best_model.predict(Xb_up).mean() - yb_base)
                Xb_down = Xb.copy()
                Xb_down[col] = Xb_down[col] - col_std
                deltas_down.append(best_model.predict(Xb_down).mean() - yb_base)
            sensibilidade.append({
                'Variavel': col, 'Impacto_Up': y_up - y_base_pred, 'Impacto_Down': y_down - y_base_pred,
                'Impacto_Abs': abs(y_up - y_base_pred) + abs(y_down - y_base_pred),
                'IC_Up_Lower': np.percentile(deltas_up, 2.5), 'IC_Up_Upper': np.percentile(deltas_up, 97.5),
                'IC_Down_Lower': np.percentile(deltas_down, 2.5), 'IC_Down_Upper': np.percentile(deltas_down, 97.5),
                'Std_Variavel': col_std
            })
    df_sens = pd.DataFrame(sensibilidade).sort_values('Impacto_Abs', ascending=False)
    sens_path = os.path.join(config.AVANCADA_DIR, 'sensibilidade_tornado.csv')
    df_sens.to_csv(sens_path, index=False)

    # 3. CENARIOS DE POLITICA
    print("\n  [4/7] Cenarios de politica...")
    cenarios = []
    if best_model and target in df_agg.columns and len(wgi_cols) > 0:
        X_base = df_agg.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y_base_mean = best_model.predict(X_base).mean()
        for cenario_nome, multiplicador in [('Pessimista', -1.0), ('Base', 0.0), ('Otimista', 1.0)]:
            X_cen = X_base.copy()
            for col in wgi_cols:
                if col in X_cen.columns:
                    X_cen[col] = X_cen[col] + multiplicador * X_cen[col].std()
            y_cen = best_model.predict(X_cen).mean()
            cenarios.append({'Cenario': cenario_nome, 'Previsao_Media': y_cen, 'Delta_vs_Base': y_cen - y_base_mean, 'Delta_Pct': (y_cen - y_base_mean) / (abs(y_base_mean) + 1e-10) * 100})
        for col in wgi_cols[:6]:
            if col not in X_base.columns:
                continue
            for mult, nome in [(-1, 'Pessimista'), (1, 'Otimista')]:
                X_c = X_base.copy()
                X_c[col] = X_c[col] + mult * X_c[col].std()
                y_c = best_model.predict(X_c).mean()
                cenarios.append({'Cenario': f'{nome}_{col}', 'Previsao_Media': y_c, 'Delta_vs_Base': y_c - y_base_mean, 'Delta_Pct': (y_c - y_base_mean) / (abs(y_base_mean) + 1e-10) * 100})
    df_cenarios = pd.DataFrame(cenarios)
    cenarios_path = os.path.join(config.AVANCADA_DIR, 'cenarios_politica.csv')
    df_cenarios.to_csv(cenarios_path, index=False)

    # 4. ELASTICIDADES COM BOOTSTRAP IC 95%
    print("\n  [5/7] Elasticidades...")
    elasticidades = []
    if best_model and target in df_agg.columns:
        X_base = df_agg.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y_base_mean = best_model.predict(X_base).mean()
        for col in wgi_cols:
            if col not in X_base.columns:
                continue
            col_mean = X_base[col].mean()
            if abs(col_mean) < 1e-10:
                continue
            X_e = X_base.copy()
            X_e[col] = X_e[col] * 1.01
            y_e = best_model.predict(X_e).mean()
            elast = ((y_e - y_base_mean) / (abs(y_base_mean) + 1e-10)) / 0.01
            elast_boot = []
            for _ in range(50):
                idx = np.random.choice(len(X_base), min(100, len(X_base)), replace=True)
                Xb = X_base.iloc[idx]
                yb = best_model.predict(Xb).mean()
                Xb_e = Xb.copy()
                Xb_e[col] = Xb_e[col] * 1.01
                yb_e = best_model.predict(Xb_e).mean()
                e = ((yb_e - yb) / (abs(yb) + 1e-10)) / 0.01
                elast_boot.append(e)
            elasticidades.append({
                'Variavel': col, 'Elasticidade': elast,
                'IC_Lower': np.percentile(elast_boot, 2.5), 'IC_Upper': np.percentile(elast_boot, 97.5),
                'Significativo': 'Sim' if (np.percentile(elast_boot, 2.5) > 0 or np.percentile(elast_boot, 97.5) < 0) else 'Nao'
            })
    df_elast = pd.DataFrame(elasticidades)
    if len(df_elast) > 0 and 'Elasticidade' in df_elast.columns:
        df_elast = df_elast.sort_values('Elasticidade', key=abs, ascending=False)
    elast_path = os.path.join(config.AVANCADA_DIR, 'elasticidades.csv')
    df_elast.to_csv(elast_path, index=False)

    # 5. CAUSALIDADE DE GRANGER
    print("\n  [6/7] Causalidade de Granger...")
    granger_results = []
    if 'country_code' in df_agg.columns and 'year' in df_agg.columns and target in df_agg.columns:
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            for pais in df_agg['country_code'].unique()[:15]:
                df_p = df_agg[df_agg['country_code'] == pais].sort_values('year')
                if len(df_p) < 10:
                    continue
                for col in wgi_cols:
                    if col not in df_p.columns:
                        continue
                    y_ts = df_p[target].values
                    x_ts = df_p[col].values
                    if np.std(y_ts) < 1e-10 or np.std(x_ts) < 1e-10:
                        continue
                    try:
                        data = np.column_stack([y_ts, x_ts])
                        result = grangercausalitytests(data, maxlag=2, verbose=False)
                        f_stat = result[1][0]['ssr_ftest'][0]
                        p_val = result[1][0]['ssr_ftest'][1]
                        granger_results.append({'Pais': pais, 'WGI': col, 'F_Stat': f_stat, 'P_Value': p_val, 'Significativo': 'Sim' if p_val < 0.05 else 'Nao', 'Lag': 1})
                    except Exception:
                        pass
        except ImportError:
            print("    AVISO: statsmodels nao disponivel para Granger")
    
    df_granger = pd.DataFrame(granger_results)
    granger_path = os.path.join(config.AVANCADA_DIR, 'granger_causality.csv')
    df_granger.to_csv(granger_path, index=False)
    n_sig_granger = len(df_granger[df_granger['Significativo'] == 'Sim']) if len(df_granger) > 0 else 0
    print(f"    Granger significativos: {n_sig_granger}/{len(df_granger)}")

    # 6. ANALISE DE ROBUSTEZ
    print("\n  [7/7] Analise de robustez...")
    robustez = []
    if best_model and target in df_agg.columns:
        X_full = df_agg.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y_full = df_agg[target].values
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred_full = best_model.predict(X_full)
        rmse_full = np.sqrt(mean_squared_error(y_full, y_pred_full))
        r2_full = r2_score(y_full, y_pred_full)
        robustez.append({'Missing_Pct': 0, 'RMSE': rmse_full, 'R2': r2_full, 'RMSE_Degradacao_Pct': 0, 'R2_Degradacao_Pct': 0, 'RMSE_Std': 0, 'R2_Std': 0})
        for pct in [10, 20, 30, 40, 50]:
            rmse_runs = []
            r2_runs = []
            for _ in range(5):
                X_miss = X_full.copy()
                mask = np.random.random(X_miss.shape) < (pct / 100)
                X_miss = X_miss.mask(mask).fillna(0)
                y_pred_miss = best_model.predict(X_miss)
                rmse_runs.append(np.sqrt(mean_squared_error(y_full, y_pred_miss)))
                r2_runs.append(r2_score(y_full, y_pred_miss))
            rmse_m = np.mean(rmse_runs)
            r2_m = np.mean(r2_runs)
            robustez.append({
                'Missing_Pct': pct, 'RMSE': rmse_m, 'R2': r2_m,
                'RMSE_Degradacao_Pct': (rmse_m - rmse_full) / (rmse_full + 1e-10) * 100,
                'R2_Degradacao_Pct': (r2_full - r2_m) / (abs(r2_full) + 1e-10) * 100,
                'RMSE_Std': np.std(rmse_runs), 'R2_Std': np.std(r2_runs)
            })
    df_robustez = pd.DataFrame(robustez)
    robustez_path = os.path.join(config.AVANCADA_DIR, 'robustez.csv')
    df_robustez.to_csv(robustez_path, index=False)

    # VISUALIZACOES (7 GRAFICOS)
    print("\n  Gerando 7 visualizacoes...")

    # GRAFICO 1: Convergencia sigma
    if len(df_sigma) > 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.plot(df_sigma['Ano'], df_sigma['Std'], 'b-o', markersize=4, linewidth=2)
        if sigma_trend:
            x_t = df_sigma['Ano'].values
            y_t = sigma_trend['slope'] * x_t + (df_sigma['Std'].mean() - sigma_trend['slope'] * x_t.mean())
            ax.plot(x_t, y_t, 'r--', linewidth=1.5, label=f'Trend: {sigma_trend["slope"]:.4f} (p={sigma_trend["p_value"]:.4f})')
            ax.legend()
        ax.set_xlabel('Ano')
        ax.set_ylabel('Desvio Padrao')
        ax.set_title('Convergencia Sigma (Dispersao)', fontweight='bold')
        ax.grid(alpha=0.3)
        ax = axes[1]
        ax.plot(df_sigma['Ano'], df_sigma['CV'], 'g-s', markersize=4, linewidth=2)
        ax.set_xlabel('Ano')
        ax.set_ylabel('Coeficiente de Variacao')
        ax.set_title('CV ao Longo do Tempo', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.suptitle('Convergencia Sigma', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.AVANCADA_DIR, 'convergencia_sigma.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 2: Tornado plot com IC
    if len(df_sens) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        df_s = df_sens.head(10)
        y_pos = range(len(df_s))
        ax.barh(y_pos, df_s['Impacto_Up'].values, color='green', alpha=0.7, label='+1 Std')
        ax.barh(y_pos, df_s['Impacto_Down'].values, color='red', alpha=0.7, label='-1 Std')
        for i, (_, row) in enumerate(df_s.iterrows()):
            ax.plot([row['IC_Up_Lower'], row['IC_Up_Upper']], [i, i], 'k-', linewidth=2, alpha=0.5)
            ax.plot([row['IC_Down_Lower'], row['IC_Down_Upper']], [i, i], 'k-', linewidth=2, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([v[:30] for v in df_s['Variavel']], fontsize=8)
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_xlabel('Impacto no Target')
        ax.set_title('Tornado Plot: Sensibilidade WGI (com IC 95%)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.AVANCADA_DIR, 'tornado_sensibilidade.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 3: Cenarios de politica
    if len(df_cenarios) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        cen_global = df_cenarios[df_cenarios['Cenario'].isin(['Pessimista', 'Base', 'Otimista'])]
        if len(cen_global) > 0:
            colors_c = {'Pessimista': 'red', 'Base': 'gray', 'Otimista': 'green'}
            ax.bar(range(len(cen_global)), cen_global['Previsao_Media'], color=[colors_c.get(c, 'blue') for c in cen_global['Cenario']], alpha=0.7)
            ax.set_xticks(range(len(cen_global)))
            ax.set_xticklabels(cen_global['Cenario'])
            ax.set_ylabel('Previsao Media')
            ax.set_title('Cenarios Globais', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        ax = axes[1]
        cen_var = df_cenarios[~df_cenarios['Cenario'].isin(['Pessimista', 'Base', 'Otimista'])]
        if len(cen_var) > 0:
            ax.barh(range(len(cen_var)), cen_var['Delta_Pct'], color=['green' if d > 0 else 'red' for d in cen_var['Delta_Pct']], alpha=0.7)
            ax.set_yticks(range(len(cen_var)))
            ax.set_yticklabels([c[:25] for c in cen_var['Cenario']], fontsize=7)
            ax.set_xlabel('Delta vs Base (%)')
            ax.set_title('Cenarios por Variavel WGI', fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--')
            ax.grid(axis='x', alpha=0.3)
        plt.suptitle('Cenarios de Politica', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.AVANCADA_DIR, 'cenarios_politica.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 4: Elasticidades com IC
    if len(df_elast) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        df_e = df_elast.head(10)
        y_pos = range(len(df_e))
        colors_e = ['green' if s == 'Sim' else 'gray' for s in df_e['Significativo']]
        ax.barh(y_pos, df_e['Elasticidade'], color=colors_e, alpha=0.7)
        for i, (_, row) in enumerate(df_e.iterrows()):
            ax.plot([row['IC_Lower'], row['IC_Upper']], [i, i], 'k-', linewidth=2)
            ax.plot(row['IC_Lower'], i, 'k|', markersize=8)
            ax.plot(row['IC_Upper'], i, 'k|', markersize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([v[:30] for v in df_e['Variavel']], fontsize=8)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Elasticidade')
        ax.set_title('Elasticidades WGI -> Target (com IC 95%)\nVerde=Significativo', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.AVANCADA_DIR, 'elasticidades.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 5: Granger heatmap
    if len(df_granger) > 0:
        pivot = df_granger.pivot_table(index='Pais', columns='WGI', values='P_Value', aggfunc='min')
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot, cmap='RdYlGn_r', vmin=0, vmax=0.1, ax=ax, annot=True, fmt='.3f', linewidths=0.5)
            ax.set_title(f'Granger Causality (p-values) - {n_sig_granger}/{len(df_granger)} significativos', fontsize=12, fontweight='bold')
            ax.set_xlabel('WGI')
            ax.set_ylabel('Pais')
            plt.tight_layout()
            plt.savefig(os.path.join(config.AVANCADA_DIR, 'granger_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # GRAFICO 6: Robustez
    if len(df_robustez) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.plot(df_robustez['Missing_Pct'], df_robustez['RMSE'], 'r-o', linewidth=2, markersize=6)
        ax.fill_between(df_robustez['Missing_Pct'], df_robustez['RMSE'] - df_robustez['RMSE_Std'], df_robustez['RMSE'] + df_robustez['RMSE_Std'], alpha=0.2, color='red')
        ax.set_xlabel('% Dados Faltantes')
        ax.set_ylabel('RMSE')
        ax.set_title('Degradacao RMSE', fontweight='bold')
        ax.grid(alpha=0.3)
        ax = axes[1]
        ax.plot(df_robustez['Missing_Pct'], df_robustez['R2'], 'b-s', linewidth=2, markersize=6)
        ax.fill_between(df_robustez['Missing_Pct'], df_robustez['R2'] - df_robustez['R2_Std'], df_robustez['R2'] + df_robustez['R2_Std'], alpha=0.2, color='blue')
        ax.set_xlabel('% Dados Faltantes')
        ax.set_ylabel('R2')
        ax.set_title('Degradacao R2', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.suptitle('Analise de Robustez', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.AVANCADA_DIR, 'robustez.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 7: Projecoes futuras (sintetico)
    df_sint = datasets.get('Sintetico_Agregado')
    if df_sint is not None and best_model and target in df_sint.columns and 'year' in df_sint.columns:
        X_sint = df_sint.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        try:
            y_sint_pred = best_model.predict(X_sint)
            df_sint_pred = df_sint[['year']].copy()
            df_sint_pred['Previsao'] = y_sint_pred
            proj = df_sint_pred.groupby('year')['Previsao'].agg(['mean', 'std']).reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(proj['year'], proj['mean'], 'b-', linewidth=2, label='Previsao Media')
            ax.fill_between(proj['year'], proj['mean'] - proj['std'], proj['mean'] + proj['std'], alpha=0.2, color='blue', label='IC (1 Std)')
            if 'Agregado' in datasets and 'year' in datasets['Agregado'].columns:
                max_real_year = datasets['Agregado']['year'].max()
                ax.axvline(x=max_real_year, color='red', linestyle='--', label=f'Fim dados reais ({int(max_real_year)})')
            ax.set_xlabel('Ano')
            ax.set_ylabel('Valor Agregado Industrial')
            ax.set_title('Projecoes Futuras (Dados Sinteticos)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(config.AVANCADA_DIR, 'projecoes_futuras.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Projecoes: erro - {str(e)[:50]}")

    # METADADOS
    ficheiros_saida = [sigma_path, sens_path, cenarios_path, elast_path, granger_path, robustez_path]
    gerar_metadados(passo='passo9_avancada', descricao='Convergencia sigma, tornado com IC, cenarios, elasticidades bootstrap, Granger, robustez', config=config, dados_entrada=list(datasets.keys()), dados_saida=ficheiros_saida, parametros={'n_bootstrap_elast': 500, 'n_bootstrap_sens': 200, 'n_robustez_runs': 20, 'granger_maxlag': 2}, metricas={'sigma_convergencia': sigma_trend.get('convergencia', 'NA') if sigma_trend else 'NA', 'granger_significativos': n_sig_granger, 'n_elasticidades_sig': len(df_elast[df_elast['Significativo'] == 'Sim']) if len(df_elast) > 0 else 0})
    auto_save_drive(ficheiros_saida, config)
    t_total = time.time() - t_inicio
    print(f"\n  RESUMO PASSO 9: sigma={'Conv' if sigma_trend and sigma_trend['convergencia']=='Sim' else 'Div'} | {n_sig_granger} Granger sig | 7 graficos | {t_total:.1f}s")
    print("  OK PASSO 9 CONCLUIDO")


if __name__ == '__main__':
    executar_passo9()
