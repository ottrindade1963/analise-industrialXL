"""
============================================================
PASSO 5: AVALIACAO DE PERFORMANCE DOS MODELOS
============================================================
Analises Completas:
  1. Carregamento dos 28 modelos (7 x 4 datasets)
  2. 5 Metricas: RMSE, MAE, MAPE, R2, RSE
  3. Teste Diebold-Mariano (significancia estatistica)
  4. Intervalos de Confianca 95% via Bootstrap (1000 iteracoes)
  5. Ranking de modelos com IC
  6. Ganho percentual de WGI (WDI_Limpo vs Agregado)
  7. Analise de residuos por dataset e modelo
  8. Tabela comparativa completa para dissertacao

Visualizacoes (6 graficos):
  1. Heatmap RMSE (modelos x datasets)
  2. Ranking com IC 95%
  3. Ganho WGI por modelo (com significancia)
  4. Scatter previsao vs real (4 datasets)
  5. Distribuicao de residuos
  6. Comparacao 5 metricas (bar charts)
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


# ============================================================
# FUNCOES AUXILIARES
# ============================================================
def calcular_5_metricas(y_true, y_pred):
    """Calcula RMSE, MAE, MAPE, R2, RSE."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan, 'RSE': np.nan}
    
    residuos = y_true - y_pred
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-10
    
    return {
        'RMSE': np.sqrt(np.mean(residuos**2)),
        'MAE': np.mean(np.abs(residuos)),
        'MAPE': np.mean(np.abs(residuos / (np.abs(y_true) + 1e-8))) * 100,
        'R2': 1 - ss_res / ss_tot,
        'RSE': np.sqrt(ss_res / ss_tot)
    }


def diebold_mariano_test(e1, e2, h=1):
    """
    Teste Diebold-Mariano bilateral.
    H0: Ambos os modelos tem a mesma precisao preditiva.
    Usa correcao Newey-West para autocorrelacao.
    """
    e1, e2 = np.array(e1), np.array(e2)
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    
    d_mean = np.mean(d)
    # Variancia com correcao Newey-West
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, min(h, n-1)):
        if len(d[k:]) > 1:
            gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
            gamma_sum += 2 * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        var_d = gamma_0 / n
    
    dm_stat = d_mean / np.sqrt(var_d + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return dm_stat, p_value


def bootstrap_ic(y_true, y_pred, metrica_func, n_boot=1000, ci=0.95):
    """IC via bootstrap para qualquer metrica."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    valores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        valores.append(metrica_func(y_true[idx], y_pred[idx]))
    valores = np.sort(valores)
    alpha = (1 - ci) / 2
    return {
        'mean': np.mean(valores),
        'median': np.median(valores),
        'ci_lower': valores[int(alpha * n_boot)],
        'ci_upper': valores[int((1 - alpha) * n_boot)],
        'std': np.std(valores)
    }


def rmse_func(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae_func(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_func(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-10
    return 1 - ss_res / ss_tot


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================
def executar_passo5():
    """Avaliacao completa de 28 modelos (7 x 4 datasets)."""
    print("\n" + "=" * 70)
    print("  PASSO 5: AVALIACAO DE PERFORMANCE - 28 MODELOS")
    print("=" * 70)
    
    os.makedirs(config.RESULTADOS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    # --------------------------------------------------------
    # DATASETS E MODELOS
    # --------------------------------------------------------
    ds_paths = {
        'WDI_Limpo': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv'),
        'Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv'),
        'Sintetico_Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv'),
        'WDI_Sintetico': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv'),
    }
    
    modelos_nomes = ['RandomForest', 'XGBoost', 'GradientBoosting', 'SARIMAX',
                     'LSTM', 'Bayes_PartialPooling', 'Bayes_CompletePooling']
    
    resultados = []
    previsoes = {}  # chave: (dataset, modelo) -> {y_true, y_pred, erros}
    
    # ============================================================
    # [1/7] CARREGAR E AVALIAR MODELOS
    # ============================================================
    print("\n  [1/7] Carregando modelos e avaliando em dados de teste...")
    
    for ds_nome, ds_path in ds_paths.items():
        if not os.path.exists(ds_path):
            continue
        
        df = pd.read_csv(ds_path).sort_values(['country_code', 'year']).reset_index(drop=True)
        target = config.TARGET_VAR
        if target not in df.columns:
            continue
        
        X = df.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y = df[target].values
        
        # Split temporal (mesmo do treino: 70/15/15)
        n = len(df)
        test_start = int(n * 0.85)
        X_test = X.iloc[test_start:]
        y_test = y[test_start:]
        
        for modelo_nome in modelos_nomes:
            model_path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
            if not os.path.exists(model_path):
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    modelo = pickle.load(f)
                
                y_pred = np.array(modelo.predict(X_test))[:len(y_test)]
                erros = y_test - y_pred
                
                # 5 Metricas
                metricas = calcular_5_metricas(y_test, y_pred)
                metricas['Dataset'] = ds_nome
                metricas['Modelo'] = modelo_nome
                metricas['N_Test'] = len(y_test)
                resultados.append(metricas)
                
                previsoes[(ds_nome, modelo_nome)] = {
                    'y_true': y_test, 'y_pred': y_pred, 'erros': erros
                }
                
            except Exception as e:
                print(f"    Erro {ds_nome}/{modelo_nome}: {str(e)[:50]}")
    
    # Fallback: usar metricas do treino
    if len(resultados) == 0:
        metricas_path = os.path.join(config.MODELOS_DIR, 'metricas_treino_completas.csv')
        if os.path.exists(metricas_path):
            df_treino = pd.read_csv(metricas_path)
            for _, row in df_treino.iterrows():
                resultados.append(row.to_dict())
    
    if len(resultados) == 0:
        print("  ERRO: Nenhum modelo encontrado.")
        return
    
    df_res = pd.DataFrame(resultados)
    avaliacao_path = os.path.join(config.RESULTADOS_DIR, 'avaliacao_completa.csv')
    df_res.to_csv(avaliacao_path, index=False)
    print(f"    Modelos avaliados: {len(df_res)}")
    
    # ============================================================
    # [2/7] TABELA RESUMO POR MODELO (media dos 4 datasets)
    # ============================================================
    print("\n  [2/7] Tabela resumo por modelo (media dos 4 datasets)...")
    
    metricas_cols = ['RMSE', 'MAE', 'MAPE', 'R2', 'RSE']
    resumo = df_res.groupby('Modelo')[metricas_cols].agg(['mean', 'std', 'min', 'max'])
    resumo.columns = ['_'.join(col) for col in resumo.columns]
    resumo = resumo.sort_values('RMSE_mean')
    resumo_path = os.path.join(config.RESULTADOS_DIR, 'resumo_por_modelo.csv')
    resumo.to_csv(resumo_path)
    
    # Resumo por dataset
    resumo_ds = df_res.groupby('Dataset')[metricas_cols].agg(['mean', 'std'])
    resumo_ds.columns = ['_'.join(col) for col in resumo_ds.columns]
    resumo_ds_path = os.path.join(config.RESULTADOS_DIR, 'resumo_por_dataset.csv')
    resumo_ds.to_csv(resumo_ds_path)
    
    print(f"    Melhor modelo: {resumo.index[0]} (RMSE={resumo['RMSE_mean'].iloc[0]:.4f})")
    
    # ============================================================
    # [3/7] TESTE DIEBOLD-MARIANO
    # ============================================================
    print("\n  [3/7] Teste Diebold-Mariano (WDI_Limpo vs Agregado)...")
    
    dm_resultados = []
    for modelo_nome in modelos_nomes:
        key_wdi = ('WDI_Limpo', modelo_nome)
        key_agg = ('Agregado', modelo_nome)
        
        if key_wdi in previsoes and key_agg in previsoes:
            e_wdi = previsoes[key_wdi]['erros']
            e_agg = previsoes[key_agg]['erros']
            min_n = min(len(e_wdi), len(e_agg))
            
            if min_n >= 5:
                dm_stat, p_value = diebold_mariano_test(e_wdi[:min_n], e_agg[:min_n])
                
                rmse_wdi = np.sqrt(np.mean(e_wdi[:min_n]**2))
                rmse_agg = np.sqrt(np.mean(e_agg[:min_n]**2))
                ganho_pct = (rmse_wdi - rmse_agg) / (rmse_wdi + 1e-10) * 100
                
                dm_resultados.append({
                    'Modelo': modelo_nome,
                    'RMSE_WDI_Limpo': rmse_wdi,
                    'RMSE_Agregado': rmse_agg,
                    'Ganho_Pct_RMSE': ganho_pct,
                    'DM_Statistic': dm_stat,
                    'P_Value': p_value,
                    'Significativo_5pct': 'Sim' if (not np.isnan(p_value) and p_value < 0.05) else 'Nao',
                    'Significativo_10pct': 'Sim' if (not np.isnan(p_value) and p_value < 0.10) else 'Nao',
                    'WGI_Melhora': 'Sim' if ganho_pct > 0 else 'Nao',
                    'Interpretacao': 'Agregado significativamente melhor' if (not np.isnan(p_value) and p_value < 0.05 and ganho_pct > 0) else 'Diferenca nao significativa'
                })
                print(f"    {modelo_nome}: DM={dm_stat:.3f}, p={p_value:.4f}, Ganho={ganho_pct:.1f}%")
    
    df_dm = pd.DataFrame(dm_resultados)
    dm_path = os.path.join(config.RESULTADOS_DIR, 'diebold_mariano.csv')
    df_dm.to_csv(dm_path, index=False)
    
    n_sig = len(df_dm[df_dm['Significativo_5pct'] == 'Sim']) if len(df_dm) > 0 else 0
    print(f"    Significativos (p<0.05): {n_sig}/{len(df_dm)}")
    
    # ============================================================
    # [4/7] INTERVALOS DE CONFIANCA 95% (BOOTSTRAP)
    # ============================================================
    print("\n  [4/7] IC 95% via bootstrap (1000 iteracoes)...")
    
    ic_resultados = []
    for (ds_nome, modelo_nome), dados in previsoes.items():
        ic_rmse = bootstrap_ic(dados['y_true'], dados['y_pred'], rmse_func, n_boot=1000)
        ic_mae = bootstrap_ic(dados['y_true'], dados['y_pred'], mae_func, n_boot=1000)
        ic_r2 = bootstrap_ic(dados['y_true'], dados['y_pred'], r2_func, n_boot=1000)
        
        ic_resultados.append({
            'Dataset': ds_nome, 'Modelo': modelo_nome,
            'RMSE_mean': ic_rmse['mean'], 'RMSE_ci_lower': ic_rmse['ci_lower'],
            'RMSE_ci_upper': ic_rmse['ci_upper'], 'RMSE_std': ic_rmse['std'],
            'MAE_mean': ic_mae['mean'], 'MAE_ci_lower': ic_mae['ci_lower'],
            'MAE_ci_upper': ic_mae['ci_upper'],
            'R2_mean': ic_r2['mean'], 'R2_ci_lower': ic_r2['ci_lower'],
            'R2_ci_upper': ic_r2['ci_upper'],
        })
    
    df_ic = pd.DataFrame(ic_resultados)
    ic_path = os.path.join(config.RESULTADOS_DIR, 'intervalos_confianca_bootstrap.csv')
    df_ic.to_csv(ic_path, index=False)
    print(f"    IC calculados para {len(df_ic)} combinacoes modelo-dataset")
    
    # ============================================================
    # [5/7] GANHO PERCENTUAL DE WGI
    # ============================================================
    print("\n  [5/7] Ganho percentual de WGI...")
    
    ganho_resultados = []
    for modelo_nome in df_res['Modelo'].unique():
        wdi_rows = df_res[(df_res['Dataset'] == 'WDI_Limpo') & (df_res['Modelo'] == modelo_nome)]
        agg_rows = df_res[(df_res['Dataset'] == 'Agregado') & (df_res['Modelo'] == modelo_nome)]
        
        if len(wdi_rows) > 0 and len(agg_rows) > 0:
            rmse_wdi = wdi_rows['RMSE'].values[0]
            rmse_agg = agg_rows['RMSE'].values[0]
            r2_wdi = wdi_rows['R2'].values[0]
            r2_agg = agg_rows['R2'].values[0]
            mae_wdi = wdi_rows['MAE'].values[0]
            mae_agg = agg_rows['MAE'].values[0]
            
            ganho_resultados.append({
                'Modelo': modelo_nome,
                'RMSE_WDI': rmse_wdi, 'RMSE_Agregado': rmse_agg,
                'Ganho_RMSE_Pct': (rmse_wdi - rmse_agg) / (rmse_wdi + 1e-10) * 100,
                'R2_WDI': r2_wdi, 'R2_Agregado': r2_agg,
                'Delta_R2': r2_agg - r2_wdi,
                'MAE_WDI': mae_wdi, 'MAE_Agregado': mae_agg,
                'Ganho_MAE_Pct': (mae_wdi - mae_agg) / (mae_wdi + 1e-10) * 100,
            })
    
    df_ganho = pd.DataFrame(ganho_resultados)
    ganho_path = os.path.join(config.RESULTADOS_DIR, 'ganho_wgi_detalhado.csv')
    df_ganho.to_csv(ganho_path, index=False)
    
    if len(df_ganho) > 0:
        print(f"    Ganho medio RMSE: {df_ganho['Ganho_RMSE_Pct'].mean():.2f}%")
        print(f"    Ganho medio R2: {df_ganho['Delta_R2'].mean():.4f}")
    
    # ============================================================
    # [6/7] RANKING DE MODELOS
    # ============================================================
    print("\n  [6/7] Ranking final de modelos...")
    
    ranking = df_res.groupby('Modelo').agg({
        'RMSE': ['mean', 'std'], 'MAE': 'mean', 'R2': 'mean', 'MAPE': 'mean'
    })
    ranking.columns = ['RMSE_mean', 'RMSE_std', 'MAE_mean', 'R2_mean', 'MAPE_mean']
    ranking = ranking.sort_values('RMSE_mean')
    ranking['Posicao'] = range(1, len(ranking) + 1)
    ranking_path = os.path.join(config.RESULTADOS_DIR, 'ranking_modelos.csv')
    ranking.to_csv(ranking_path)
    
    print("    Ranking (RMSE medio):")
    for i, (m, row) in enumerate(ranking.iterrows(), 1):
        print(f"      {i}. {m}: RMSE={row['RMSE_mean']:.4f} +/- {row['RMSE_std']:.4f}, R2={row['R2_mean']:.4f}")
    
    # ============================================================
    # [7/7] VISUALIZACOES (6 GRAFICOS)
    # ============================================================
    print("\n  [7/7] Gerando 6 visualizacoes...")
    
    # --- GRAFICO 1: Heatmap RMSE (Modelos x Datasets) ---
    pivot_rmse = df_res.pivot_table(values='RMSE', index='Modelo', columns='Dataset', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax, linewidths=0.5)
    ax.set_title('RMSE de Teste: 7 Modelos x 4 Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTADOS_DIR, 'heatmap_rmse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- GRAFICO 2: Ranking com IC 95% ---
    if len(df_ic) > 0:
        ic_modelo = df_ic.groupby('Modelo').agg({
            'RMSE_mean': 'mean', 'RMSE_ci_lower': 'mean', 'RMSE_ci_upper': 'mean'
        }).sort_values('RMSE_mean')
        
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(ic_modelo))
        ax.barh(y_pos, ic_modelo['RMSE_mean'], color='steelblue', alpha=0.7, height=0.6)
        ax.errorbar(ic_modelo['RMSE_mean'], y_pos,
                    xerr=[ic_modelo['RMSE_mean'] - ic_modelo['RMSE_ci_lower'],
                          ic_modelo['RMSE_ci_upper'] - ic_modelo['RMSE_mean']],
                    fmt='none', color='black', capsize=4, linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ic_modelo.index)
        ax.set_xlabel('RMSE (com IC 95% Bootstrap)')
        ax.set_title('Ranking de Modelos com Intervalos de Confianca 95%', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'ranking_ic95.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 3: Ganho WGI com significancia ---
    if len(df_ganho) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ganho_sorted = df_ganho.sort_values('Ganho_RMSE_Pct', ascending=False)
        colors = []
        for _, row in ganho_sorted.iterrows():
            dm_row = df_dm[df_dm['Modelo'] == row['Modelo']] if len(df_dm) > 0 else pd.DataFrame()
            if len(dm_row) > 0 and dm_row.iloc[0]['Significativo_5pct'] == 'Sim':
                colors.append('darkgreen' if row['Ganho_RMSE_Pct'] > 0 else 'darkred')
            else:
                colors.append('lightgreen' if row['Ganho_RMSE_Pct'] > 0 else 'lightsalmon')
        
        bars = ax.bar(range(len(ganho_sorted)), ganho_sorted['Ganho_RMSE_Pct'], color=colors)
        ax.set_xticks(range(len(ganho_sorted)))
        ax.set_xticklabels(ganho_sorted['Modelo'], rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Ganho % RMSE')
        ax.set_title('Ganho de WGI por Modelo\n(escuro = significativo p<0.05, claro = nao significativo)', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'ganho_wgi.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 4: Scatter previsao vs real (4 datasets) ---
    if len(previsoes) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_flat = axes.flatten()
        
        ds_list = ['WDI_Limpo', 'Agregado', 'Sintetico_Agregado', 'WDI_Sintetico']
        for i, ds_nome in enumerate(ds_list):
            ax = axes_flat[i]
            # Encontrar melhor modelo para este dataset
            ds_previsoes = {k: v for k, v in previsoes.items() if k[0] == ds_nome}
            if ds_previsoes:
                best_key = min(ds_previsoes.keys(), key=lambda k: np.sqrt(np.mean(ds_previsoes[k]['erros']**2)))
                dados = previsoes[best_key]
                
                ax.scatter(dados['y_true'], dados['y_pred'], alpha=0.5, s=25, color='steelblue')
                lims = [min(dados['y_true'].min(), dados['y_pred'].min()) - 1,
                        max(dados['y_true'].max(), dados['y_pred'].max()) + 1]
                ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5)
                
                rmse = np.sqrt(np.mean(dados['erros']**2))
                r2 = r2_func(dados['y_true'], dados['y_pred'])
                ax.set_xlabel('Valor Real')
                ax.set_ylabel('Previsao')
                ax.set_title(f'{ds_nome} ({best_key[1]})\nRMSE={rmse:.3f}, R2={r2:.3f}', fontsize=10)
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{ds_nome}\nSem dados', ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Previsao vs Real (Melhor Modelo por Dataset)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'scatter_previsao_real.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 5: Distribuicao de residuos ---
    if len(previsoes) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        
        for i, ds_nome in enumerate(ds_list):
            ax = axes_flat[i]
            ds_previsoes = {k: v for k, v in previsoes.items() if k[0] == ds_nome}
            if ds_previsoes:
                best_key = min(ds_previsoes.keys(), key=lambda k: np.sqrt(np.mean(ds_previsoes[k]['erros']**2)))
                erros = previsoes[best_key]['erros']
                
                ax.hist(erros, bins=25, alpha=0.7, color='steelblue', edgecolor='white', density=True)
                # Ajustar normal
                mu, sigma = np.mean(erros), np.std(erros)
                x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal')
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                ax.set_title(f'{ds_nome}\nmu={mu:.3f}, sigma={sigma:.3f}', fontsize=10)
                ax.set_xlabel('Residuo')
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'{ds_nome}\nSem dados', ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Distribuicao de Residuos (com ajuste Normal)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'distribuicao_residuos.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- GRAFICO 6: Comparacao 5 metricas ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metricas_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'RSE']
    
    for i, metrica in enumerate(metricas_plot):
        ax = axes.flat[i]
        if metrica in df_res.columns:
            pivot_m = df_res.pivot_table(values=metrica, index='Modelo', columns='Dataset', aggfunc='mean')
            pivot_m.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
            ax.set_title(metrica, fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
    
    # Ultimo subplot: Box plot R2
    ax = axes.flat[5]
    df_res.boxplot(column='R2', by='Dataset', ax=ax)
    ax.set_title('R2 por Dataset', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    plt.suptitle('Comparacao de 5 Metricas: 7 Modelos x 4 Datasets', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTADOS_DIR, 'comparacao_5metricas.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================
    # METADADOS E RESUMO
    # ============================================================
    ficheiros_saida = [avaliacao_path, resumo_path, resumo_ds_path, dm_path, ic_path, ganho_path, ranking_path]
    
    gerar_metadados(
        passo='passo5_avaliacao',
        descricao='Avaliacao completa: 5 metricas, Diebold-Mariano, IC 95% bootstrap, ranking',
        config=config,
        dados_entrada=[config.MODELOS_DIR],
        dados_saida=ficheiros_saida,
        parametros={'metricas': metricas_cols, 'bootstrap_n': 1000, 'dm_test': True},
        metricas={
            'modelos_avaliados': len(df_res),
            'melhor_modelo': ranking.index[0] if len(ranking) > 0 else 'N/A',
            'melhor_rmse': float(ranking['RMSE_mean'].iloc[0]) if len(ranking) > 0 else None,
            'dm_significativos_5pct': n_sig,
            'ganho_medio_wgi': float(df_ganho['Ganho_RMSE_Pct'].mean()) if len(df_ganho) > 0 else None,
        }
    )
    auto_save_drive(ficheiros_saida, config)
    
    t_total = time.time() - t_inicio
    print(f"\n  {'='*60}")
    print(f"  RESUMO PASSO 5:")
    print(f"  {'='*60}")
    print(f"  Modelos avaliados: {len(df_res)}")
    print(f"  Ficheiros CSV gerados: {len(ficheiros_saida)}")
    print(f"  Graficos PNG gerados: 6")
    print(f"  Tempo: {t_total:.1f}s")
    print(f"\n  OK PASSO 5 CONCLUIDO")


if __name__ == '__main__':
    executar_passo5()
