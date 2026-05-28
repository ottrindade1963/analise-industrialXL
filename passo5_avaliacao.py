"""
============================================================
PASSO 5: AVALIACAO DE PERFORMANCE DOS MODELOS
============================================================
Analises Completas:
  1. Carregamento dos 28 modelos (7 x 4 datasets)
  2. 8 Metricas: RMSE, MAE, MAPE, R2, RSE, MASE, sMAPE, AIC
  3. Baselines ingenuas (persistencia, media movel, drift)
  4. Teste Diebold-Mariano (significancia estatistica)
  5. Intervalos de Confianca 95% via Bootstrap (1000 iteracoes)
  6. Ranking de modelos com IC (ranking composto multi-metrica)
  7. Ganho percentual de WGI (WDI_Limpo vs Agregado)
  8. Analise de residuos por dataset e modelo
  9. Tabela comparativa completa para dissertacao

Visualizacoes (8 graficos):
  1. Heatmap RMSE (modelos x datasets)
  2. Ranking com IC 95%
  3. Ganho WGI por modelo (com significancia)
  4. Scatter previsao vs real (4 datasets)
  5. Distribuicao de residuos
  6. Comparacao 8 metricas (bar charts)
  7. Baselines vs Modelos (bar chart comparativo)
  8. Ranking composto multi-metrica

Recomendacoes Implementadas:
  - Rec. 7: Baselines ingenuas (Persistencia, Media Movel, Drift)
  - Rec. 12: MASE, sMAPE, AIC + ranking composto multi-metrica
  - Rec. 14: Docstrings completas com Parameters/Returns/Notes

References:
  - Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
  - Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of
    forecast accuracy. International Journal of Forecasting, 22(4).
  - Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
  - Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles
    and Practice, 3rd edition. OTexts.
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
    """
    Calcula as 5 metricas base de avaliacao de modelos preditivos.

    Parameters
    ----------
    y_true : array-like
        Valores reais observados da variavel dependente.
    y_pred : array-like
        Valores previstos pelo modelo.

    Returns
    -------
    dict
        Dicionario com RMSE, MAE, MAPE, R2 e RSE.
        Retorna NaN para todas as metricas se n < 3.

    Notes
    -----
    - RMSE = sqrt(mean((y - yhat)^2)): penaliza erros grandes.
    - MAE = mean(|y - yhat|): robusto a outliers.
    - MAPE = mean(|y - yhat| / |y|) * 100: escala relativa (%).
    - R2 = 1 - SS_res / SS_tot: variancia explicada.
    - RSE = sqrt(SS_res / SS_tot): erro relativo quadrado.
    """
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


def calcular_metricas_avancadas(y_true, y_pred, y_train=None, n_params=None):
    """
    Calcula metricas avancadas: MASE, sMAPE e AIC (Rec. 12).

    Parameters
    ----------
    y_true : array-like
        Valores reais observados no conjunto de teste.
    y_pred : array-like
        Valores previstos pelo modelo no conjunto de teste.
    y_train : array-like, optional
        Valores reais do conjunto de treino (necessario para MASE).
        Se None, MASE usa naive no proprio teste.
    n_params : int, optional
        Numero de parametros do modelo (para AIC).

    Returns
    -------
    dict
        Dicionario com MASE, sMAPE e opcionalmente AIC.

    Notes
    -----
    - MASE (Mean Absolute Scaled Error): Compara erro do modelo com
      erro naive (persistencia). MASE < 1 = melhor que naive.
      Ref: Hyndman & Koehler (2006).
    - sMAPE (Symmetric MAPE): Versao simetrica do MAPE. [0%, 200%].
    - AIC = n * ln(MSE) + 2k: penaliza complexidade. Menor = melhor.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    result = {}
    
    # MASE
    if y_train is not None and len(y_train) > 1:
        naive_errors = np.abs(np.diff(y_train))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1e-10
    elif n > 1:
        naive_errors = np.abs(np.diff(y_true))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1e-10
    else:
        mae_naive = 1e-10
    mae_model = np.mean(np.abs(y_true - y_pred))
    result['MASE'] = mae_model / (mae_naive + 1e-10)
    
    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-10
    result['sMAPE'] = (2.0 / n) * np.sum(np.abs(y_true - y_pred) / denom) * 100
    
    # AIC
    if n_params is not None and n > 0:
        mse = np.mean((y_true - y_pred)**2)
        if mse > 0:
            result['AIC'] = n * np.log(mse) + 2 * n_params
        else:
            result['AIC'] = np.nan
    
    return result


def calcular_baselines(y_train, y_test):
    """
    Calcula previsoes de 3 modelos baseline ingenuas (Rec. 7).

    Parameters
    ----------
    y_train : array-like
        Serie temporal de treino.
    y_test : array-like
        Serie temporal de teste.

    Returns
    -------
    dict
        Dicionario com 'Persistencia', 'MediaMovel3', 'Drift'.

    Notes
    -----
    1. Persistencia (Naive): yhat(t) = y(t-1). Ultimo valor do treino.
    2. Media Movel (k=3): yhat(t) = mean(y(t-1), y(t-2), y(t-3)).
    3. Drift: yhat(h) = y(n) + h * (y(n) - y(1)) / (n-1).
    
    Ref: Hyndman & Athanasopoulos (2021), Chapter 5.2.
    """
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)
    n_train = len(y_train)
    n_test = len(y_test)
    baselines = {}
    
    # 1. Persistencia
    y_pred_persist = np.full(n_test, y_train[-1])
    baselines['Persistencia'] = {
        'y_pred': y_pred_persist,
        **calcular_5_metricas(y_test, y_pred_persist),
        **calcular_metricas_avancadas(y_test, y_pred_persist, y_train)
    }
    
    # 2. Media Movel (k=3)
    k = min(3, n_train)
    y_pred_ma = np.full(n_test, np.mean(y_train[-k:]))
    baselines['MediaMovel3'] = {
        'y_pred': y_pred_ma,
        **calcular_5_metricas(y_test, y_pred_ma),
        **calcular_metricas_avancadas(y_test, y_pred_ma, y_train)
    }
    
    # 3. Drift
    if n_train > 1:
        drift_slope = (y_train[-1] - y_train[0]) / (n_train - 1)
        y_pred_drift = y_train[-1] + drift_slope * np.arange(1, n_test + 1)
    else:
        y_pred_drift = np.full(n_test, y_train[-1])
    baselines['Drift'] = {
        'y_pred': y_pred_drift,
        **calcular_5_metricas(y_test, y_pred_drift),
        **calcular_metricas_avancadas(y_test, y_pred_drift, y_train)
    }
    
    return baselines


def diebold_mariano_test(e1, e2, h=1):
    """
    Teste Diebold-Mariano bilateral para comparacao de modelos.

    Parameters
    ----------
    e1 : array-like
        Erros de previsao do modelo 1 (referencia).
    e2 : array-like
        Erros de previsao do modelo 2 (alternativo).
    h : int, optional
        Horizonte de previsao (default=1).

    Returns
    -------
    tuple (float, float)
        (DM_statistic, p_value). DM > 0 = modelo 2 melhor.

    Notes
    -----
    H0: Ambos os modelos tem a mesma precisao preditiva.
    Funcao de perda quadratica: L(e) = e^2.
    Correcao Newey-West para autocorrelacao ate lag h-1.
    Estatistica ~ t(n-1) sob H0.

    References
    ----------
    Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    e1, e2 = np.array(e1), np.array(e2)
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    
    d_mean = np.mean(d)
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
    """
    Calcula intervalo de confianca via bootstrap nao-parametrico.

    Parameters
    ----------
    y_true : array-like
        Valores reais observados.
    y_pred : array-like
        Valores previstos pelo modelo.
    metrica_func : callable
        Funcao que recebe (y_true, y_pred) e retorna escalar.
    n_boot : int, optional
        Numero de iteracoes bootstrap (default=1000).
    ci : float, optional
        Nivel de confianca (default=0.95).

    Returns
    -------
    dict
        Dicionario com mean, median, ci_lower, ci_upper, std.

    References
    ----------
    Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
    """
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
    """Calcula RMSE entre valores reais e previstos."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae_func(y_true, y_pred):
    """Calcula MAE entre valores reais e previstos."""
    return np.mean(np.abs(y_true - y_pred))

def r2_func(y_true, y_pred):
    """Calcula R2 (coeficiente de determinacao)."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-10
    return 1 - ss_res / ss_tot


def estimar_n_params(modelo_nome):
    """
    Estima o numero de parametros de um modelo para calculo do AIC.

    Parameters
    ----------
    modelo_nome : str
        Nome do modelo (ex: 'RandomForest', 'XGBoost').

    Returns
    -------
    int
        Estimativa do numero de parametros.
    """
    estimativas = {
        'RandomForest': 500,
        'XGBoost': 300,
        'GradientBoosting': 400,
        'SARIMAX': 15,
        'LSTM': 5000,
        'Bayes_PartialPooling': 25,
        'Bayes_CompletePooling': 22,
    }
    return estimativas.get(modelo_nome, 100)


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================
def executar_passo5():
    """
    Avaliacao completa de 28 modelos (7 arquitecturas x 4 datasets).

    Executa:
      1. Avaliacao com 8 metricas (RMSE, MAE, MAPE, R2, RSE, MASE, sMAPE, AIC)
      2. Comparacao com 3 baselines ingenuas (Rec. 7)
      3. Teste Diebold-Mariano com correcao Newey-West
      4. IC 95% via bootstrap (1000 iteracoes)
      5. Ranking composto multi-metrica (Rec. 12)
      6. Ganho percentual de WGI
      7. 8 visualizacoes diagnosticas

    Returns
    -------
    None
        Gera ficheiros CSV e PNG no directorio config.RESULTADOS_DIR.
    """
    print("\n" + "=" * 70)
    print("  PASSO 5: AVALIACAO DE PERFORMANCE - 28 MODELOS + BASELINES")
    print("=" * 70)
    
    os.makedirs(config.RESULTADOS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    ds_paths = {
        'WDI_Limpo': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv'),
        'Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv'),
        'Sintetico_Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv'),
        'WDI_Sintetico': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv'),
    }
    
    modelos_nomes = ['RandomForest', 'XGBoost', 'GradientBoosting', 'SARIMAX',
                     'LSTM', 'Bayes_PartialPooling', 'Bayes_CompletePooling']
    
    resultados = []
    previsoes = {}
    baselines_all = {}
    
    # ============================================================
    # [1/9] CARREGAR E AVALIAR MODELOS
    # ============================================================
    print("\n  [1/9] Carregando modelos e avaliando em dados de teste...")
    
    for ds_nome, ds_path in ds_paths.items():
        if not os.path.exists(ds_path):
            continue
        
        df = pd.read_csv(ds_path).sort_values(['country_code', 'year']).reset_index(drop=True)
        target = config.TARGET_VAR
        if target not in df.columns:
            continue
        
        X = df.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y = df[target].values
        
        n = len(df)
        test_start = int(n * 0.85)
        X_test = X.iloc[test_start:]
        y_test = y[test_start:]
        y_train = y[:test_start]
        
        # --- BASELINES INGENUAS (Rec. 7) ---
        baselines = calcular_baselines(y_train, y_test)
        baselines_all[ds_nome] = baselines
        
        for modelo_nome in modelos_nomes:
            model_path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
            if not os.path.exists(model_path):
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    modelo = pickle.load(f)
                
                y_pred = np.array(modelo.predict(X_test))[:len(y_test)]
                erros = y_test - y_pred
                
                metricas = calcular_5_metricas(y_test, y_pred)
                n_params = estimar_n_params(modelo_nome)
                metricas_avanc = calcular_metricas_avancadas(y_test, y_pred, y_train, n_params)
                metricas.update(metricas_avanc)
                metricas['Dataset'] = ds_nome
                metricas['Modelo'] = modelo_nome
                metricas['N_Test'] = len(y_test)
                resultados.append(metricas)
                
                previsoes[(ds_nome, modelo_nome)] = {
                    'y_true': y_test, 'y_pred': y_pred, 'erros': erros
                }
                
            except Exception as e:
                print(f"    Erro {ds_nome}/{modelo_nome}: {str(e)[:50]}")
    
    # Fallback
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
    # [2/9] BASELINES INGENUAS (Rec. 7)
    # ============================================================
    print("\n  [2/9] Avaliando baselines ingenuas (Persistencia, Media Movel, Drift)...")
    
    baselines_resultados = []
    for ds_nome, baselines in baselines_all.items():
        for bl_nome, bl_data in baselines.items():
            row = {k: v for k, v in bl_data.items() if k != 'y_pred'}
            row['Dataset'] = ds_nome
            row['Modelo'] = f'Baseline_{bl_nome}'
            row['N_Test'] = len(bl_data['y_pred'])
            baselines_resultados.append(row)
    
    df_baselines = pd.DataFrame(baselines_resultados)
    baselines_path = os.path.join(config.RESULTADOS_DIR, 'baselines_ingenuas.csv')
    df_baselines.to_csv(baselines_path, index=False)
    
    ganho_vs_baseline = 0
    if len(df_res) > 0 and len(df_baselines) > 0:
        best_rmse = df_res['RMSE'].min()
        best_baseline_rmse = df_baselines['RMSE'].min()
        ganho_vs_baseline = (best_baseline_rmse - best_rmse) / (best_baseline_rmse + 1e-10) * 100
        print(f"    Melhor modelo RMSE: {best_rmse:.4f}")
        print(f"    Melhor baseline RMSE: {best_baseline_rmse:.4f}")
        print(f"    Ganho vs baseline: {ganho_vs_baseline:.1f}%")
        
        mase_cols = df_res[df_res['MASE'].notna()] if 'MASE' in df_res.columns else pd.DataFrame()
        if len(mase_cols) > 0:
            n_mase_ok = (mase_cols['MASE'] < 1.0).sum()
            print(f"    Modelos com MASE < 1 (melhor que naive): {n_mase_ok}/{len(mase_cols)}")
    
    # ============================================================
    # [3/9] TABELA RESUMO POR MODELO
    # ============================================================
    print("\n  [3/9] Tabela resumo por modelo (media dos 4 datasets)...")
    
    metricas_cols = ['RMSE', 'MAE', 'MAPE', 'R2', 'RSE', 'MASE', 'sMAPE']
    metricas_disponiveis = [c for c in metricas_cols if c in df_res.columns]
    resumo = df_res.groupby('Modelo')[metricas_disponiveis].agg(['mean', 'std', 'min', 'max'])
    resumo.columns = ['_'.join(col) for col in resumo.columns]
    resumo = resumo.sort_values('RMSE_mean')
    resumo_path = os.path.join(config.RESULTADOS_DIR, 'resumo_por_modelo.csv')
    resumo.to_csv(resumo_path)
    
    resumo_ds = df_res.groupby('Dataset')[metricas_disponiveis].agg(['mean', 'std'])
    resumo_ds.columns = ['_'.join(col) for col in resumo_ds.columns]
    resumo_ds_path = os.path.join(config.RESULTADOS_DIR, 'resumo_por_dataset.csv')
    resumo_ds.to_csv(resumo_ds_path)
    
    print(f"    Melhor modelo: {resumo.index[0]} (RMSE={resumo['RMSE_mean'].iloc[0]:.4f})")
    
    # ============================================================
    # [4/9] TESTE DIEBOLD-MARIANO
    # ============================================================
    print("\n  [4/9] Teste Diebold-Mariano (WDI_Limpo vs Agregado)...")
    
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
    # [5/9] INTERVALOS DE CONFIANCA 95% (BOOTSTRAP)
    # ============================================================
    print("\n  [5/9] IC 95% via bootstrap (1000 iteracoes)...")
    
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
    # [6/9] GANHO PERCENTUAL DE WGI
    # ============================================================
    print("\n  [6/9] Ganho percentual de WGI...")
    
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
            mase_wdi = wdi_rows['MASE'].values[0] if 'MASE' in wdi_rows.columns else np.nan
            mase_agg = agg_rows['MASE'].values[0] if 'MASE' in agg_rows.columns else np.nan
            
            ganho_resultados.append({
                'Modelo': modelo_nome,
                'RMSE_WDI': rmse_wdi, 'RMSE_Agregado': rmse_agg,
                'Ganho_RMSE_Pct': (rmse_wdi - rmse_agg) / (rmse_wdi + 1e-10) * 100,
                'R2_WDI': r2_wdi, 'R2_Agregado': r2_agg,
                'Delta_R2': r2_agg - r2_wdi,
                'MAE_WDI': mae_wdi, 'MAE_Agregado': mae_agg,
                'Ganho_MAE_Pct': (mae_wdi - mae_agg) / (mae_wdi + 1e-10) * 100,
                'MASE_WDI': mase_wdi, 'MASE_Agregado': mase_agg,
            })
    
    df_ganho = pd.DataFrame(ganho_resultados)
    ganho_path = os.path.join(config.RESULTADOS_DIR, 'ganho_wgi_detalhado.csv')
    df_ganho.to_csv(ganho_path, index=False)
    
    if len(df_ganho) > 0:
        print(f"    Ganho medio RMSE: {df_ganho['Ganho_RMSE_Pct'].mean():.2f}%")
        print(f"    Ganho medio R2: {df_ganho['Delta_R2'].mean():.4f}")
    
    # ============================================================
    # [7/9] RANKING COMPOSTO MULTI-METRICA (Rec. 12)
    # ============================================================
    print("\n  [7/9] Ranking composto multi-metrica...")
    
    ranking = df_res.groupby('Modelo').agg({
        'RMSE': ['mean', 'std'], 'MAE': 'mean', 'R2': 'mean', 'MAPE': 'mean'
    })
    ranking.columns = ['RMSE_mean', 'RMSE_std', 'MAE_mean', 'R2_mean', 'MAPE_mean']
    
    if 'MASE' in df_res.columns:
        ranking['MASE_mean'] = df_res.groupby('Modelo')['MASE'].mean()
    if 'sMAPE' in df_res.columns:
        ranking['sMAPE_mean'] = df_res.groupby('Modelo')['sMAPE'].mean()
    
    # Score composto normalizado
    rank_cols = ['RMSE_mean', 'MAE_mean', 'MAPE_mean']
    if 'MASE_mean' in ranking.columns:
        rank_cols.append('MASE_mean')
    if 'sMAPE_mean' in ranking.columns:
        rank_cols.append('sMAPE_mean')
    
    ranking_norm = ranking[rank_cols].copy()
    for col in rank_cols:
        col_min = ranking_norm[col].min()
        col_max = ranking_norm[col].max()
        if col_max > col_min:
            ranking_norm[col] = (ranking_norm[col] - col_min) / (col_max - col_min)
        else:
            ranking_norm[col] = 0
    
    ranking['Score_Composto'] = ranking_norm.mean(axis=1)
    if 'R2_mean' in ranking.columns:
        r2_norm = ranking['R2_mean']
        r2_min, r2_max = r2_norm.min(), r2_norm.max()
        if r2_max > r2_min:
            r2_score = 1 - (r2_norm - r2_min) / (r2_max - r2_min)
        else:
            r2_score = 0
        ranking['Score_Composto'] = (ranking['Score_Composto'] + r2_score) / 2
    
    ranking = ranking.sort_values('Score_Composto')
    ranking['Posicao'] = range(1, len(ranking) + 1)
    ranking_path = os.path.join(config.RESULTADOS_DIR, 'ranking_modelos.csv')
    ranking.to_csv(ranking_path)
    
    print("    Ranking composto (menor = melhor):")
    for i, (m, row) in enumerate(ranking.iterrows(), 1):
        mase_str = f", MASE={row['MASE_mean']:.3f}" if 'MASE_mean' in row.index else ""
        print(f"      {i}. {m}: Score={row['Score_Composto']:.4f}, RMSE={row['RMSE_mean']:.4f}{mase_str}")
    
    # ============================================================
    # [8/9] VISUALIZACOES (8 GRAFICOS)
    # ============================================================
    print("\n  [8/9] Gerando 8 visualizacoes...")
    
    # GRAFICO 1: Heatmap RMSE
    pivot_rmse = df_res.pivot_table(values='RMSE', index='Modelo', columns='Dataset', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax, linewidths=0.5)
    ax.set_title('RMSE de Teste: 7 Modelos x 4 Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTADOS_DIR, 'heatmap_rmse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # GRAFICO 2: Ranking com IC 95%
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
        ax.set_title('Ranking de Modelos com IC 95%', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'ranking_ic95.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # GRAFICO 3: Ganho WGI com significancia
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
        ax.bar(range(len(ganho_sorted)), ganho_sorted['Ganho_RMSE_Pct'], color=colors)
        ax.set_xticks(range(len(ganho_sorted)))
        ax.set_xticklabels(ganho_sorted['Modelo'], rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Ganho % RMSE')
        ax.set_title('Ganho de WGI por Modelo\n(escuro=significativo p<0.05)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'ganho_wgi.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # GRAFICO 4: Scatter previsao vs real
    ds_list = ['WDI_Limpo', 'Agregado', 'Sintetico_Agregado', 'WDI_Sintetico']
    if len(previsoes) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        for i, ds_nome in enumerate(ds_list):
            ax = axes.flat[i]
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
    
    # GRAFICO 5: Distribuicao de residuos
    if len(previsoes) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, ds_nome in enumerate(ds_list):
            ax = axes.flat[i]
            ds_previsoes = {k: v for k, v in previsoes.items() if k[0] == ds_nome}
            if ds_previsoes:
                best_key = min(ds_previsoes.keys(), key=lambda k: np.sqrt(np.mean(ds_previsoes[k]['erros']**2)))
                erros = previsoes[best_key]['erros']
                ax.hist(erros, bins=25, alpha=0.7, color='steelblue', edgecolor='white', density=True)
                mu, sigma = np.mean(erros), np.std(erros)
                x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal')
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                ax.set_title(f'{ds_nome}\nmu={mu:.3f}, sigma={sigma:.3f}', fontsize=10)
                ax.set_xlabel('Residuo')
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'{ds_nome}\nSem dados', ha='center', va='center', transform=ax.transAxes)
        plt.suptitle('Distribuicao de Residuos', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'distribuicao_residuos.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # GRAFICO 6: Comparacao 8 metricas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    metricas_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'RSE', 'MASE', 'sMAPE']
    for i, metrica in enumerate(metricas_plot):
        ax = axes.flat[i]
        if metrica in df_res.columns:
            pivot_m = df_res.pivot_table(values=metrica, index='Modelo', columns='Dataset', aggfunc='mean')
            pivot_m.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
            ax.set_title(metrica, fontsize=12, fontweight='bold')
            ax.legend(fontsize=6, loc='best')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
    ax = axes.flat[7]
    if 'AIC' in df_res.columns:
        pivot_aic = df_res.pivot_table(values='AIC', index='Modelo', columns='Dataset', aggfunc='mean')
        pivot_aic.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
        ax.set_title('AIC', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Comparacao de 8 Metricas', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTADOS_DIR, 'comparacao_8metricas.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # GRAFICO 7: Baselines vs Modelos (Rec. 7)
    if len(df_baselines) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        df_all = pd.concat([df_res[['Modelo', 'RMSE']], df_baselines[['Modelo', 'RMSE']]], ignore_index=True)
        rmse_media = df_all.groupby('Modelo')['RMSE'].mean().sort_values()
        colors_bl = ['red' if 'Baseline' in m else 'steelblue' for m in rmse_media.index]
        ax.barh(range(len(rmse_media)), rmse_media.values, color=colors_bl, alpha=0.7)
        ax.set_yticks(range(len(rmse_media)))
        ax.set_yticklabels(rmse_media.index, fontsize=8)
        ax.set_xlabel('RMSE Medio')
        ax.set_title('RMSE: Modelos vs Baselines\n(Vermelho = Baseline)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        ax = axes[1]
        if 'MASE' in df_res.columns:
            mase_media = df_res.groupby('Modelo')['MASE'].mean().sort_values()
            colors_mase = ['green' if v < 1 else 'orange' for v in mase_media.values]
            ax.barh(range(len(mase_media)), mase_media.values, color=colors_mase, alpha=0.7)
            ax.set_yticks(range(len(mase_media)))
            ax.set_yticklabels(mase_media.index, fontsize=8)
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='MASE=1 (Naive)')
            ax.set_xlabel('MASE')
            ax.set_title('MASE: Modelos vs Naive\n(Verde = Melhor que Naive)', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'baselines_vs_modelos.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # GRAFICO 8: Ranking composto
    if len(ranking) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        ranking_sorted = ranking.sort_values('Score_Composto')
        y_pos = np.arange(len(ranking_sorted))
        colors_rank = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranking_sorted)))
        ax.barh(y_pos, ranking_sorted['Score_Composto'], color=colors_rank, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranking_sorted.index)
        ax.set_xlabel('Score Composto (menor = melhor)')
        ax.set_title('Ranking Composto Multi-Metrica\n(RMSE+MAE+MAPE+MASE+sMAPE+R2)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (idx, row) in enumerate(ranking_sorted.iterrows()):
            ax.text(row['Score_Composto'] + 0.01, i, f"#{int(row['Posicao'])}", va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTADOS_DIR, 'ranking_composto.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # ============================================================
    # [9/9] METADADOS E RESUMO
    # ============================================================
    ficheiros_saida = [avaliacao_path, resumo_path, resumo_ds_path, dm_path,
                       ic_path, ganho_path, ranking_path, baselines_path]
    
    gerar_metadados(
        passo='passo5_avaliacao',
        descricao='Avaliacao completa: 8 metricas, baselines ingenuas, DM, IC 95%, ranking composto',
        config=config,
        dados_entrada=[config.MODELOS_DIR],
        dados_saida=ficheiros_saida,
        parametros={'metricas': metricas_cols + ['AIC'], 'bootstrap_n': 1000,
                    'baselines': ['Persistencia', 'MediaMovel3', 'Drift']},
        metricas={
            'modelos_avaliados': len(df_res),
            'melhor_modelo': ranking.index[0] if len(ranking) > 0 else 'N/A',
            'melhor_rmse': float(ranking['RMSE_mean'].iloc[0]) if len(ranking) > 0 else None,
            'dm_significativos_5pct': n_sig,
            'ganho_vs_baseline_pct': float(ganho_vs_baseline),
        }
    )
    auto_save_drive(ficheiros_saida, config)
    
    t_total = time.time() - t_inicio
    print(f"\n  {'='*60}")
    print(f"  RESUMO PASSO 5:")
    print(f"  {'='*60}")
    print(f"  Modelos avaliados: {len(df_res)}")
    print(f"  Baselines avaliadas: {len(df_baselines)}")
    print(f"  Metricas: 8 (RMSE, MAE, MAPE, R2, RSE, MASE, sMAPE, AIC)")
    print(f"  Ficheiros CSV: {len(ficheiros_saida)} | Graficos PNG: 8")
    print(f"  Tempo: {t_total:.1f}s")
    print(f"\n  OK PASSO 5 CONCLUIDO")


if __name__ == '__main__':
    executar_passo5()
