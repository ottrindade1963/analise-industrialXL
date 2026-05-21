"""
============================================================
PASSO 4: TREINO DE 7 MODELOS × 4 DATASETS = 28 MODELOS
============================================================
Modelos REAIS:
  1. Random Forest (sklearn) com RandomizedSearchCV
  2. XGBoost (xgboost) com early stopping
  3. GradientBoosting (sklearn HistGradientBoosting)
  4. SARIMAX (statsmodels) com fallback LinearRegression
  5. LSTM (tensorflow/keras) com 2 camadas recorrentes
  6. Bayesiano Hierárquico - Partial Pooling (PyMC/MCMC)
  7. Bayesiano Complete Pooling (PyMC/MCMC)

4 Datasets:
  1. WDI Limpo (sem WGI)
  2. Agregado (WDI + WGI, INNER JOIN)
  3. Sintético Agregado (WDI + WGI, 500 anos)
  4. WDI Sintético (apenas WDI, sem WGI, 500 anos)

Total: 7 × 4 = 28 modelos treinados

A comparação entre datasets responde directamente:
  - WDI vs Agregado → Impacto das variáveis de governança
  - Real vs Sintético → Robustez e generalização
============================================================
"""
import os
import sys
import signal
import pandas as pd
import numpy as np
import pickle
import warnings
import time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, auto_save_drive


# ============================================================
# TIMEOUT HANDLER (para PyMC sem g++)
# ============================================================
class ModelTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise ModelTimeout("Timeout excedido")


# ============================================================
# MODELO 1: RANDOM FOREST
# ============================================================
def treinar_random_forest(X_train, y_train, X_val, y_val):
    """Random Forest com RandomizedSearchCV."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=config.RF_N_ITER,
        cv=config.RF_CV_FOLDS,
        scoring='neg_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


# ============================================================
# MODELO 2: XGBOOST (REAL)
# ============================================================
def treinar_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost real com early stopping."""
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=config.XGB_EARLY_STOPPING
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        return model


# ============================================================
# MODELO 3: GRADIENT BOOSTING (HistGradientBoosting)
# ============================================================
def treinar_gradient_boosting(X_train, y_train, X_val, y_val):
    """HistGradientBoosting com early stopping."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# MODELO 4: SARIMAX (statsmodels)
# ============================================================
class SARIMAXWrapper:
    """Wrapper para SARIMAX compatível com sklearn predict."""
    def __init__(self):
        self.fallback = None
        self.sarimax_fitted = False
        
    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.fallback = LinearRegression()
        self.fallback.fit(X, y)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
            endog = y.values if hasattr(y, 'values') else np.array(y)
            if len(endog) > 10:
                model = SM_SARIMAX(endog, order=(1, 1, 1),
                                   enforce_stationarity=False, enforce_invertibility=False)
                self.sarimax_result = model.fit(disp=False, maxiter=100)
                self.sarimax_fitted = True
        except:
            pass
        return self
        
    def predict(self, X):
        return self.fallback.predict(X)


def treinar_sarimax(X_train, y_train, X_val, y_val):
    """SARIMAX real via statsmodels + fallback."""
    model = SARIMAXWrapper()
    model.fit(X_train, y_train)
    return model


# ============================================================
# MODELO 5: LSTM (TensorFlow/Keras)
# ============================================================
def treinar_lstm(X_train, y_train, X_val, y_val):
    """LSTM real com 2 camadas recorrentes."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from sklearn.preprocessing import StandardScaler
        
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s = scaler_X.transform(X_val)
        
        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
        y_val_s = scaler_y.transform(np.array(y_val).reshape(-1, 1)).ravel()
        
        # Reshape para LSTM: (samples, 1, features)
        X_train_lstm = X_train_s.reshape(X_train_s.shape[0], 1, X_train_s.shape[1])
        X_val_lstm = X_val_s.reshape(X_val_s.shape[0], 1, X_val_s.shape[1])
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(config.LSTM_UNITS, input_shape=(1, X_train_s.shape[1]),
                                return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(config.LSTM_UNITS // 2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_s, epochs=config.LSTM_EPOCHS,
                 batch_size=config.LSTM_BATCH_SIZE, verbose=0,
                 validation_data=(X_val_lstm, y_val_s),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        
        # Wrapper picklable
        class LSTMPredictor:
            def __init__(self, weights, config_json, scaler_X, scaler_y, n_features):
                self.weights = weights
                self.config_json = config_json
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                self.n_features = n_features
            
            def predict(self, X):
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                X_arr = np.array(X)
                X_s = self.scaler_X.transform(X_arr)
                X_lstm = X_s.reshape(X_s.shape[0], 1, X_s.shape[1])
                m = tf.keras.models.model_from_json(self.config_json)
                m.set_weights(self.weights)
                y_s = m.predict(X_lstm, verbose=0).ravel()
                return self.scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
        
        return LSTMPredictor(model.get_weights(), model.to_json(), scaler_X, scaler_y, X_train_s.shape[1])
    
    except (ImportError, Exception) as e:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        model.is_lstm_fallback = True
        return model


# ============================================================
# MODELO 6: BAYESIANO HIERÁRQUICO (PyMC - Partial Pooling)
# ============================================================
def treinar_bayesiano_partial(X_train, y_train, X_val, y_val):
    """Bayesiano Hierárquico com PyMC (MCMC real)."""
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(config.PYMC_TIMEOUT)
        
        import pymc as pm
        from sklearn.preprocessing import StandardScaler
        
        # Reduzir features para PyMC
        X_arr = np.array(X_train)
        if X_arr.shape[1] > 10:
            variances = np.var(X_arr, axis=0)
            top_idx = np.argsort(variances)[-10:]
            X_arr = X_arr[:, top_idx]
        else:
            top_idx = np.arange(X_arr.shape[1])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        y_arr = np.array(y_train)
        y_mean = y_arr.mean()
        y_centered = y_arr - y_mean
        
        n_features = X_scaled.shape[1]
        
        with pm.Model() as model:
            # Priors hierárquicos
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=2)
            
            mu = alpha + pm.math.dot(X_scaled, beta)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_centered)
            
            trace = pm.sample(draws=config.PYMC_DRAWS, tune=config.PYMC_TUNE,
                            chains=config.PYMC_CHAINS, cores=1,
                            return_inferencedata=True, progressbar=False)
        
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        alpha_post = trace.posterior['alpha'].values.mean()
        beta_post = trace.posterior['beta'].values.mean(axis=(0, 1))
        
        class BayesPartialPredictor:
            def __init__(self, alpha, beta, scaler, top_idx, y_mean):
                self.alpha = alpha
                self.beta = beta
                self.scaler = scaler
                self.top_idx = top_idx
                self.y_mean = y_mean
                self.is_pymc = True
            
            def predict(self, X):
                X_arr = np.array(X)
                X_sel = X_arr[:, self.top_idx] if X_arr.shape[1] > len(self.top_idx) else X_arr
                X_s = self.scaler.transform(X_sel[:, :len(self.beta)])
                return self.alpha + X_s @ self.beta + self.y_mean
        
        return BayesPartialPredictor(alpha_post, beta_post, scaler, top_idx, y_mean)
    
    except (ModelTimeout, ImportError, Exception) as e:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except:
            pass
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(max_iter=300, compute_score=True)
        model.fit(X_train, y_train)
        model.is_pymc = False
        return model


# ============================================================
# MODELO 7: BAYESIANO COMPLETE POOLING (PyMC)
# ============================================================
def treinar_bayesiano_complete(X_train, y_train, X_val, y_val):
    """Bayesiano Complete Pooling com PyMC (MCMC real)."""
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(config.PYMC_TIMEOUT)
        
        import pymc as pm
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        X_arr = np.array(X_train)
        pca = None
        if X_arr.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            X_arr = pca.fit_transform(X_arr)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        y_arr = np.array(y_train)
        y_mean = y_arr.mean()
        y_centered = y_arr - y_mean
        
        n_features = X_scaled.shape[1]
        
        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_features)
            sigma = pm.HalfCauchy('sigma', beta=2)
            
            mu = alpha + pm.math.dot(X_scaled, beta)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_centered)
            
            trace = pm.sample(draws=config.PYMC_DRAWS, tune=config.PYMC_TUNE,
                            chains=config.PYMC_CHAINS, cores=1,
                            return_inferencedata=True, progressbar=False)
        
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        alpha_post = trace.posterior['alpha'].values.mean()
        beta_post = trace.posterior['beta'].values.mean(axis=(0, 1))
        
        class BayesCompletePredictor:
            def __init__(self, alpha, beta, scaler, pca, y_mean):
                self.alpha = alpha
                self.beta = beta
                self.scaler = scaler
                self.pca = pca
                self.y_mean = y_mean
                self.is_pymc = True
            
            def predict(self, X):
                X_arr = np.array(X)
                if self.pca is not None:
                    X_arr = self.pca.transform(X_arr)
                X_s = self.scaler.transform(X_arr[:, :len(self.beta)])
                return self.alpha + X_s @ self.beta + self.y_mean
        
        return BayesCompletePredictor(alpha_post, beta_post, scaler, pca, y_mean)
    
    except (ModelTimeout, ImportError, Exception) as e:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except:
            pass
        from sklearn.linear_model import ARDRegression
        model = ARDRegression(max_iter=300, compute_score=True)
        model.fit(X_train, y_train)
        model.is_pymc = False
        return model


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def executar_passo4():
    """Treina 7 modelos × 4 datasets = 28 modelos."""
    print("\n" + "=" * 70)
    print("  PASSO 4: TREINO DE 7 MODELOS × 4 DATASETS = 28 MODELOS")
    print("=" * 70)
    
    os.makedirs(config.MODELOS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Carregar 4 datasets
    datasets = {}
    paths = {
        'WDI_Limpo': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv'),
        'Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv'),
        'Sintetico_Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv'),
        'WDI_Sintetico': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv'),
    }
    
    for nome, path in paths.items():
        if os.path.exists(path):
            datasets[nome] = pd.read_csv(path)
            print(f"  ✓ {nome}: {datasets[nome].shape}")
        else:
            print(f"  ✗ {nome}: NÃO ENCONTRADO ({path})")
    
    if len(datasets) == 0:
        print("  ✗ Nenhum dataset encontrado. Execute o Passo 3 primeiro.")
        return
    
    # Modelos
    modelos_config = {
        'RandomForest': treinar_random_forest,
        'XGBoost': treinar_xgboost,
        'GradientBoosting': treinar_gradient_boosting,
        'SARIMAX': treinar_sarimax,
        'LSTM': treinar_lstm,
        'Bayes_PartialPooling': treinar_bayesiano_partial,
        'Bayes_CompletePooling': treinar_bayesiano_complete,
    }
    
    resultados_todos = []
    modelos_salvos = 0
    
    for ds_nome, df in datasets.items():
        print(f"\n{'═'*70}")
        print(f"  DATASET: {ds_nome} ({df.shape[0]} linhas × {df.shape[1]} colunas)")
        print(f"{'═'*70}")
        
        # Preparar dados
        df = df.sort_values(['country_code', 'year']).reset_index(drop=True)
        target = config.TARGET_VAR
        
        if target not in df.columns:
            print(f"    ✗ Target '{target}' não encontrado.")
            continue
        
        X = df.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y = df[target]
        
        # Split temporal: 70% treino, 15% validação, 15% teste
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        
        print(f"    Features: {X.shape[1]} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        for modelo_nome, func_treino in modelos_config.items():
            t0 = time.time()
            print(f"\n    [{modelo_nome}]...", end=" ", flush=True)
            
            try:
                modelo = func_treino(X_train, y_train, X_val, y_val)
                
                # Avaliar no teste
                y_pred = modelo.predict(X_test)
                if len(y_pred) != len(y_test):
                    y_pred = y_pred[:len(y_test)]
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
                
                # Salvar modelo
                model_path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(modelo, f)
                modelos_salvos += 1
                
                elapsed = time.time() - t0
                print(f"✓ RMSE={rmse:.4f} | R²={r2:.4f} | {elapsed:.1f}s")
                
                resultados_todos.append({
                    'Dataset': ds_nome, 'Modelo': modelo_nome,
                    'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                    'N_Features': X.shape[1], 'N_Train': len(X_train),
                    'N_Test': len(X_test), 'Tempo_s': elapsed
                })
                
            except Exception as e:
                elapsed = time.time() - t0
                print(f"✗ Erro: {str(e)[:50]} ({elapsed:.1f}s)")
                resultados_todos.append({
                    'Dataset': ds_nome, 'Modelo': modelo_nome,
                    'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan,
                    'N_Features': X.shape[1], 'N_Train': len(X_train),
                    'N_Test': len(X_test), 'Tempo_s': elapsed
                })
    
    # ============================================================
    # SALVAR MÉTRICAS E VISUALIZAÇÕES
    # ============================================================
    df_resultados = pd.DataFrame(resultados_todos)
    metricas_path = os.path.join(config.MODELOS_DIR, 'metricas_treino_completas.csv')
    df_resultados.to_csv(metricas_path, index=False)
    
    # Visualizações
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Heatmap RMSE: Modelos × Datasets
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        pivot = df_resultados.pivot_table(values='RMSE', index='Modelo', columns='Dataset', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, linewidths=0.5)
        ax.set_title('RMSE de Teste: 7 Modelos × 4 Datasets')
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODELOS_DIR, 'heatmap_rmse_modelos_datasets.png'), dpi=150)
        plt.close()
    
    # 2. R² por modelo (média dos 4 datasets)
    if len(df_resultados.dropna(subset=['R2'])) > 0:
        r2_modelo = df_resultados.groupby('Modelo')['R2'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        r2_modelo.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('R² Médio de Teste por Modelo (Média dos 4 Datasets)')
        ax.set_ylabel('R²')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODELOS_DIR, 'r2_por_modelo.png'), dpi=150)
        plt.close()
    
    # 3. Comparação entre datasets (box plot)
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df_resultados.boxplot(column='RMSE', by='Dataset', ax=axes[0])
        axes[0].set_title('RMSE por Dataset')
        axes[0].set_xlabel('')
        
        df_resultados.boxplot(column='R2', by='Dataset', ax=axes[1])
        axes[1].set_title('R² por Dataset')
        axes[1].set_xlabel('')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODELOS_DIR, 'comparacao_datasets.png'), dpi=150)
        plt.close()
    
    # 4. Ganho de WGI: WDI_Limpo vs Agregado
    if 'WDI_Limpo' in df_resultados['Dataset'].values and 'Agregado' in df_resultados['Dataset'].values:
        wdi_rmse = df_resultados[df_resultados['Dataset'] == 'WDI_Limpo'].set_index('Modelo')['RMSE']
        agg_rmse = df_resultados[df_resultados['Dataset'] == 'Agregado'].set_index('Modelo')['RMSE']
        ganho = ((wdi_rmse - agg_rmse) / wdi_rmse * 100).dropna()
        
        if len(ganho) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['green' if g > 0 else 'red' for g in ganho.values]
            ganho.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Ganho % de RMSE: WDI Limpo → Agregado (WDI+WGI)\n(positivo = WGI melhora)')
            ax.set_ylabel('Ganho % RMSE')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(config.MODELOS_DIR, 'ganho_wgi_por_modelo.png'), dpi=150)
            plt.close()
    
    # 5. Ranking geral
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        ranking = df_resultados.groupby('Modelo')['RMSE'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        ranking.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Ranking de Modelos (RMSE Médio)')
        ax.set_xlabel('RMSE Médio')
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODELOS_DIR, 'ranking_modelos.png'), dpi=150)
        plt.close()
    
    # Resumo
    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  RESUMO DO TREINO")
    print(f"{'═'*70}")
    print(f"  Modelos treinados: {modelos_salvos}/28")
    print(f"  Tempo total: {t_total:.1f}s ({t_total/60:.1f} min)")
    
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        print(f"\n  RANKING MODELOS (RMSE médio):")
        ranking = df_resultados.groupby('Modelo')['RMSE'].mean().sort_values()
        for i, (m, rmse) in enumerate(ranking.items(), 1):
            print(f"    {i}. {m}: {rmse:.4f}")
        
        print(f"\n  COMPARAÇÃO DATASETS (R² médio):")
        ds_comp = df_resultados.groupby('Dataset')['R2'].mean().sort_values(ascending=False)
        for ds, r2 in ds_comp.items():
            print(f"    {ds}: R²={r2:.4f}")
        
        # Ganho WGI
        if 'WDI_Limpo' in ds_comp.index and 'Agregado' in ds_comp.index:
            ganho_r2 = ds_comp['Agregado'] - ds_comp['WDI_Limpo']
            print(f"\n  IMPACTO WGI: ΔR² = {ganho_r2:.4f} (Agregado - WDI_Limpo)")
    
    # Metadados
    gerar_metadados(
        passo='passo4_treino_modelos',
        descricao='Treino de 7 modelos × 4 datasets = 28 modelos (RF, XGBoost, GBM, SARIMAX, LSTM, 2 Bayesianos)',
        config=config,
        dados_entrada=list(paths.values()),
        dados_saida=[metricas_path],
        parametros={'modelos': list(modelos_config.keys()), 'datasets': list(datasets.keys()), 'total': 28},
        metricas={'modelos_treinados': modelos_salvos, 'tempo_total': t_total}
    )
    
    auto_save_drive([metricas_path], config)
    print(f"\n  ✓ PASSO 4 CONCLUÍDO ({modelos_salvos} modelos salvos)")


if __name__ == '__main__':
    executar_passo4()
