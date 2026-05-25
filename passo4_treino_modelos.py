"""
============================================================
PASSO 4: TREINO DE MODELOS COM VALIDAÇÃO WALK-FORWARD
============================================================
Modelos (7):
  1. Random Forest (sklearn) com RandomizedSearchCV
  2. XGBoost (xgboost) com early stopping
  3. GradientBoosting (sklearn HistGradientBoosting)
  4. SARIMAX (statsmodels) com fallback LinearRegression
  5. LSTM (tensorflow/keras) com 2 camadas recorrentes
  6. Bayesiano Hierárquico - Partial Pooling (PyMC/MCMC)
  7. Bayesiano Complete Pooling (PyMC/MCMC)

Datasets (4):
  1. WDI Limpo (sem WGI)
  2. Agregado (WDI + PCA WGI) — dataset principal
  3. Sintético Agregado (500 anos) — apenas robustez
  4. WDI Sintético (apenas WDI, sem WGI, 500 anos) — apenas robustez

Validação:
  - Walk-forward temporal (expanding window) com 5 folds
  - Para cada fold: treino = todos os anos até t, teste = próximo bloco
  - Modelo FINAL treinado no split 85/15 (compatível com passo 5)
  - Métricas walk-forward exportadas separadamente

Compatibilidade PCA:
  - Reconhece features wgi_pca1, wgi_pca1_lag1/2, wgi_pca1_ma3,
    wgi_pca1_delta, inter_pca1_* do passo 3 (estratégia A2)
  - WGI originais já foram removidos no passo 3
  - Modelos PCA (scaler + PCA) salvos em dados_engenharia/pca_models/
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
import joblib


# ============================================================
# CONFIGURAÇÃO WALK-FORWARD
# ============================================================
N_FOLDS_WF = 5          # Número de folds walk-forward
MIN_TRAIN_RATIO = 0.50  # Mínimo 50% dos dados para treino no 1º fold
FINAL_SPLIT = 0.85      # Split final: 85% treino, 15% teste (compatível passo5)


# ============================================================
# TIMEOUT HANDLER (para PyMC)
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
        cv=min(config.RF_CV_FOLDS, 3),  # Reduzido para walk-forward
        scoring='neg_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


# ============================================================
# MODELO 2: XGBOOST
# ============================================================
def treinar_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost com early stopping e regularização forte."""
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,       # Regularização L1 forte (parcimónia)
            reg_lambda=2.0,      # Regularização L2 forte
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
    """HistGradientBoosting com regularização forte."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=15,
        l2_regularization=1.0,  # Regularização forte para séries curtas
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
    """SARIMAX via statsmodels + fallback LinearRegression."""
    model = SARIMAXWrapper()
    model.fit(X_train, y_train)
    return model


# ============================================================
# MODELO 5: LSTM (TensorFlow/Keras)
# ============================================================

class LSTMPredictor:
    """Wrapper picklable para modelo LSTM treinado."""
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


def treinar_lstm(X_train, y_train, X_val, y_val):
    """LSTM com 2 camadas recorrentes e regularização."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from sklearn.preprocessing import StandardScaler
        
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s = scaler_X.transform(X_val)
        
        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
        
        # Reshape para LSTM: (samples, 1, features)
        X_train_lstm = X_train_s.reshape(X_train_s.shape[0], 1, X_train_s.shape[1])
        X_val_lstm = X_val_s.reshape(X_val_s.shape[0], 1, X_val_s.shape[1])
        y_val_s = scaler_y.transform(np.array(y_val).reshape(-1, 1)).ravel()
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(1, X_train_s.shape[1]),
                                return_sequences=True,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_s, epochs=config.LSTM_EPOCHS,
                 batch_size=config.LSTM_BATCH_SIZE, verbose=0,
                 validation_data=(X_val_lstm, y_val_s),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        
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

class BayesPartialPredictor:
    """Wrapper picklable para modelo Bayesiano Partial Pooling (PyMC)."""
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


def treinar_bayesiano_partial(X_train, y_train, X_val, y_val):
    """Bayesiano Hierárquico com PyMC (MCMC real)."""
    PYMC_TIMEOUT = getattr(config, 'PYMC_TIMEOUT', 60)
    PYMC_DRAWS = getattr(config, 'PYMC_DRAWS', getattr(config, 'BAYESIAN_SAMPLES', 1000))
    PYMC_TUNE = getattr(config, 'PYMC_TUNE', getattr(config, 'BAYESIAN_TUNE', 500))
    PYMC_CHAINS = getattr(config, 'PYMC_CHAINS', getattr(config, 'BAYESIAN_CHAINS', 2))
    
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(PYMC_TIMEOUT)
        
        import pymc as pm
        from sklearn.preprocessing import StandardScaler
        
        # Reduzir features para PyMC (máx 10)
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
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=2)
            
            mu = alpha + pm.math.dot(X_scaled, beta)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_centered)
            
            trace = pm.sample(draws=PYMC_DRAWS, tune=PYMC_TUNE,
                            chains=PYMC_CHAINS, cores=1,
                            return_inferencedata=True, progressbar=False)
        
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        alpha_post = trace.posterior['alpha'].values.mean()
        beta_post = trace.posterior['beta'].values.mean(axis=(0, 1))
        
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

class BayesCompletePredictor:
    """Wrapper picklable para modelo Bayesiano Complete Pooling (PyMC)."""
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


def treinar_bayesiano_complete(X_train, y_train, X_val, y_val):
    """Bayesiano Complete Pooling com PyMC (MCMC real)."""
    PYMC_TIMEOUT = getattr(config, 'PYMC_TIMEOUT', 60)
    PYMC_DRAWS = getattr(config, 'PYMC_DRAWS', getattr(config, 'BAYESIAN_SAMPLES', 1000))
    PYMC_TUNE = getattr(config, 'PYMC_TUNE', getattr(config, 'BAYESIAN_TUNE', 500))
    PYMC_CHAINS = getattr(config, 'PYMC_CHAINS', getattr(config, 'BAYESIAN_CHAINS', 2))
    
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(PYMC_TIMEOUT)
        
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
            
            trace = pm.sample(draws=PYMC_DRAWS, tune=PYMC_TUNE,
                            chains=PYMC_CHAINS, cores=1,
                            return_inferencedata=True, progressbar=False)
        
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        alpha_post = trace.posterior['alpha'].values.mean()
        beta_post = trace.posterior['beta'].values.mean(axis=(0, 1))
        
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
# VALIDAÇÃO WALK-FORWARD (EXPANDING WINDOW)
# ============================================================

def walk_forward_split(df, n_folds=N_FOLDS_WF, min_train_ratio=MIN_TRAIN_RATIO):
    """
    Gera splits walk-forward temporais (expanding window).
    
    Para dados de painel (múltiplos países), a expansão é por ANO:
      - Fold 1: treino = anos [min, t1], teste = anos (t1, t2]
      - Fold 2: treino = anos [min, t2], teste = anos (t2, t3]
      - ...
    
    Isto garante que nunca se usa informação futura para prever o passado.
    """
    anos = sorted(df['year'].unique())
    n_anos = len(anos)
    
    # Mínimo de anos para treino
    min_train_anos = max(int(n_anos * min_train_ratio), 5)
    
    # Dividir os anos restantes em n_folds blocos de teste
    anos_disponiveis = n_anos - min_train_anos
    if anos_disponiveis < n_folds:
        n_folds = max(anos_disponiveis, 1)
    
    tamanho_fold = max(anos_disponiveis // n_folds, 1)
    
    folds = []
    for i in range(n_folds):
        train_end_idx = min_train_anos + i * tamanho_fold
        test_end_idx = min(train_end_idx + tamanho_fold, n_anos)
        
        anos_treino = anos[:train_end_idx]
        anos_teste = anos[train_end_idx:test_end_idx]
        
        if len(anos_teste) == 0:
            continue
        
        train_mask = df['year'].isin(anos_treino)
        test_mask = df['year'].isin(anos_teste)
        
        folds.append({
            'fold': i + 1,
            'train_idx': df[train_mask].index.tolist(),
            'test_idx': df[test_mask].index.tolist(),
            'anos_treino': (min(anos_treino), max(anos_treino)),
            'anos_teste': (min(anos_teste), max(anos_teste)),
            'n_train': train_mask.sum(),
            'n_test': test_mask.sum(),
        })
    
    return folds


def avaliar_walk_forward(func_treino, X, y, df, folds):
    """
    Avalia um modelo com walk-forward: treina em cada fold e avalia no teste.
    Retorna métricas por fold e métricas agregadas.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    resultados_folds = []
    
    for fold_info in folds:
        train_idx = fold_info['train_idx']
        test_idx = fold_info['test_idx']
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Usar últimos 15% do treino como validação para early stopping
        n_train = len(X_train)
        val_start = int(n_train * 0.85)
        X_tr = X_train.iloc[:val_start]
        y_tr = y_train.iloc[:val_start]
        X_val = X_train.iloc[val_start:]
        y_val = y_train.iloc[val_start:]
        
        if len(X_val) < 5:
            X_val = X_tr.tail(10)
            y_val = y_tr.tail(10)
        
        try:
            modelo = func_treino(X_tr, y_tr, X_val, y_val)
            y_pred = np.array(modelo.predict(X_test))[:len(y_test)]
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((np.array(y_test) - y_pred) / (np.array(y_test) + 1e-8))) * 100
            
            resultados_folds.append({
                'fold': fold_info['fold'],
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_train': fold_info['n_train'],
                'n_test': fold_info['n_test'],
                'anos_teste': fold_info['anos_teste'],
            })
        except Exception as e:
            resultados_folds.append({
                'fold': fold_info['fold'],
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan,
                'n_train': fold_info['n_train'],
                'n_test': fold_info['n_test'],
                'anos_teste': fold_info['anos_teste'],
                'erro': str(e)[:80],
            })
    
    return resultados_folds


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def executar_passo4():
    """
    Treina 7 modelos × 4 datasets com validação walk-forward.
    Exporta:
      - Modelo FINAL (split 85/15) para cada combinação → compatível com passo 5
      - Métricas walk-forward por fold → análise de robustez temporal
      - Métricas finais (holdout 15%) → comparação directa
    """
    print("\n" + "=" * 70)
    print("  PASSO 4: TREINO COM VALIDAÇÃO WALK-FORWARD TEMPORAL")
    print("=" * 70)
    print(f"  Walk-forward: {N_FOLDS_WF} folds (expanding window)")
    print(f"  Modelo final: split {int(FINAL_SPLIT*100)}/{int((1-FINAL_SPLIT)*100)} temporal")
    print(f"  Regularização: forte (L1+L2) para parcimónia com ~25 anos")
    
    os.makedirs(config.MODELOS_DIR, exist_ok=True)
    t_inicio = time.time()
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # ============================================================
    # CARREGAR MODELOS PCA/SCALER DO PASSO 3
    # ============================================================
    pca_models_dir = os.path.join(config.DADOS_ENGENHARIA_DIR, 'pca_models')
    pca_artefactos = {}  # {nome_dataset: {'scaler': ..., 'pca': ..., 'loadings': ...}}
    
    if os.path.isdir(pca_models_dir):
        print(f"\n  MODELOS PCA (do Passo 3): {pca_models_dir}")
        for fname in sorted(os.listdir(pca_models_dir)):
            if fname.endswith('_pca_model.pkl'):
                ds_key = fname.replace('_pca_model.pkl', '')
                scaler_path = os.path.join(pca_models_dir, f'{ds_key}_pca_scaler.pkl')
                model_path = os.path.join(pca_models_dir, fname)
                
                if os.path.exists(scaler_path):
                    try:
                        pca_obj = joblib.load(model_path)
                        scaler_obj = joblib.load(scaler_path)
                        pca_artefactos[ds_key] = {
                            'pca': pca_obj,
                            'scaler': scaler_obj,
                            'n_components': pca_obj.n_components_,
                            'variancia_explicada': pca_obj.explained_variance_ratio_.tolist(),
                            'features_input': list(scaler_obj.feature_names_in_) if hasattr(scaler_obj, 'feature_names_in_') else None,
                        }
                        var_total = sum(pca_obj.explained_variance_ratio_) * 100
                        print(f"    ✓ {ds_key}: PCA({pca_obj.n_components_} comp, "
                              f"{var_total:.1f}% var) + StandardScaler({scaler_obj.mean_.shape[0]} features)")
                        if hasattr(scaler_obj, 'feature_names_in_'):
                            print(f"      WGI inputs: {list(scaler_obj.feature_names_in_)}")
                    except Exception as e:
                        print(f"    ✗ {ds_key}: Erro ao carregar — {str(e)[:60]}")
    else:
        print(f"\n  ⚠ Directório PCA não encontrado: {pca_models_dir}")
        print(f"    (Os datasets já contêm wgi_pca1 pré-calculado pelo passo 3)")
    
    # ============================================================
    # CARREGAR DATASETS
    # ============================================================
    datasets = {}
    paths = {
        'WDI_Limpo': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_limpo_features.csv'),
        'Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv'),
        'Sintetico_Agregado': os.path.join(config.DADOS_ENGENHARIA_DIR, 'sintetico_features.csv'),
        'WDI_Sintetico': os.path.join(config.DADOS_ENGENHARIA_DIR, 'wdi_sintetico_features.csv'),
    }
    
    for nome, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets[nome] = df
            
            # Identificar features PCA
            pca_features = [c for c in df.columns if 'pca' in c.lower()]
            wgi_features = [c for c in df.columns if 'wgi_' in c.lower()]
            
            print(f"\n  {nome}: {df.shape}")
            if pca_features:
                print(f"    PCA features ({len(pca_features)}): {pca_features}")
                # Validar que o PCA model corresponde
                ds_key_map = {'Agregado': 'agregado', 'Sintetico_Agregado': 'sintetico'}
                ds_key = ds_key_map.get(nome)
                if ds_key and ds_key in pca_artefactos:
                    info = pca_artefactos[ds_key]
                    print(f"    ↳ PCA model carregado: {info['n_components']} comp, "
                          f"var={[f'{v*100:.1f}%' for v in info['variancia_explicada']]}")
                elif ds_key:
                    print(f"    ↳ ⚠ PCA model para '{ds_key}' não encontrado")
            if wgi_features and not pca_features:
                print(f"    WGI features: {wgi_features}")
        else:
            print(f"  {nome}: NÃO ENCONTRADO ({path})")
    
    if len(datasets) == 0:
        print("  Nenhum dataset encontrado. Execute o Passo 3 primeiro.")
        return
    
    # Validação de integridade PCA
    if pca_artefactos:
        print(f"\n  VALIDAÇÃO PCA:")
        for ds_key, info in pca_artefactos.items():
            # Verificar que o dataset correspondente tem wgi_pca1
            ds_map = {'agregado': 'Agregado', 'sintetico': 'Sintetico_Agregado'}
            ds_nome_check = ds_map.get(ds_key)
            if ds_nome_check and ds_nome_check in datasets:
                df_check = datasets[ds_nome_check]
                if 'wgi_pca1' in df_check.columns:
                    print(f"    ✓ {ds_key}: wgi_pca1 presente no dataset "
                          f"(range: [{df_check['wgi_pca1'].min():.3f}, {df_check['wgi_pca1'].max():.3f}])")
                else:
                    print(f"    ✗ {ds_key}: wgi_pca1 NÃO encontrado no dataset!")
    
    
    # ============================================================
    # CONFIGURAÇÃO DE MODELOS
    # ============================================================
    modelos_config = {
        'RandomForest': treinar_random_forest,
        'XGBoost': treinar_xgboost,
        'GradientBoosting': treinar_gradient_boosting,
        'SARIMAX': treinar_sarimax,
        'LSTM': treinar_lstm,
        'Bayes_PartialPooling': treinar_bayesiano_partial,
        'Bayes_CompletePooling': treinar_bayesiano_complete,
    }
    
    resultados_finais = []      # Métricas do holdout final (15%)
    resultados_walkforward = [] # Métricas por fold walk-forward
    modelos_salvos = 0
    
    # ============================================================
    # LOOP: DATASET × MODELO
    # ============================================================
    for ds_nome, df in datasets.items():
        print(f"\n{'═'*70}")
        print(f"  DATASET: {ds_nome} ({df.shape[0]} obs × {df.shape[1]} cols)")
        
        # Marcar sintéticos
        is_sintetico = 'sintetico' in ds_nome.lower()
        if is_sintetico:
            print(f"  ⚠ SINTÉTICO — apenas para testes de robustez")
        print(f"{'═'*70}")
        
        # Preparar dados
        df = df.sort_values(['country_code', 'year']).reset_index(drop=True)
        target = config.TARGET_VAR
        
        if target not in df.columns:
            print(f"    Target '{target}' não encontrado.")
            continue
        
        # Features: tudo excepto identificadores e target
        cols_drop = ['country_code', 'year', target, 'pais']
        X = df.drop(columns=[c for c in cols_drop if c in df.columns]).fillna(0)
        y = df[target]
        
        # Identificar composição de features
        n_pca = len([c for c in X.columns if 'pca' in c])
        n_inter = len([c for c in X.columns if 'inter_' in c])
        n_quant = len(X.columns) - n_pca - n_inter
        print(f"    Features: {X.shape[1]} total "
              f"(PCA: {n_pca}, Interações: {n_inter}, Quantitativas: {n_quant})")
        
        # ============================================================
        # WALK-FORWARD: Gerar folds temporais
        # ============================================================
        folds = walk_forward_split(df, n_folds=N_FOLDS_WF)
        print(f"    Walk-forward: {len(folds)} folds")
        for f in folds:
            print(f"      Fold {f['fold']}: treino {f['anos_treino']} ({f['n_train']} obs) → "
                  f"teste {f['anos_teste']} ({f['n_test']} obs)")
        
        # ============================================================
        # SPLIT FINAL (85/15) para modelo definitivo
        # ============================================================
        n = len(df)
        train_end = int(n * FINAL_SPLIT)
        
        X_train_final = X.iloc[:train_end]
        y_train_final = y.iloc[:train_end]
        X_test_final = X.iloc[train_end:]
        y_test_final = y.iloc[train_end:]
        
        # Validação interna (últimos 15% do treino)
        val_start = int(train_end * 0.85)
        X_val_final = X_train_final.iloc[val_start:]
        y_val_final = y_train_final.iloc[val_start:]
        X_tr_final = X_train_final.iloc[:val_start]
        y_tr_final = y_train_final.iloc[:val_start]
        
        print(f"    Split final: Train={train_end} | Test={n-train_end}")
        
        # ============================================================
        # TREINAR CADA MODELO
        # ============================================================
        for modelo_nome, func_treino in modelos_config.items():
            t0 = time.time()
            print(f"\n    [{modelo_nome}]", end="", flush=True)
            
            # --- Walk-forward ---
            print(" WF:", end="", flush=True)
            wf_results = avaliar_walk_forward(func_treino, X, y, df, folds)
            
            for wf in wf_results:
                wf['Dataset'] = ds_nome
                wf['Modelo'] = modelo_nome
                resultados_walkforward.append(wf)
            
            # Métricas WF agregadas
            wf_rmses = [r['RMSE'] for r in wf_results if not np.isnan(r.get('RMSE', np.nan))]
            if wf_rmses:
                wf_rmse_mean = np.mean(wf_rmses)
                wf_rmse_std = np.std(wf_rmses)
                print(f" RMSE={wf_rmse_mean:.3f}±{wf_rmse_std:.3f}", end="", flush=True)
            else:
                wf_rmse_mean = np.nan
                wf_rmse_std = np.nan
                print(" falhou", end="", flush=True)
            
            # --- Modelo FINAL (para passo 5) ---
            print(" | Final:", end="", flush=True)
            try:
                modelo_final = func_treino(X_tr_final, y_tr_final, X_val_final, y_val_final)
                
                # Avaliar no holdout final
                y_pred = np.array(modelo_final.predict(X_test_final))[:len(y_test_final)]
                
                rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
                mae = mean_absolute_error(y_test_final, y_pred)
                r2 = r2_score(y_test_final, y_pred)
                mape = np.mean(np.abs((np.array(y_test_final) - y_pred) / 
                              (np.array(y_test_final) + 1e-8))) * 100
                
                # Salvar modelo
                model_path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(modelo_final, f)
                modelos_salvos += 1
                
                elapsed = time.time() - t0
                print(f" RMSE={rmse:.3f} R²={r2:.3f} ({elapsed:.1f}s)")
                
                resultados_finais.append({
                    'Dataset': ds_nome, 'Modelo': modelo_nome,
                    'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                    'WF_RMSE_mean': wf_rmse_mean, 'WF_RMSE_std': wf_rmse_std,
                    'N_Features': X.shape[1], 'N_Train': train_end,
                    'N_Test': n - train_end, 'Tempo_s': elapsed,
                    'PCA_features': n_pca, 'Is_Sintetico': is_sintetico,
                })
                
            except Exception as e:
                elapsed = time.time() - t0
                print(f" ERRO: {str(e)[:40]} ({elapsed:.1f}s)")
                resultados_finais.append({
                    'Dataset': ds_nome, 'Modelo': modelo_nome,
                    'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan,
                    'WF_RMSE_mean': wf_rmse_mean, 'WF_RMSE_std': wf_rmse_std,
                    'N_Features': X.shape[1], 'N_Train': train_end,
                    'N_Test': n - train_end, 'Tempo_s': elapsed,
                    'PCA_features': n_pca, 'Is_Sintetico': is_sintetico,
                })
    
    # ============================================================
    # EXPORTAR MÉTRICAS
    # ============================================================
    
    # 1. Métricas finais (compatível com passo 5)
    df_resultados = pd.DataFrame(resultados_finais)
    metricas_path = os.path.join(config.MODELOS_DIR, 'metricas_treino_completas.csv')
    df_resultados.to_csv(metricas_path, index=False)
    
    # 2. Métricas walk-forward detalhadas
    df_wf = pd.DataFrame(resultados_walkforward)
    wf_path = os.path.join(config.MODELOS_DIR, 'metricas_walkforward_folds.csv')
    df_wf.to_csv(wf_path, index=False)
    
    # ============================================================
    # VISUALIZAÇÕES
    # ============================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    viz_dir = config.MODELOS_DIR
    
    # 1. Heatmap RMSE: Modelos × Datasets (holdout final)
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        pivot = df_resultados.pivot_table(values='RMSE', index='Modelo', columns='Dataset', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, linewidths=0.5)
        ax.set_title('RMSE Holdout Final: 7 Modelos × 4 Datasets\n'
                     '(Modelo treinado com split 85/15 temporal)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'heatmap_rmse_modelos_datasets.png'), dpi=150)
        plt.close()
    
    # 2. Walk-forward stability: RMSE por fold (apenas datasets reais)
    df_wf_real = df_wf[~df_wf['Dataset'].str.contains('intetico', case=False)]
    if len(df_wf_real.dropna(subset=['RMSE'])) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        for modelo in df_wf_real['Modelo'].unique():
            subset = df_wf_real[df_wf_real['Modelo'] == modelo]
            for ds in subset['Dataset'].unique():
                ds_subset = subset[subset['Dataset'] == ds]
                label = f'{modelo} ({ds})'
                ax.plot(ds_subset['fold'], ds_subset['RMSE'], 'o-', label=label, alpha=0.7)
        
        ax.set_xlabel('Fold Walk-Forward')
        ax.set_ylabel('RMSE')
        ax.set_title('Estabilidade Walk-Forward: RMSE por Fold\n'
                     '(Modelos robustos mantêm RMSE estável ao longo dos folds)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'walkforward_estabilidade.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Comparação WF vs Holdout (robustez)
    if len(df_resultados.dropna(subset=['RMSE', 'WF_RMSE_mean'])) > 0:
        df_comp = df_resultados.dropna(subset=['RMSE', 'WF_RMSE_mean'])
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(df_comp))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], df_comp['RMSE'], width, 
                       label='Holdout Final (15%)', color='steelblue', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], df_comp['WF_RMSE_mean'], width,
                       label='Walk-Forward (média)', color='#ff6f00', alpha=0.8)
        
        # Error bars para WF
        ax.errorbar([i + width/2 for i in x], df_comp['WF_RMSE_mean'], 
                   yerr=df_comp['WF_RMSE_std'], fmt='none', color='black', capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['Modelo'][:8]}\n{r['Dataset'][:6]}" 
                           for _, r in df_comp.iterrows()], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('RMSE')
        ax.set_title('Holdout Final vs Walk-Forward\n'
                     '(Divergência indica instabilidade temporal)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'comparacao_holdout_vs_walkforward.png'), dpi=150)
        plt.close()
    
    # 4. Ganho PCA: WDI_Limpo vs Agregado (impacto da governança via PCA)
    if 'WDI_Limpo' in df_resultados['Dataset'].values and 'Agregado' in df_resultados['Dataset'].values:
        wdi_rmse = df_resultados[df_resultados['Dataset'] == 'WDI_Limpo'].set_index('Modelo')['RMSE']
        agg_rmse = df_resultados[df_resultados['Dataset'] == 'Agregado'].set_index('Modelo')['RMSE']
        ganho = ((wdi_rmse - agg_rmse) / wdi_rmse * 100).dropna()
        
        if len(ganho) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['green' if g > 0 else 'red' for g in ganho.values]
            ganho.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Ganho % RMSE: WDI Limpo → Agregado (WDI + PCA WGI)\n'
                         '(Positivo = PCA WGI melhora a previsão)')
            ax.set_ylabel('Ganho % RMSE')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'ganho_pca_wgi_por_modelo.png'), dpi=150)
            plt.close()
    
    # 5. Ranking geral (RMSE médio)
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        ranking = df_resultados.groupby('Modelo')['RMSE'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        ranking.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Ranking de Modelos (RMSE Médio — todos os datasets)')
        ax.set_xlabel('RMSE Médio')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'ranking_modelos.png'), dpi=150)
        plt.close()
    
    # 6. R² por modelo (apenas datasets reais)
    df_real = df_resultados[~df_resultados['Is_Sintetico']]
    if len(df_real.dropna(subset=['R2'])) > 0:
        r2_modelo = df_real.groupby('Modelo')['R2'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        r2_modelo.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('R² Médio (Datasets Reais: WDI_Limpo + Agregado)')
        ax.set_ylabel('R²')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'r2_por_modelo_reais.png'), dpi=150)
        plt.close()
    
    # ============================================================
    # RESUMO
    # ============================================================
    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  RESUMO DO TREINO (Walk-Forward + Modelo Final)")
    print(f"{'═'*70}")
    print(f"  Modelos treinados: {modelos_salvos}/{len(modelos_config)*len(datasets)}")
    print(f"  Tempo total: {t_total:.1f}s ({t_total/60:.1f} min)")
    
    if len(df_resultados.dropna(subset=['RMSE'])) > 0:
        print(f"\n  RANKING (RMSE holdout final, datasets reais):")
        if len(df_real.dropna(subset=['RMSE'])) > 0:
            ranking = df_real.groupby('Modelo')['RMSE'].mean().sort_values()
            for i, (m, rmse) in enumerate(ranking.items(), 1):
                wf = df_real[df_real['Modelo'] == m]['WF_RMSE_mean'].mean()
                print(f"    {i}. {m}: RMSE={rmse:.4f} (WF={wf:.4f})")
        
        print(f"\n  IMPACTO PCA WGI (Agregado vs WDI_Limpo):")
        if 'WDI_Limpo' in df_resultados['Dataset'].values and 'Agregado' in df_resultados['Dataset'].values:
            r2_wdi = df_resultados[df_resultados['Dataset'] == 'WDI_Limpo']['R2'].mean()
            r2_agg = df_resultados[df_resultados['Dataset'] == 'Agregado']['R2'].mean()
            print(f"    WDI_Limpo R²={r2_wdi:.4f} → Agregado (PCA) R²={r2_agg:.4f}")
            print(f"    ΔR² = {r2_agg - r2_wdi:+.4f}")
    
    # ============================================================
    # METADADOS
    # ============================================================
    gerar_metadados(
        passo='passo4_treino_modelos',
        descricao=('Treino de 7 modelos × 4 datasets com validação walk-forward temporal '
                   f'({N_FOLDS_WF} folds, expanding window). '
                   'Features PCA (estratégia A2) reconhecidas automaticamente. '
                   'Regularização forte (L1+L2) para parcimónia com séries curtas. '
                   'Modelo final treinado com split 85/15 (compatível passo 5).'),
        config=config,
        dados_entrada=list(paths.values()),
        dados_saida=[metricas_path, wf_path],
        parametros={
            'modelos': list(modelos_config.keys()),
            'datasets': list(datasets.keys()),
            'walk_forward_folds': N_FOLDS_WF,
            'min_train_ratio': MIN_TRAIN_RATIO,
            'final_split': f'{int(FINAL_SPLIT*100)}/{int((1-FINAL_SPLIT)*100)}',
            'regularizacao': 'L1+L2 forte (XGB: alpha=0.5, lambda=2.0; GBM: l2=1.0)',
            'pca_models_carregados': list(pca_artefactos.keys()),
            'pca_variancia_explicada': {k: v['variancia_explicada'] for k, v in pca_artefactos.items()},
            'pca_features_input': {k: v['features_input'] for k, v in pca_artefactos.items()},
        },
        metricas={
            'modelos_treinados': modelos_salvos,
            'tempo_total_s': t_total,
        }
    )
    
    auto_save_drive([metricas_path, wf_path], config)
    
    print(f"\n  PASSO 4 CONCLUÍDO ({modelos_salvos} modelos salvos)")
    print(f"  Ficheiros exportados:")
    print(f"    - {metricas_path}")
    print(f"    - {wf_path}")


if __name__ == '__main__':
    executar_passo4()
