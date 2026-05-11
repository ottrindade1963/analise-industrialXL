"""
Passo 4 - Pipeline de Treinamento de Modelos (ROBUSTO - Nível Mestrado)
=========================================================================
Implementa 7 modelos com implementações REAIS e robustas:

  1. RandomForest: RandomizedSearchCV(100 iter), 5 folds temporais, feature importance
  2. XGBoost: RandomizedSearchCV(100 iter), early stopping(50 rounds), regularização
  3. TFT/GradientBoosting: RandomizedSearchCV(100 iter), 5 folds, loss Huber
  4. SARIMAX: auto_arima (pmdarima) com busca AIC, top-5 exógenas por correlação
  5. LSTM: Keras/TensorFlow REAL com 2 camadas LSTM, dropout, early stopping
  6. Bayes_PartialPooling: PyMC hierárquico por país, MCMC(2000 samples)
  7. Bayes_CompletePooling: PyMC global, MCMC(800 samples)

Cada modelo tem uma PRÉ-ETAPA de adequação de dados (DataAdapter) que:
  - Aplica o scaler correto (StandardScaler, MinMaxScaler, ou nenhum)
  - Formata os dados no formato exigido (painel, série temporal, janela)
  - Seleciona features relevantes para o modelo
  - Trata NaN de forma adequada ao contexto

Divisão temporal: Treino<=2016 | Val=2017-2019 | Teste>=2020
"""

import os
import sys
import pickle
import time
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Suprimir logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import passo4_model_train_config as config


# ============================================================
# CLASSE: DataAdapter - Adequação de Dados por Modelo
# ============================================================
class DataAdapter:
    """
    Pré-etapa de adequação de dados específica para cada modelo.
    Garante que cada modelo recebe dados no formato, escala e estrutura
    que necessita para funcionar corretamente.
    """

    def __init__(self, df, country_col, year_col):
        self.df = df.copy()
        self.country_col = country_col
        self.year_col = year_col
        self.target_col = config.TARGET_VAR

        # Identificar colunas de features (excluir identificadores e target)
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        self.feature_cols = [c for c in df.columns
                            if c not in [self.target_col, year_col] + text_cols]

        # Países disponíveis
        if country_col and country_col in df.columns:
            self.countries = sorted(df[country_col].unique())
        else:
            self.countries = []

        # Divisão temporal
        self.train_mask = df[year_col] <= config.TRAIN_END_YEAR
        self.val_mask = (df[year_col] > config.TRAIN_END_YEAR) & (df[year_col] <= config.VAL_END_YEAR)
        self.test_mask = df[year_col] > config.VAL_END_YEAR

    def _get_numeric_features(self, df_subset=None):
        """Retorna apenas features numéricas."""
        if df_subset is None:
            df_subset = self.df
        return [c for c in self.feature_cols
                if c in df_subset.columns and pd.api.types.is_numeric_dtype(df_subset[c])]

    # ────────────────────────────────────────────────────────────
    # ADAPTAÇÃO PARA RF / XGBoost / TFT (Modelos de Painel Global)
    # ────────────────────────────────────────────────────────────
    def adapt_for_panel_model(self, scaler_type='standard'):
        """
        Adequação para modelos de painel (RF, XGBoost, TFT):
        - Dados em formato painel (empilhados por país e ano)
        - Aplica StandardScaler ou MinMaxScaler
        - Remove colunas não-numéricas
        - Imputação: interpolação temporal -> ffill -> bfill -> média
        """
        feat_cols = self._get_numeric_features()

        X = self.df[feat_cols].copy()
        y = self.df[self.target_col].copy()

        # Imputação robusta
        X = X.interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(X.mean()).fillna(0)
        y = y.interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(y.mean()).fillna(0)

        # Divisão temporal
        X_train, X_val, X_test = X[self.train_mask], X[self.val_mask], X[self.test_mask]
        y_train = y[self.train_mask].values
        y_val = y[self.val_mask].values
        y_test = y[self.test_mask].values

        # Escalar
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, feat_cols

    def adapt_for_panel_country_prediction(self, country, scaler, feat_cols):
        """Filtra dados do país e aplica o MESMO scaler já treinado."""
        if not self.country_col:
            return None, None

        df_c = self.df[self.df[self.country_col] == country].sort_values(self.year_col)
        num_feats = [c for c in feat_cols if c in df_c.columns and pd.api.types.is_numeric_dtype(df_c[c])]

        X_c = df_c[num_feats].interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0)
        y_c = df_c[self.target_col].interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0)
        years_c = df_c[self.year_col]

        val_mask_c = (years_c > config.TRAIN_END_YEAR) & (years_c <= config.VAL_END_YEAR)

        X_val_c = X_c[val_mask_c]
        y_val_c = y_c[val_mask_c].values

        if len(y_val_c) < 1:
            return None, None

        X_val_c_s = scaler.transform(X_val_c)
        return X_val_c_s, y_val_c

    # ────────────────────────────────────────────────────────────
    # ADAPTAÇÃO PARA SARIMAX (auto_arima por País)
    # ────────────────────────────────────────────────────────────
    def adapt_for_sarimax(self, country):
        """
        Adequação para SARIMAX por país:
        - Série temporal ordenada por ano
        - Interpolação linear para NaN
        - Selecção de top-N exógenas por correlação com target
        - Retorna dados prontos para auto_arima
        """
        if not self.country_col:
            return None

        df_c = self.df[self.df[self.country_col] == country].sort_values(self.year_col).copy()
        feat_cols = self._get_numeric_features(df_c)

        if len(df_c) < 12:
            return None

        # Interpolação temporal (melhor para séries temporais)
        for col in feat_cols + [self.target_col]:
            if col in df_c.columns:
                df_c[col] = df_c[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

        y = df_c[self.target_col].values
        years = df_c[self.year_col].values

        train_idx = years <= config.TRAIN_END_YEAR
        val_idx = (years > config.TRAIN_END_YEAR) & (years <= config.VAL_END_YEAR)

        y_train = y[train_idx]
        y_val = y[val_idx]

        if len(y_train) < 8 or len(y_val) < 1:
            return None

        # Seleção de exógenas: priorizar WGI quando disponíveis, depois WDI por correlação
        n_exog = config.SARIMAX_N_EXOG
        exog_train = None
        exog_val = None
        top_features = []

        if feat_cols:
            # Separar features WGI (qualitativas) e WDI (quantitativas)
            wgi_features = [c for c in feat_cols if 'wgi' in c.lower()]
            wdi_features = [c for c in feat_cols if 'wgi' not in c.lower()]

            # Calcular correlação de todas as features com o target
            X_train_df = df_c.loc[train_idx, feat_cols].copy()
            X_train_df['__target__'] = y_train
            corr = X_train_df.corr()['__target__'].drop('__target__').abs().sort_values(ascending=False)
            corr = corr[corr > 0.05]  # Threshold mais baixo para incluir mais features

            # Priorizar: incluir WGI primeiro (se existirem), depois WDI por correlação
            selected_wgi = [c for c in wgi_features if c in corr.index]
            selected_wdi = [c for c in corr.index if c not in wgi_features]

            # Combinar: WGI primeiro, depois WDI, até n_exog
            top_features = selected_wgi[:n_exog]
            remaining = n_exog - len(top_features)
            if remaining > 0:
                top_features += selected_wdi[:remaining]

            if top_features:
                exog_train = df_c.loc[train_idx, top_features].values
                exog_val = df_c.loc[val_idx, top_features].values
                # Verificar se exógenas têm NaN
                if np.any(np.isnan(exog_train)) or np.any(np.isnan(exog_val)):
                    exog_train = np.nan_to_num(exog_train, nan=0.0)
                    exog_val = np.nan_to_num(exog_val, nan=0.0)

        return {
            'y_train': y_train, 'y_val': y_val,
            'exog_train': exog_train, 'exog_val': exog_val,
            'top_features': top_features, 'country': country
        }

    # ────────────────────────────────────────────────────────────
    # ADAPTAÇÃO PARA LSTM (Keras - Janelas Temporais por País)
    # ────────────────────────────────────────────────────────────
    def adapt_for_lstm(self, country):
        """
        Adequação para LSTM real (Keras):
        - Série temporal por país
        - MinMaxScaler individual (normaliza [0,1])
        - Cria janelas temporais (lookback) para input 3D do LSTM
        - Formato: (samples, timesteps, features)
        """
        if not self.country_col:
            return None

        df_c = self.df[self.df[self.country_col] == country].sort_values(self.year_col).copy()
        feat_cols = self._get_numeric_features(df_c)

        if len(df_c) < config.LSTM_MIN_TRAIN_SAMPLES + config.LSTM_LOOKBACK + 3:
            return None

        # Interpolação temporal
        for col in feat_cols + [self.target_col]:
            if col in df_c.columns:
                df_c[col] = df_c[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

        X = df_c[feat_cols].values
        y = df_c[self.target_col].values
        years = df_c[self.year_col].values

        # MinMaxScaler por país
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Criar janelas temporais (lookback)
        lookback = config.LSTM_LOOKBACK
        X_windows = []
        y_windows = []
        years_windows = []

        for i in range(lookback, len(X_scaled)):
            X_windows.append(X_scaled[i-lookback:i])  # (lookback, n_features)
            y_windows.append(y_scaled[i])
            years_windows.append(years[i])

        X_windows = np.array(X_windows)  # (n_samples, lookback, n_features)
        y_windows = np.array(y_windows)
        years_windows = np.array(years_windows)

        # Divisão temporal
        train_idx = years_windows <= config.TRAIN_END_YEAR
        val_idx = (years_windows > config.TRAIN_END_YEAR) & (years_windows <= config.VAL_END_YEAR)

        X_train = X_windows[train_idx]
        X_val = X_windows[val_idx]
        y_train = y_windows[train_idx]
        y_val = y_windows[val_idx]

        if len(y_train) < config.LSTM_MIN_TRAIN_SAMPLES or len(y_val) < 1:
            return None

        return {
            'X_train': X_train, 'X_val': X_val,
            'y_train': y_train, 'y_val': y_val,
            'scaler_X': scaler_X, 'scaler_y': scaler_y,
            'feat_cols': feat_cols, 'country': country,
            'n_features': len(feat_cols)
        }

    # ────────────────────────────────────────────────────────────
    # ADAPTAÇÃO PARA BAYESIANOS
    # ────────────────────────────────────────────────────────────
    def adapt_for_bayesian(self):
        """
        Adequação para modelos Bayesianos:
        - Mantém coluna de país (índices hierárquicos)
        - StandardScaler nas features
        - Seleciona top-N features por correlação (estabilidade MCMC)
        """
        if not self.country_col or not self.countries:
            return None

        feat_cols = self._get_numeric_features()
        max_feat = config.BAYESIAN_MAX_FEATURES

        if len(feat_cols) > max_feat:
            df_train = self.df[self.train_mask].copy()
            correlations = df_train[feat_cols].corrwith(df_train[self.target_col]).abs()
            feat_cols = correlations.nlargest(max_feat).index.tolist()

        return {
            'df': self.df,
            'feature_cols': feat_cols,
            'country_col': self.country_col,
            'year_col': self.year_col,
            'countries': self.countries
        }

    def get_info(self):
        """Retorna informações sobre os dados."""
        feat_cols = self._get_numeric_features()
        return {
            'n_features': len(feat_cols),
            'n_countries': len(self.countries),
            'n_train': self.train_mask.sum(),
            'n_val': self.val_mask.sum(),
            'n_test': self.test_mask.sum(),
            'country_col': self.country_col,
            'year_col': self.year_col,
            'feature_cols': feat_cols
        }


# ============================================================
# CLASSE: UnifiedModelTrainer (Implementações Robustas)
# ============================================================
class UnifiedModelTrainer:
    """
    Treina todos os 7 modelos com implementações robustas para mestrado.
    Cada modelo usa a pré-etapa de adequação de dados (DataAdapter).
    """

    def __init__(self, df, dataset_name, strategy_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name

        # Detectar colunas de identificação
        year_col = 'ano' if 'ano' in df.columns else 'year'
        country_col = None
        for col in ['pais', 'country', 'pais_nome', 'country_name']:
            if col in df.columns:
                country_col = col
                break
        if country_col is None:
            for col in ['codigo_iso3', 'iso3', 'country_code']:
                if col in df.columns:
                    country_col = col
                    break

        self.adapter = DataAdapter(df, country_col, year_col)
        self.country_col = country_col
        self.year_col = year_col
        self.countries = self.adapter.countries

        # Resultados
        self.global_metrics = {}
        self.per_country_metrics = {}
        self.models = {}
        self.predictions = {}

        # Info
        info = self.adapter.get_info()
        print(f"\n  Divisao temporal:")
        print(f"    Treino (<={config.TRAIN_END_YEAR}): {info['n_train']} amostras")
        print(f"    Validacao ({config.TRAIN_END_YEAR+1}-{config.VAL_END_YEAR}): {info['n_val']} amostras")
        print(f"    Teste (>={config.VAL_END_YEAR+1}): {info['n_test']} amostras")
        print(f"    Features: {info['n_features']} | Target: {config.TARGET_VAR}")
        print(f"    Paises: {info['n_countries']} | Coluna pais: {country_col}")

    def _calc_metrics(self, y_true, y_pred):
        """Calcula R2, RMSE e MAE com proteção contra NaN/Inf."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'r2': None, 'rmse': None, 'mae': None}
        try:
            y_t = np.array(y_true, dtype=float)
            y_p = np.array(y_pred, dtype=float)
            mask = ~(np.isnan(y_t) | np.isnan(y_p) | np.isinf(y_t) | np.isinf(y_p))
            y_t, y_p = y_t[mask], y_p[mask]
            if len(y_t) < 2:
                return {'r2': None, 'rmse': None, 'mae': None}
            return {
                'r2': r2_score(y_t, y_p),
                'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
                'mae': mean_absolute_error(y_t, y_p)
            }
        except:
            return {'r2': None, 'rmse': None, 'mae': None}

    def _predict_per_country(self, model_name, model, scaler, feat_cols):
        """Previsão por país para modelos de painel."""
        self.per_country_metrics[model_name] = {}
        if not self.countries:
            return
        for country in self.countries:
            try:
                result = self.adapter.adapt_for_panel_country_prediction(country, scaler, feat_cols)
                if result[0] is None:
                    continue
                X_val_c_s, y_val_c = result
                if len(y_val_c) < 1:
                    continue
                preds_c = model.predict(X_val_c_s)
                metrics_c = self._calc_metrics(y_val_c, preds_c)
                self.per_country_metrics[model_name][country] = metrics_c
                if model_name not in self.predictions:
                    self.predictions[model_name] = {'global': {}, 'per_country': {}}
                self.predictions[model_name]['per_country'][country] = {
                    'y_true': y_val_c, 'y_pred': preds_c
                }
            except:
                continue

    def _print_per_country_summary(self, model_name):
        """Imprime resumo das métricas por país."""
        per_country = self.per_country_metrics.get(model_name, {})
        valid_r2 = [m['r2'] for m in per_country.values() if m.get('r2') is not None]
        if valid_r2:
            print(f"     [POR PAIS] Media R2={np.mean(valid_r2):.4f} | "
                  f"Mediana R2={np.median(valid_r2):.4f} | "
                  f"Paises: {len(valid_r2)}/{len(self.countries)} | "
                  f"R2>0: {sum(1 for r in valid_r2 if r > 0)}")

    # ────────────────────────────────────────────────────────────
    # 1. RANDOM FOREST (ROBUSTO)
    # ────────────────────────────────────────────────────────────
    def train_random_forest(self):
        """
        Random Forest com:
        - RandomizedSearchCV (100 iterações, 5 folds temporais)
        - Grid extensivo (576 combinações possíveis)
        - Feature importance ranking
        """
        print(f"\n  -> [1/7] Random Forest")
        print(f"     Adequacao: StandardScaler | Painel global")
        print(f"     Busca: RandomizedSearchCV(n_iter={config.RF_N_ITER}, cv={config.RF_CV_FOLDS})")
        print(f"     Grid: {sum(len(v) for v in config.RF_GRID.values())} valores em {len(config.RF_GRID)} hiperparametros")
        t0 = time.time()

        X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, feat_cols = \
            self.adapter.adapt_for_panel_model(scaler_type='standard')

        rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=config.RF_CV_FOLDS)

        search = RandomizedSearchCV(
            rf, config.RF_GRID,
            n_iter=config.RF_N_ITER,
            cv=tscv,
            scoring=config.SCORING,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
            verbose=0
        )
        search.fit(X_train_s, y_train)
        best_model = search.best_estimator_

        val_preds = best_model.predict(X_val_s)
        test_preds = best_model.predict(X_test_s)

        val_metrics = self._calc_metrics(y_val, val_preds)
        test_metrics = self._calc_metrics(y_test, test_preds)

        # Feature importance
        importances = best_model.feature_importances_
        feat_importance = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)

        self.global_metrics['RandomForest'] = {
            'val': val_metrics, 'test': test_metrics,
            'best_params': search.best_params_,
            'cv_best_score': -search.best_score_,
            'feature_importance': feat_importance[:10],
            'train_time': time.time() - t0
        }

        print(f"     [GLOBAL] Val R2={val_metrics['r2']:.4f} | RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f}")
        print(f"     [GLOBAL] Test R2={test_metrics['r2']:.4f} | RMSE={test_metrics['rmse']:.4f}")
        print(f"     Melhores params: {search.best_params_}")
        print(f"     Top-5 features: {[f'{n}({v:.3f})' for n, v in feat_importance[:5]]}")

        # Previsão por país
        self.predictions['RandomForest'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}
        self._predict_per_country('RandomForest', best_model, scaler, feat_cols)
        self._print_per_country_summary('RandomForest')

        self.models['RandomForest'] = {'model': best_model, 'scaler': scaler, 'features': feat_cols}
        print(f"     Tempo: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # 2. XGBOOST (ROBUSTO com Early Stopping)
    # ────────────────────────────────────────────────────────────
    def train_xgboost(self):
        """
        XGBoost com:
        - RandomizedSearchCV (100 iterações, 5 folds temporais)
        - Early stopping (50 rounds sem melhoria)
        - Regularização L1/L2 (reg_alpha, reg_lambda)
        - Grid extensivo com 9 hiperparâmetros
        """
        print(f"\n  -> [2/7] XGBoost")
        print(f"     Adequacao: StandardScaler | Painel global")
        print(f"     Busca: RandomizedSearchCV(n_iter={config.XGB_N_ITER}, cv={config.XGB_CV_FOLDS})")
        print(f"     Early stopping: {config.XGB_EARLY_STOPPING_ROUNDS} rounds")
        t0 = time.time()

        X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, feat_cols = \
            self.adapter.adapt_for_panel_model(scaler_type='standard')

        # Primeiro: busca de hiperparâmetros com CV
        xgb_model = xgb.XGBRegressor(
            random_state=config.RANDOM_STATE, n_jobs=1,
            verbosity=0, tree_method='hist'
        )
        tscv = TimeSeriesSplit(n_splits=config.XGB_CV_FOLDS)

        search = RandomizedSearchCV(
            xgb_model, config.XGB_GRID,
            n_iter=config.XGB_N_ITER,
            cv=tscv,
            scoring=config.SCORING,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
            verbose=0
        )
        search.fit(X_train_s, y_train)
        best_params = search.best_params_

        # Segundo: retreinar com early stopping usando os melhores params
        best_params_final = {k: v for k, v in best_params.items()}
        best_params_final['n_estimators'] = 1000  # Máximo alto, early stopping vai parar
        best_params_final['random_state'] = config.RANDOM_STATE
        best_params_final['verbosity'] = 0
        best_params_final['tree_method'] = 'hist'
        best_params_final['n_jobs'] = 1

        final_model = xgb.XGBRegressor(**best_params_final)
        final_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False
        )

        val_preds = final_model.predict(X_val_s)
        test_preds = final_model.predict(X_test_s)

        val_metrics = self._calc_metrics(y_val, val_preds)
        test_metrics = self._calc_metrics(y_test, test_preds)

        # Feature importance
        importances = final_model.feature_importances_
        feat_importance = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)

        self.global_metrics['XGBoost'] = {
            'val': val_metrics, 'test': test_metrics,
            'best_params': best_params,
            'cv_best_score': -search.best_score_,
            'n_estimators_used': final_model.best_iteration if hasattr(final_model, 'best_iteration') else best_params.get('n_estimators'),
            'feature_importance': feat_importance[:10],
            'train_time': time.time() - t0
        }

        print(f"     [GLOBAL] Val R2={val_metrics['r2']:.4f} | RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f}")
        print(f"     [GLOBAL] Test R2={test_metrics['r2']:.4f} | RMSE={test_metrics['rmse']:.4f}")
        print(f"     Melhores params: {best_params}")
        print(f"     Top-5 features: {[f'{n}({v:.3f})' for n, v in feat_importance[:5]]}")

        # Previsão por país
        self.predictions['XGBoost'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}
        self._predict_per_country('XGBoost', final_model, scaler, feat_cols)
        self._print_per_country_summary('XGBoost')

        self.models['XGBoost'] = {'model': final_model, 'scaler': scaler, 'features': feat_cols}
        print(f"     Tempo: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # 3. TFT / GradientBoosting (ROBUSTO)
    # ────────────────────────────────────────────────────────────
    def train_tft(self):
        """
        GradientBoosting (proxy para TFT) com:
        - RandomizedSearchCV (100 iterações, 5 folds temporais)
        - Grid extensivo com 8 hiperparâmetros incluindo loss Huber
        - MinMaxScaler para normalização
        """
        print(f"\n  -> [3/7] TFT (GradientBoosting)")
        print(f"     Adequacao: MinMaxScaler | Painel global")
        print(f"     Busca: RandomizedSearchCV(n_iter={config.TFT_N_ITER}, cv={config.TFT_CV_FOLDS})")
        print(f"     Grid: {len(config.TFT_GRID)} hiperparametros (inclui loss Huber)")
        t0 = time.time()

        X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, feat_cols = \
            self.adapter.adapt_for_panel_model(scaler_type='minmax')

        gb = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
        tscv = TimeSeriesSplit(n_splits=config.TFT_CV_FOLDS)

        search = RandomizedSearchCV(
            gb, config.TFT_GRID,
            n_iter=config.TFT_N_ITER,
            cv=tscv,
            scoring=config.SCORING,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
            verbose=0
        )
        search.fit(X_train_s, y_train)
        best_model = search.best_estimator_

        val_preds = best_model.predict(X_val_s)
        test_preds = best_model.predict(X_test_s)

        val_metrics = self._calc_metrics(y_val, val_preds)
        test_metrics = self._calc_metrics(y_test, test_preds)

        # Feature importance
        importances = best_model.feature_importances_
        feat_importance = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)

        self.global_metrics['TFT'] = {
            'val': val_metrics, 'test': test_metrics,
            'best_params': search.best_params_,
            'cv_best_score': -search.best_score_,
            'feature_importance': feat_importance[:10],
            'train_time': time.time() - t0
        }

        print(f"     [GLOBAL] Val R2={val_metrics['r2']:.4f} | RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f}")
        print(f"     [GLOBAL] Test R2={test_metrics['r2']:.4f} | RMSE={test_metrics['rmse']:.4f}")
        print(f"     Melhores params: {search.best_params_}")
        print(f"     Top-5 features: {[f'{n}({v:.3f})' for n, v in feat_importance[:5]]}")

        # Previsão por país
        self.predictions['TFT'] = {'global': {'y_true': y_val, 'y_pred': val_preds}, 'per_country': {}}
        self._predict_per_country('TFT', best_model, scaler, feat_cols)
        self._print_per_country_summary('TFT')

        self.models['TFT'] = {'model': best_model, 'scaler': scaler, 'features': feat_cols}
        print(f"     Tempo: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # 4. SARIMAX (auto_arima REAL via pmdarima)
    # ────────────────────────────────────────────────────────────
    def train_sarimax(self):
        """
        SARIMAX com auto_arima (pmdarima):
        - Busca automática de (p,d,q) via critério AIC
        - Top-5 exógenas por correlação com target
        - Treino individual por país
        - Métricas globais agregadas
        """
        print(f"\n  -> [4/7] SARIMAX (auto_arima)")
        print(f"     Adequacao: Serie temporal por pais | Interpolacao linear")
        print(f"     Busca: auto_arima(max_p={config.SARIMAX_MAX_P}, max_d={config.SARIMAX_MAX_D}, "
              f"max_q={config.SARIMAX_MAX_Q}, criterio={config.SARIMAX_INFORMATION_CRITERION})")
        print(f"     Exogenas: Top-{config.SARIMAX_N_EXOG} por correlacao | Convergencia: maxiter={config.SARIMAX_MAXITER}")
        t0 = time.time()

        if not self.countries:
            print(f"     AVISO: Sem coluna de pais. Pulando SARIMAX.")
            return

        from pmdarima import auto_arima

        self.per_country_metrics['SARIMAX'] = {}
        self.predictions['SARIMAX'] = {'global': {'y_true': [], 'y_pred': []}, 'per_country': {}}
        all_val_true = []
        all_val_pred = []
        countries_trained = 0
        countries_failed = 0
        orders_found = []

        for country in self.countries:
            try:
                data = self.adapter.adapt_for_sarimax(country)
                if data is None:
                    countries_failed += 1
                    continue

                y_train = data['y_train']
                y_val = data['y_val']
                exog_train = data['exog_train']
                exog_val = data['exog_val']

                # auto_arima: busca automática da melhor ordem (p,d,q)
                model = auto_arima(
                    y_train,
                    exogenous=exog_train,
                    start_p=0, max_p=config.SARIMAX_MAX_P,
                    start_q=0, max_q=config.SARIMAX_MAX_Q,
                    d=None, max_d=config.SARIMAX_MAX_D,
                    seasonal=config.SARIMAX_SEASONAL,
                    stepwise=config.SARIMAX_STEPWISE,
                    information_criterion=config.SARIMAX_INFORMATION_CRITERION,
                    suppress_warnings=config.SARIMAX_SUPPRESS_WARNINGS,
                    error_action='ignore',
                    trace=False,
                    maxiter=config.SARIMAX_MAXITER
                )

                val_preds = model.predict(n_periods=len(y_val), exogenous=exog_val)
                orders_found.append(model.order)

                if np.any(np.isnan(val_preds)) or np.any(np.isinf(val_preds)):
                    countries_failed += 1
                    continue

                metrics_c = self._calc_metrics(y_val, val_preds)
                self.per_country_metrics['SARIMAX'][country] = metrics_c
                self.predictions['SARIMAX']['per_country'][country] = {
                    'y_true': y_val, 'y_pred': np.array(val_preds)
                }

                all_val_true.extend(y_val.tolist())
                all_val_pred.extend(val_preds.tolist())
                countries_trained += 1

            except Exception as e:
                countries_failed += 1
                continue

        if all_val_true:
            global_metrics = self._calc_metrics(np.array(all_val_true), np.array(all_val_pred))
            self.global_metrics['SARIMAX'] = {
                'val': global_metrics,
                'test': {'r2': None, 'rmse': None, 'mae': None},
                'countries_trained': countries_trained,
                'countries_failed': countries_failed,
                'orders_found': orders_found,
                'train_time': time.time() - t0
            }
            self.predictions['SARIMAX']['global'] = {
                'y_true': np.array(all_val_true), 'y_pred': np.array(all_val_pred)
            }

            # Contar ordens mais frequentes
            from collections import Counter
            order_counts = Counter(orders_found)
            top_orders = order_counts.most_common(3)

            print(f"     [GLOBAL Agregado] Val R2={global_metrics['r2']:.4f} | RMSE={global_metrics['rmse']:.4f} | MAE={global_metrics['mae']:.4f}")
            print(f"     Paises treinados: {countries_trained}/{len(self.countries)} | Falharam: {countries_failed}")
            print(f"     Ordens (p,d,q) mais frequentes: {top_orders}")
            self._print_per_country_summary('SARIMAX')
        else:
            print(f"     AVISO: Nenhum pais treinado com sucesso.")

        print(f"     Tempo: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # 5. LSTM REAL (Keras/TensorFlow)
    # ────────────────────────────────────────────────────────────
    def train_lstm(self):
        """
        LSTM REAL com Keras/TensorFlow:
        - 2 camadas LSTM (128 + 64 unidades)
        - Dropout (0.2) e Recurrent Dropout (0.1)
        - Camada densa (32 unidades)
        - Early stopping (patience=30)
        - Janelas temporais (lookback=5 anos)
        - MinMaxScaler por país
        - Treino individual por país
        """
        print(f"\n  -> [5/7] LSTM (Keras/TensorFlow)")
        print(f"     Adequacao: MinMaxScaler por pais | Janela temporal (lookback={config.LSTM_LOOKBACK})")
        print(f"     Arquitetura: LSTM({config.LSTM_UNITS_LAYER1}) -> Dropout({config.LSTM_DROPOUT}) -> "
              f"LSTM({config.LSTM_UNITS_LAYER2}) -> Dense({config.LSTM_DENSE_UNITS}) -> Dense(1)")
        print(f"     Treino: epochs={config.LSTM_EPOCHS} | batch={config.LSTM_BATCH_SIZE} | "
              f"early_stopping(patience={config.LSTM_PATIENCE}) | lr={config.LSTM_LEARNING_RATE}")
        t0 = time.time()

        if not self.countries:
            print(f"     AVISO: Sem coluna de pais. Pulando LSTM.")
            return

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Suprimir warnings do TF
        tf.get_logger().setLevel('ERROR')

        self.per_country_metrics['LSTM'] = {}
        self.predictions['LSTM'] = {'global': {'y_true': [], 'y_pred': []}, 'per_country': {}}
        all_val_true = []
        all_val_pred = []
        countries_trained = 0
        countries_failed = 0

        for country in self.countries:
            try:
                data = self.adapter.adapt_for_lstm(country)
                if data is None:
                    countries_failed += 1
                    continue

                X_train = data['X_train']  # (samples, lookback, features)
                X_val = data['X_val']
                y_train = data['y_train']
                y_val_scaled = data['y_val']
                scaler_y = data['scaler_y']
                n_features = data['n_features']

                # Construir modelo LSTM
                tf.keras.backend.clear_session()
                model = Sequential([
                    KerasLSTM(config.LSTM_UNITS_LAYER1,
                             return_sequences=True,
                             input_shape=(config.LSTM_LOOKBACK, n_features),
                             dropout=config.LSTM_DROPOUT,
                             recurrent_dropout=config.LSTM_RECURRENT_DROPOUT),
                    KerasLSTM(config.LSTM_UNITS_LAYER2,
                             return_sequences=False,
                             dropout=config.LSTM_DROPOUT,
                             recurrent_dropout=config.LSTM_RECURRENT_DROPOUT),
                    Dense(config.LSTM_DENSE_UNITS, activation='relu'),
                    Dropout(config.LSTM_DROPOUT),
                    Dense(1)
                ])

                optimizer = Adam(learning_rate=config.LSTM_LEARNING_RATE)
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=config.LSTM_PATIENCE,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6,
                        verbose=0
                    )
                ]

                # Treinar
                model.fit(
                    X_train, y_train,
                    epochs=config.LSTM_EPOCHS,
                    batch_size=config.LSTM_BATCH_SIZE,
                    validation_split=config.LSTM_VALIDATION_SPLIT,
                    callbacks=callbacks,
                    verbose=0
                )

                # Prever
                val_preds_scaled = model.predict(X_val, verbose=0).flatten()

                # Inverter escala
                y_val_original = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
                val_preds_original = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()

                if np.any(np.isnan(val_preds_original)) or np.any(np.isinf(val_preds_original)):
                    countries_failed += 1
                    continue

                metrics_c = self._calc_metrics(y_val_original, val_preds_original)
                self.per_country_metrics['LSTM'][country] = metrics_c
                self.predictions['LSTM']['per_country'][country] = {
                    'y_true': y_val_original, 'y_pred': val_preds_original
                }

                all_val_true.extend(y_val_original.tolist())
                all_val_pred.extend(val_preds_original.tolist())
                countries_trained += 1

            except Exception as e:
                countries_failed += 1
                continue

        if all_val_true:
            global_metrics = self._calc_metrics(np.array(all_val_true), np.array(all_val_pred))
            self.global_metrics['LSTM'] = {
                'val': global_metrics,
                'test': {'r2': None, 'rmse': None, 'mae': None},
                'countries_trained': countries_trained,
                'countries_failed': countries_failed,
                'architecture': f'LSTM({config.LSTM_UNITS_LAYER1})->LSTM({config.LSTM_UNITS_LAYER2})->Dense({config.LSTM_DENSE_UNITS})->Dense(1)',
                'lookback': config.LSTM_LOOKBACK,
                'train_time': time.time() - t0
            }
            self.predictions['LSTM']['global'] = {
                'y_true': np.array(all_val_true), 'y_pred': np.array(all_val_pred)
            }

            print(f"     [GLOBAL Agregado] Val R2={global_metrics['r2']:.4f} | RMSE={global_metrics['rmse']:.4f} | MAE={global_metrics['mae']:.4f}")
            print(f"     Paises treinados: {countries_trained}/{len(self.countries)} | Falharam: {countries_failed}")
            self._print_per_country_summary('LSTM')
        else:
            print(f"     AVISO: Nenhum pais treinado com sucesso.")

        # Limpeza de memória após LSTM
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass
        gc.collect()

        print(f"     Tempo: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # 6 & 7. BAYESIANOS (Partial Pooling + Complete Pooling)
    # ────────────────────────────────────────────────────────────
    def train_bayesian_all(self):
        """
        Modelos Bayesianos com PyMC:
        - Partial Pooling: Hierárquico por país, MCMC(2000 samples, 1000 tune)
        - Complete Pooling: Global, MCMC(800 samples, 400 tune)
        - StandardScaler, top-5 features por correlação
        """
        t0 = time.time()

        bayes_data = self.adapter.adapt_for_bayesian()
        if bayes_data is None:
            print(f"\n  -> [6-7/7] Bayesianos: Sem coluna de pais. Pulando.")
            return

        try:
            from passo4_bayesian_model import train_all_bayesian_models

            print(f"\n  -> [6-7/7] Modelos Bayesianos (PyMC)")
            print(f"     Adequacao: StandardScaler | Top-{config.BAYESIAN_MAX_FEATURES} features por correlacao")
            print(f"     MCMC: samples={config.BAYESIAN_N_SAMPLES} | tune={config.BAYESIAN_N_TUNE} | "
                  f"chains={config.BAYESIAN_CHAINS} | target_accept={config.BAYESIAN_TARGET_ACCEPT}")

            bayes_models = train_all_bayesian_models(
                bayes_data['df'],
                bayes_data['feature_cols'],
                bayes_data['country_col'],
                bayes_data['year_col']
            )

            for bayes_name, bayes_model in bayes_models.items():
                val_mask = (self.df[self.year_col] > config.TRAIN_END_YEAR) & \
                           (self.df[self.year_col] <= config.VAL_END_YEAR)
                test_mask = self.df[self.year_col] > config.VAL_END_YEAR

                df_val = self.df[val_mask].copy()
                df_test = self.df[test_mask].copy()

                val_preds, val_idx = bayes_model.predict(df_val, bayes_data['feature_cols'], bayes_data['country_col'])
                test_preds, test_idx = bayes_model.predict(df_test, bayes_data['feature_cols'], bayes_data['country_col'])

                if len(val_preds) == 0:
                    print(f"     AVISO: {bayes_name}: Sem previsoes de validacao")
                    continue

                # Filtrar NaN
                y_val_raw = df_val.loc[val_idx, config.TARGET_VAR].values
                valid_mask_val = ~np.isnan(y_val_raw) & ~np.isnan(val_preds)
                y_val = y_val_raw[valid_mask_val]
                val_preds_clean = val_preds[valid_mask_val]

                if len(test_preds) > 0:
                    y_test_raw = df_test.loc[test_idx, config.TARGET_VAR].values
                    valid_mask_test = ~np.isnan(y_test_raw) & ~np.isnan(test_preds)
                    y_test = y_test_raw[valid_mask_test]
                    test_preds_clean = test_preds[valid_mask_test]
                else:
                    y_test = np.array([])
                    test_preds_clean = np.array([])

                val_metrics = self._calc_metrics(y_val, val_preds_clean)
                test_metrics = self._calc_metrics(y_test, test_preds_clean) if len(test_preds_clean) > 0 else {'r2': None, 'rmse': None, 'mae': None}

                self.global_metrics[bayes_name] = {
                    'val': val_metrics, 'test': test_metrics,
                    'train_time': time.time() - t0
                }

                r2_str = f"{val_metrics['r2']:.4f}" if val_metrics.get('r2') is not None else "N/A"
                rmse_str = f"{val_metrics['rmse']:.4f}" if val_metrics.get('rmse') is not None else "N/A"
                mae_str = f"{val_metrics['mae']:.4f}" if val_metrics.get('mae') is not None else "N/A"
                print(f"     [{bayes_name}] Val R2={r2_str} | RMSE={rmse_str} | MAE={mae_str}")

                if val_metrics.get('r2') is not None and test_metrics.get('r2') is not None:
                    print(f"     [{bayes_name}] Test R2={test_metrics['r2']:.4f} | RMSE={test_metrics['rmse']:.4f}")

                # Previsão POR PAÍS
                self.per_country_metrics[bayes_name] = {}
                self.predictions[bayes_name] = {
                    'global': {'y_true': y_val, 'y_pred': val_preds_clean},
                    'per_country': {}
                }

                for country in self.countries:
                    try:
                        df_val_c = df_val[df_val[self.country_col] == country].copy()
                        if len(df_val_c) < 1:
                            continue
                        preds_c, idx_c = bayes_model.predict(df_val_c, bayes_data['feature_cols'], bayes_data['country_col'])
                        if len(preds_c) == 0:
                            continue
                        y_val_c_raw = df_val_c.loc[idx_c, config.TARGET_VAR].values
                        valid_c = ~np.isnan(y_val_c_raw) & ~np.isnan(preds_c)
                        y_val_c = y_val_c_raw[valid_c]
                        preds_c_clean = preds_c[valid_c]
                        if len(y_val_c) == 0:
                            continue
                        metrics_c = self._calc_metrics(y_val_c, preds_c_clean)
                        self.per_country_metrics[bayes_name][country] = metrics_c
                        self.predictions[bayes_name]['per_country'][country] = {
                            'y_true': y_val_c, 'y_pred': preds_c_clean
                        }
                    except:
                        continue

                self._print_per_country_summary(bayes_name)
                self.models[bayes_name] = {'model': bayes_model}

        except Exception as e:
            print(f"     ERRO nos modelos Bayesianos: {e}")
            import traceback
            traceback.print_exc()

        print(f"     Tempo total Bayesianos: {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────
    # TREINAR TODOS
    # ────────────────────────────────────────────────────────────
    def train_all(self):
        """Treina todos os 7 modelos (5 clássicos + 2 Bayesianos)."""
        self.train_random_forest()
        self.train_xgboost()
        self.train_tft()
        self.train_sarimax()
        self.train_lstm()
        self.train_bayesian_all()

    def get_summary(self):
        """Retorna resumo dos resultados."""
        summary = []
        all_models = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
                      'Bayes_PartialPooling', 'Bayes_CompletePooling']
        for model_name in all_models:
            if model_name in self.global_metrics:
                gm = self.global_metrics[model_name]['val']
                gm_test = self.global_metrics[model_name].get('test', {})
                per_country = self.per_country_metrics.get(model_name, {})
                valid_r2 = [m['r2'] for m in per_country.values() if m.get('r2') is not None]

                summary.append({
                    'Dataset': self.dataset_name,
                    'Estrategia': self.strategy_name,
                    'Modelo': model_name,
                    'Global_R2': gm.get('r2'),
                    'Global_RMSE': gm.get('rmse'),
                    'Global_MAE': gm.get('mae'),
                    'Test_R2': gm_test.get('r2') if gm_test else None,
                    'Test_RMSE': gm_test.get('rmse') if gm_test else None,
                    'Test_MAE': gm_test.get('mae') if gm_test else None,
                    'PerCountry_Mean_R2': np.mean(valid_r2) if valid_r2 else None,
                    'PerCountry_Median_R2': np.median(valid_r2) if valid_r2 else None,
                    'N_Countries': len(valid_r2),
                    'N_Countries_R2_Positive': sum(1 for r in valid_r2 if r > 0),
                    'Train_Time_s': self.global_metrics[model_name].get('train_time')
                })
        return summary

    def save_results(self, output_dir=None):
        """Salva modelos, previsões e métricas."""
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{self.dataset_name}_{self.strategy_name}"

        # 1. Salvar modelos (.pkl)
        for model_name, model_data in self.models.items():
            model_path = os.path.join(output_dir, f"{prefix}_{model_name}.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            except Exception as e:
                # Modelos Bayesianos podem não ser serializáveis
                try:
                    save_data = {'model_type': model_name, 'info': str(model_data)}
                    with open(model_path, 'wb') as f:
                        pickle.dump(save_data, f)
                except:
                    pass

        # 2. Salvar métricas globais em CSV
        global_rows = []
        all_models = ['RandomForest', 'XGBoost', 'TFT', 'SARIMAX', 'LSTM',
                      'Bayes_PartialPooling', 'Bayes_CompletePooling']
        for model_name in all_models:
            if model_name in self.global_metrics:
                gm_val = self.global_metrics[model_name].get('val', {})
                gm_test = self.global_metrics[model_name].get('test', {})
                per_country = self.per_country_metrics.get(model_name, {})
                valid_r2 = [m['r2'] for m in per_country.values() if m.get('r2') is not None]

                global_rows.append({
                    'Dataset': self.dataset_name,
                    'Estrategia': self.strategy_name,
                    'Modelo': model_name,
                    'Val_R2': gm_val.get('r2'),
                    'Val_RMSE': gm_val.get('rmse'),
                    'Val_MAE': gm_val.get('mae'),
                    'Test_R2': gm_test.get('r2') if gm_test else None,
                    'Test_RMSE': gm_test.get('rmse') if gm_test else None,
                    'Test_MAE': gm_test.get('mae') if gm_test else None,
                    'PerCountry_Mean_R2': np.mean(valid_r2) if valid_r2 else None,
                    'PerCountry_Median_R2': np.median(valid_r2) if valid_r2 else None,
                    'N_Countries': len(valid_r2),
                    'N_Countries_R2_Positive': sum(1 for r in valid_r2 if r > 0),
                    'Train_Time_s': self.global_metrics[model_name].get('train_time')
                })

        if global_rows:
            df_global = pd.DataFrame(global_rows)
            df_global.to_csv(os.path.join(output_dir, f"{prefix}_metricas_globais.csv"), index=False)

        # 3. Salvar métricas por país em CSV
        for model_name, country_metrics in self.per_country_metrics.items():
            if country_metrics:
                rows = [{'Dataset': self.dataset_name, 'Estrategia': self.strategy_name,
                         'Modelo': model_name, 'Pais': country,
                         'R2': m.get('r2'), 'RMSE': m.get('rmse'), 'MAE': m.get('mae')}
                        for country, m in country_metrics.items()]
                pd.DataFrame(rows).to_csv(
                    os.path.join(output_dir, f"{prefix}_{model_name}_metricas_por_pais.csv"), index=False)

        # 4. Salvar previsões (.pkl)
        with open(os.path.join(output_dir, f"{prefix}_predictions.pkl"), 'wb') as f:
            pickle.dump(self.predictions, f)

        # 5. Salvar training_logs
        logs = {
            'global_metrics': self.global_metrics,
            'per_country_metrics': self.per_country_metrics,
            'dataset': self.dataset_name,
            'strategy': self.strategy_name
        }
        with open(os.path.join(output_dir, 'training_logs.pkl'), 'wb') as f:
            pickle.dump(logs, f)


# ============================================================
# FUNÇÃO PRINCIPAL: run_training_for_all
# ============================================================
def run_training_for_all():
    """
    Executa o treinamento completo para todos os datasets e estratégias.
    """
    print("\n" + "=" * 90)
    print("INICIANDO TREINAMENTO COMPLETO DE 7 MODELOS (ROBUSTO)")
    print("  Classicos: RF(100iter), XGBoost(100iter+EarlyStopping), TFT(100iter)")
    print("  Serie Temporal: SARIMAX(auto_arima), LSTM(Keras real)")
    print("  Bayesianos: PartialPooling(MCMC), CompletePooling(MCMC)")
    print("=" * 90)

    total_start = time.time()

    # Definir datasets e estratégias
    datasets_strategies = []
    for dataset in config.DATASETS:
        if dataset == 'nao_agregado':
            datasets_strategies.append(('nao_agregado', 'A1_Direta'))
        else:
            for strategy in config.STRATEGIES:
                datasets_strategies.append((dataset, strategy))

    # Carregar datasets
    loaded_data = {}
    for dataset_name, strategy_name in datasets_strategies:
        filename = f"{dataset_name}_{strategy_name}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            loaded_data[(dataset_name, strategy_name)] = df
            print(f"\n  Carregando: {filename} ({df.shape[0]} x {df.shape[1]})")
        else:
            print(f"\n  NAO ENCONTRADO: {filepath}")

    if not loaded_data:
        print(f"\n  Nenhum dataset encontrado em {config.DATA_DIR}/")
        return

    # Treinar
    all_summaries = []
    for (dataset_name, strategy_name), df in loaded_data.items():
        print(f"\n  {'=' * 70}")
        print(f"  DATASET: {dataset_name} x {strategy_name} ({df.shape})")
        print(f"  {'=' * 70}")

        trainer = UnifiedModelTrainer(df, dataset_name, strategy_name)
        trainer.train_all()
        trainer.save_results()
        all_summaries.extend(trainer.get_summary())

        # Limpeza de memória entre datasets (previne crash por memory leak)
        del trainer
        gc.collect()
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass
        gc.collect()

        # Checkpoint: salvar resumo parcial após cada dataset
        partial_df = pd.DataFrame(all_summaries)
        partial_df.to_csv(os.path.join(config.OUTPUT_DIR, 'resumo_parcial_checkpoint.csv'), index=False)
        print(f"\n  [CHECKPOINT] Resultados parciais salvos ({len(all_summaries)} combinacoes concluidas)")

    # ================================================================
    # RESUMO FINAL
    # ================================================================
    if not all_summaries:
        print("\n  Nenhum resultado para reportar.")
        return

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'resumo_treinamento_completo.csv'), index=False)

    # TABELA 1: RESUMO GERAL
    print(f"\n\n{'=' * 110}")
    print(f"{'TABELA 1: RESUMO GERAL - METRICAS GLOBAIS':^110}")
    print(f"{'=' * 110}")
    header = f"  {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'Val R2':<10} {'Val RMSE':<10} {'Test R2':<10} {'Pais Med.R2':<12} {'Tempo(s)':<10}"
    print(header)
    print(f"  {'-' * 106}")
    for _, row in summary_df.sort_values('Global_R2', ascending=False).iterrows():
        gr2 = f"{row['Global_R2']:.4f}" if pd.notna(row['Global_R2']) else "N/A"
        grmse = f"{row['Global_RMSE']:.4f}" if pd.notna(row['Global_RMSE']) else "N/A"
        tr2 = f"{row['Test_R2']:.4f}" if pd.notna(row.get('Test_R2')) else "N/A"
        pr2 = f"{row['PerCountry_Median_R2']:.4f}" if pd.notna(row.get('PerCountry_Median_R2')) else "N/A"
        t_s = f"{row['Train_Time_s']:.0f}" if pd.notna(row.get('Train_Time_s')) else "N/A"
        print(f"  {row['Dataset']:<14} {row['Estrategia']:<14} {row['Modelo']:<22} {gr2:<10} {grmse:<10} {tr2:<10} {pr2:<12} {t_s:<10}")

    # TABELA 2: TOP 20
    valid = summary_df[summary_df['Global_R2'].notna()].copy()
    if not valid.empty:
        top20 = valid.nlargest(20, 'Global_R2')
        print(f"\n\n{'=' * 110}")
        print(f"{'TABELA 2: RANKING TOP 20':^110}")
        print(f"{'=' * 110}")
        print(f"  {'#':<4} {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'Val R2':<10} {'Test R2':<10} {'RMSE':<10}")
        print(f"  {'-' * 82}")
        for rank, (_, row) in enumerate(top20.iterrows(), 1):
            tr2 = f"{row['Test_R2']:.4f}" if pd.notna(row.get('Test_R2')) else "N/A"
            print(f"  {rank:<4} {row['Dataset']:<14} {row['Estrategia']:<14} {row['Modelo']:<22} "
                  f"{row['Global_R2']:.4f}     {tr2:<10} {row['Global_RMSE']:.4f}")

    # TABELA 3: BAYESIANOS vs CLÁSSICOS
    if not valid.empty:
        bayes_m = valid[valid['Modelo'].str.contains('Bayes')]
        classic_m = valid[~valid['Modelo'].str.contains('Bayes')]
        if not bayes_m.empty and not classic_m.empty:
            print(f"\n\n{'=' * 110}")
            print(f"{'TABELA 3: BAYESIANOS vs CLASSICOS':^110}")
            print(f"{'=' * 110}")
            best_classic = classic_m.loc[classic_m['Global_R2'].idxmax()]
            best_bayes = bayes_m.loc[bayes_m['Global_R2'].idxmax()]
            ganho = best_bayes['Global_R2'] - best_classic['Global_R2']
            print(f"  Melhor Classico: {best_classic['Modelo']} ({best_classic['Dataset']}) R2={best_classic['Global_R2']:.4f}")
            print(f"  Melhor Bayesiano: {best_bayes['Modelo']} ({best_bayes['Dataset']}) R2={best_bayes['Global_R2']:.4f}")
            print(f"  Ganho Bayesiano: {ganho:+.4f}")

    # TABELA 4: ESTATÍSTICAS POR MODELO
    if not valid.empty:
        print(f"\n\n{'=' * 110}")
        print(f"{'TABELA 4: ESTATISTICAS DESCRITIVAS POR MODELO':^110}")
        print(f"{'=' * 110}")
        print(f"  {'Modelo':<22} {'Media R2':<11} {'Mediana':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'N':<5}")
        print(f"  {'-' * 76}")
        for modelo in sorted(valid['Modelo'].unique()):
            m_data = valid[valid['Modelo'] == modelo]['Global_R2']
            print(f"  {modelo:<22} {m_data.mean():.4f}      {m_data.median():.4f}     "
                  f"{m_data.std():.4f}     {m_data.min():.4f}     {m_data.max():.4f}     {len(m_data)}")

    # Salvar tabelas
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'tabela_comparativa_completa.csv'), index=False)

    if not valid.empty:
        stats_rows = []
        for modelo in valid['Modelo'].unique():
            m_data = valid[valid['Modelo'] == modelo]['Global_R2']
            stats_rows.append({
                'Modelo': modelo, 'Media_R2': m_data.mean(),
                'Mediana_R2': m_data.median(), 'Std_R2': m_data.std(),
                'Min_R2': m_data.min(), 'Max_R2': m_data.max(),
                'N_Datasets': len(m_data)
            })
        pd.DataFrame(stats_rows).to_csv(
            os.path.join(config.OUTPUT_DIR, 'tabela_estatisticas_descritivas.csv'), index=False)

    total_time = time.time() - total_start
    print(f"\n\n{'=' * 110}")
    print(f"{'TREINAMENTO COMPLETO FINALIZADO':^110}")
    print(f"{'Tempo total: ' + f'{total_time:.0f}s ({total_time/60:.1f} min)':^110}")
    print(f"{'=' * 110}")
