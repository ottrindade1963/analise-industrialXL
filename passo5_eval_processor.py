import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import passo5_eval_config as config


def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """
    Teste de Diebold-Mariano para comparar a precisão preditiva de dois modelos.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    if var_d == 0:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


class ModelEvaluator:
    def __init__(self):
        self.results = []
        self.predictions = {}

    def load_data_and_model(self, model_name, dataset_name, strategy_name):
        """
        Carrega os dados de teste e o modelo treinado.
        Agora compatível com o novo formato de modelos salvos (dict com metadados).
        """
        # Carregar dados
        data_filename = f"{dataset_name}_{strategy_name}.csv"
        data_filepath = os.path.join(config.DATA_DIR, data_filename)
        
        if not os.path.exists(data_filepath):
            return None, None, None
            
        df = pd.read_csv(data_filepath)
        
        # Carregar modelo (novo formato: dict com metadados)
        # Formato do Passo 4: {dataset}_{strategy}_{model}.pkl
        model_filename = f"{dataset_name}_{strategy_name}_{model_name}.pkl"
        model_filepath = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return None, None, None
            
        with open(model_filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extrair modelo e metadados
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_names = model_data.get('features', model_data.get('feature_names', []))
            scaler = model_data.get('scaler', None)
            year_col = model_data.get('year_col', None)
            sarimax_features = model_data.get('sarimax_features', None)
        else:
            # Compatibilidade com formato antigo (modelo direto)
            model = model_data
            feature_names = None
            scaler = None
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
            sarimax_features = None
        
        # Detectar coluna de ano e país
        if year_col is None:
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
        
        # Padronizar coluna de ano
        if 'year' in df.columns and 'ano' not in df.columns:
            df.rename(columns={'year': 'ano'}, inplace=True)
            year_col = 'ano'
        
        # Importar config do passo 4 para ter os anos de corte
        import passo4_model_train_config as train_config
        
        # Separar target
        if config.TARGET_VAR not in df.columns:
            return None, None, None
        
        y = df[config.TARGET_VAR]
        
        # Máscara para o conjunto de teste (após VAL_END_YEAR)
        if year_col and year_col in df.columns:
            test_mask = df[year_col] > train_config.VAL_END_YEAR
        else:
            # Fallback: últimos 15%
            n = len(df)
            test_start = int(n * 0.85)
            test_mask = pd.Series([False]*test_start + [True]*(n-test_start), index=df.index)
        
        y_test = y.loc[test_mask].astype(float)
        
        # Remover NaNs do target
        valid_y_mask = ~y_test.isna()
        
        # Preparar features
        if feature_names:
            # Usar as mesmas features que o modelo foi treinado
            X = df[feature_names].copy()
        else:
            # Fallback: remover colunas de identificação
            id_cols = ['pais', 'country', 'country_code', 'codigo_iso3', 'fonte_dados']
            cols_to_drop = [c for c in id_cols if c in df.columns] + [config.TARGET_VAR]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            X = X.select_dtypes(include=[np.number])
        
        X_test = X.loc[test_mask].copy()
        X_test = X_test[valid_y_mask]
        y_test = y_test[valid_y_mask]
        
        # Preencher NaNs
        X_test = X_test.fillna(X_test.median())
        X_test = X_test.fillna(0)
        
        # Escalar features (usando o mesmo scaler do treino)
        if scaler is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = X_test
        
        return {
            'model': model,
            'model_name': model_name,
            'sarimax_features': sarimax_features,
            'feature_names': feature_names
        }, X_test_scaled, y_test

    def evaluate_model(self, model_info, X_test, y_test, model_name, dataset_name, strategy_name):
        """Avalia o modelo e calcula métricas expandidas."""
        try:
            model = model_info['model']
            sarimax_features = model_info.get('sarimax_features', None)
            
            if len(X_test) == 0 or len(y_test) == 0:
                print(f"  -> Sem dados de teste para {model_name} ({dataset_name}-{strategy_name})")
                return False
            
            if model_name == 'SARIMAX':
                # SARIMAX: usar as features exógenas corretas
                if hasattr(model, '_exog_features') and model._exog_features:
                    exog_cols = model._exog_features
                elif sarimax_features:
                    exog_cols = sarimax_features
                else:
                    # Fallback: usar primeiras N features
                    n_exog = getattr(model, '_n_exog', 5)
                    exog_cols = list(X_test.columns[:n_exog])
                
                # Garantir que as colunas existem
                exog_cols = [c for c in exog_cols if c in X_test.columns]
                
                if len(exog_cols) == 0:
                    print(f"  -> Sem features exógenas para SARIMAX ({dataset_name}-{strategy_name})")
                    return False
                
                try:
                    preds = model.forecast(steps=len(X_test), exog=X_test[exog_cols].values)
                except Exception as e:
                    # Se forecast falhar, tentar predict
                    try:
                        preds = model.predict(X_test)
                    except:
                        print(f"  -> Erro SARIMAX forecast ({dataset_name}-{strategy_name}): {e}")
                        return False
            else:
                preds = model.predict(X_test)
            
            # Converter para array numpy
            preds = np.array(preds).flatten()
            y_true = np.array(y_test).flatten()
            
            # Métricas
            mse = mean_squared_error(y_true, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, preds)
            mape = mean_absolute_percentage_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            
            # Guardar previsões para testes comparativos
            key = f"{model_name}_{dataset_name}_{strategy_name}"
            self.predictions[key] = {
                'y_true': y_true,
                'y_pred': preds
            }
            
            self.results.append({
                'Modelo': model_name,
                'Dataset': dataset_name,
                'Estrategia': strategy_name,
                'R2': r2,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape,
                'N_Test': len(y_true)
            })
            
            print(f"  -> Avaliado: {model_name} ({dataset_name}-{strategy_name}) | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
            return True
        except Exception as e:
            print(f"  -> Erro ao avaliar {model_name} ({dataset_name}-{strategy_name}): {e}")
            return False

    def _load_metrics_from_csv(self, dataset_name, strategy_name):
        """Fallback: carrega métricas diretamente dos CSVs gerados pelo Passo 4."""
        csv_path = os.path.join(config.MODEL_DIR, f"{dataset_name}_{strategy_name}_metricas_globais.csv")
        if os.path.exists(csv_path):
            df_metrics = pd.read_csv(csv_path)
            for _, row in df_metrics.iterrows():
                self.results.append({
                    'Modelo': row.get('Modelo', ''),
                    'Dataset': dataset_name,
                    'Estrategia': strategy_name,
                    'R2': row.get('Val_R2') if pd.notna(row.get('Val_R2')) else row.get('Test_R2'),
                    'RMSE': row.get('Val_RMSE') if pd.notna(row.get('Val_RMSE')) else row.get('Test_RMSE'),
                    'MSE': (row.get('Val_RMSE', 0) ** 2) if pd.notna(row.get('Val_RMSE')) else None,
                    'MAE': row.get('Val_MAE') if pd.notna(row.get('Val_MAE')) else row.get('Test_MAE'),
                    'MAPE': None,
                    'N_Test': row.get('N_Countries', 0)
                })
            return True
        return False

    def run_evaluation(self):
        """Executa a avaliação para todas as combinações."""
        # Debug: mostrar caminhos configurados
        print(f"  [DEBUG] BASE_DIR: {config.BASE_DIR}")
        print(f"  [DEBUG] DATA_DIR: {config.DATA_DIR} (existe: {os.path.exists(config.DATA_DIR)})")
        print(f"  [DEBUG] MODEL_DIR: {config.MODEL_DIR} (existe: {os.path.exists(config.MODEL_DIR)})")
        if os.path.exists(config.MODEL_DIR):
            pkl_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.pkl')]
            csv_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.csv')]
            print(f"  [DEBUG] PKL encontrados: {len(pkl_files)}")
            print(f"  [DEBUG] CSV encontrados: {len(csv_files)}")
        if os.path.exists(config.DATA_DIR):
            eng_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.csv')]
            print(f"  [DEBUG] Datasets engenharia: {eng_files}")

        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                # Skip: nao_agregado só tem A1_Direta
                if dataset == 'nao_agregado' and strategy != 'A1_Direta':
                    continue

                print(f"\nAvaliando: {dataset} - {strategy}")
                
                # Tentar carregar métricas do CSV do Passo 4 como fallback
                csv_loaded = False
                models_with_pkl = ['RandomForest', 'XGBoost', 'TFT']  # Modelos que têm PKL carregável
                
                for model_name in config.MODELS:
                    if model_name in models_with_pkl:
                        # Tentar carregar modelo e avaliar
                        model_info, X_test, y_test = self.load_data_and_model(model_name, dataset, strategy)
                        if model_info is not None and X_test is not None and y_test is not None:
                            self.evaluate_model(model_info, X_test, y_test, model_name, dataset, strategy)
                        else:
                            print(f"  -> PKL não encontrado para {model_name}, tentando CSV...")
                    else:
                        # SARIMAX, LSTM, Bayesianos: usar métricas do CSV do Passo 4
                        print(f"  -> {model_name}: modelo por país/MCMC, métricas via CSV do Passo 4")
                
                # Carregar métricas dos modelos sem PKL carregável a partir do CSV do Passo 4
                if not csv_loaded:
                    csv_path = os.path.join(config.MODEL_DIR, f"{dataset}_{strategy}_metricas_globais.csv")
                    if os.path.exists(csv_path):
                        df_csv = pd.read_csv(csv_path)
                        for _, row in df_csv.iterrows():
                            model_name = row.get('Modelo', '')
                            if model_name not in models_with_pkl:
                                self.results.append({
                                    'Modelo': model_name,
                                    'Dataset': dataset,
                                    'Estrategia': strategy,
                                    'R2': row.get('Val_R2') if pd.notna(row.get('Val_R2')) else row.get('Test_R2'),
                                    'RMSE': row.get('Val_RMSE') if pd.notna(row.get('Val_RMSE')) else row.get('Test_RMSE'),
                                    'MSE': (row.get('Val_RMSE', 0) ** 2) if pd.notna(row.get('Val_RMSE')) else None,
                                    'MAE': row.get('Val_MAE') if pd.notna(row.get('Val_MAE')) else row.get('Test_MAE'),
                                    'MAPE': None,
                                    'N_Test': row.get('N_Countries', 0)
                                })
                                print(f"  -> {model_name}: métricas carregadas do CSV (R²={row.get('Val_R2', 'N/A')})")
                    else:
                        print(f"  -> CSV de métricas não encontrado: {csv_path}")
                        
        # Salvar resultados
        if self.results:
            results_df = pd.DataFrame(self.results)
            out_filepath = os.path.join(config.OUTPUT_DIR, 'metricas_avaliacao_expandidas.csv')
            results_df.to_csv(out_filepath, index=False)
            print(f"\nResultados salvos em: {out_filepath}")
            
            # Executar testes de Diebold-Mariano
            self.run_dm_tests(results_df)
            
            return results_df
        else:
            print("\nNenhum resultado gerado. Verifique se os modelos foram treinados.")
            return pd.DataFrame()

    def run_dm_tests(self, results_df):
        """Executa testes de Diebold-Mariano comparando modelos."""
        print("\nExecutando Testes de Diebold-Mariano (H3)...")
        dm_results = []
        
        # Para cada dataset/estratégia, comparar RF vs SARIMAX e XGBoost vs SARIMAX
        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                key_rf = f"RandomForest_{dataset}_{strategy}"
                key_xgb = f"XGBoost_{dataset}_{strategy}"
                key_sarimax = f"SARIMAX_{dataset}_{strategy}"
                
                # RF vs SARIMAX
                if key_rf in self.predictions and key_sarimax in self.predictions:
                    y_true = self.predictions[key_rf]['y_true']
                    y_pred_rf = self.predictions[key_rf]['y_pred']
                    y_pred_sarimax = self.predictions[key_sarimax]['y_pred']
                    
                    min_len = min(len(y_true), len(y_pred_rf), len(y_pred_sarimax))
                    if min_len > 5:
                        dm_stat, p_val = diebold_mariano_test(
                            y_true[:min_len], y_pred_rf[:min_len], y_pred_sarimax[:min_len]
                        )
                        dm_results.append({
                            'Modelo_1': 'RandomForest', 'Modelo_2': 'SARIMAX',
                            'Dataset': dataset, 'Estrategia': strategy,
                            'DM_Stat': dm_stat, 'P_Value': p_val,
                            'Significativo_5%': p_val < 0.05
                        })
                
                # XGBoost vs SARIMAX
                if key_xgb in self.predictions and key_sarimax in self.predictions:
                    y_true = self.predictions[key_xgb]['y_true']
                    y_pred_xgb = self.predictions[key_xgb]['y_pred']
                    y_pred_sarimax = self.predictions[key_sarimax]['y_pred']
                    
                    min_len = min(len(y_true), len(y_pred_xgb), len(y_pred_sarimax))
                    if min_len > 5:
                        dm_stat, p_val = diebold_mariano_test(
                            y_true[:min_len], y_pred_xgb[:min_len], y_pred_sarimax[:min_len]
                        )
                        dm_results.append({
                            'Modelo_1': 'XGBoost', 'Modelo_2': 'SARIMAX',
                            'Dataset': dataset, 'Estrategia': strategy,
                            'DM_Stat': dm_stat, 'P_Value': p_val,
                            'Significativo_5%': p_val < 0.05
                        })
        
        if dm_results:
            dm_df = pd.DataFrame(dm_results)
            out_filepath = os.path.join(config.OUTPUT_DIR, 'testes_diebold_mariano.csv')
            dm_df.to_csv(out_filepath, index=False)
            print(f"Testes DM salvos em: {out_filepath}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
