"""
Pipeline de Treinamento de Modelos - SUBPROCESS ISOLATION
==========================================================
Executa o treinamento de 7 modelos para cada dataset num SUBPROCESSO
Python separado. Quando o subprocesso termina, TODA a memória é libertada
automaticamente pelo sistema operativo, eliminando memory leaks.

Fluxo:
  Para cada dataset:
    1. Lança subprocesso: python3 train_single_dataset.py <dataset> <strategy>
    2. Aguarda conclusão (stdout em tempo real)
    3. Memória é 100% libertada quando o subprocesso termina
    4. Consolida resultados parciais
  Após todos os datasets:
    5. Gera resumo final consolidado
    6. Gera visualizações
    7. Gera metadados
"""

import os
import sys
import json
import time
import subprocess

# Garantir compatibilidade com Colab e execução de qualquer diretório
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import pandas as pd
import passo4_model_train_config as config


def get_datasets_strategies():
    """Retorna lista de (dataset_name, strategy_name) a processar."""
    datasets_strategies = []
    for dataset in config.DATASETS:
        if dataset == 'nao_agregado':
            datasets_strategies.append(('nao_agregado', 'A1_Direta'))
        else:
            for strategy in config.STRATEGIES:
                datasets_strategies.append((dataset, strategy))
    return datasets_strategies


def run_single_dataset_subprocess(dataset_name, strategy_name, script_path):
    """
    Executa o treinamento de 1 dataset num subprocesso isolado.
    
    Returns:
        tuple: (success: bool, elapsed: float)
    """
    cmd = [sys.executable, script_path, dataset_name, strategy_name]
    
    start = time.time()
    
    try:
        # Executar subprocesso com output em tempo real
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
            cwd=_SCRIPT_DIR
        )
        
        # Imprimir output em tempo real
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        elapsed = time.time() - start
        
        if process.returncode == 0:
            return True, elapsed
        else:
            print(f"\n  [ERRO] Subprocesso retornou codigo {process.returncode}")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [ERRO] Falha ao executar subprocesso: {e}")
        return False, elapsed


def consolidate_results():
    """
    Consolida todos os ficheiros *_summary.json em resumo_treinamento_completo.csv
    """
    import glob
    
    all_summaries = []
    summary_files = glob.glob(os.path.join(config.OUTPUT_DIR, '*_summary.json'))
    
    for sf in sorted(summary_files):
        try:
            with open(sf, 'r') as f:
                summaries = json.load(f)
            all_summaries.extend(summaries)
        except Exception as e:
            print(f"  AVISO: Erro ao ler {sf}: {e}")
    
    if not all_summaries:
        print("  AVISO: Nenhum resultado encontrado para consolidar.")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(all_summaries)
    
    # Salvar resumo completo
    summary_df.to_csv(
        os.path.join(config.OUTPUT_DIR, 'resumo_treinamento_completo.csv'), 
        index=False
    )
    
    return summary_df


def print_final_tables(summary_df):
    """Imprime as tabelas finais de resumo."""
    if summary_df.empty:
        print("\n  Nenhum resultado para reportar.")
        return
    
    # TABELA 1: RESUMO GERAL
    print(f"\n\n{'=' * 110}")
    print(f"{'TABELA 1: RESUMO GERAL - METRICAS GLOBAIS':^110}")
    print(f"{'=' * 110}")
    header = f"  {'Dataset':<14} {'Estrat.':<14} {'Modelo':<22} {'Val R2':<10} {'Val RMSE':<10} {'Test R2':<10} {'Pais Med.R2':<12} {'Tempo(s)':<10}"
    print(header)
    print(f"  {'-' * 106}")
    for _, row in summary_df.sort_values('Global_R2', ascending=False).iterrows():
        gr2 = f"{row['Global_R2']:.4f}" if pd.notna(row.get('Global_R2')) else "N/A"
        grmse = f"{row['Global_RMSE']:.4f}" if pd.notna(row.get('Global_RMSE')) else "N/A"
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


def run_model_training_pipeline():
    """
    Pipeline principal com SUBPROCESS ISOLATION.
    Cada dataset é treinado num processo Python separado.
    Quando o processo termina, toda a memória é libertada.
    """
    print("=" * 70)
    print("PIPELINE DE TREINAMENTO - SUBPROCESS ISOLATION")
    print("=" * 70)
    print(f"  Modelos: RF, XGBoost, TFT, SARIMAX, LSTM, Bayes_PP, Bayes_CP")
    print(f"  Cada dataset corre num subprocesso isolado (sem memory leak)")
    print(f"  Tempo estimado: ~40-60 min por dataset")
    print("=" * 70)
    
    total_start = time.time()
    
    # Script de treino individual
    train_script = os.path.join(_SCRIPT_DIR, 'train_single_dataset.py')
    if not os.path.exists(train_script):
        print(f"  ERRO: Script nao encontrado: {train_script}")
        return
    
    # Criar directório de output
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Obter lista de datasets
    datasets_strategies = get_datasets_strategies()
    total_datasets = len(datasets_strategies)
    
    print(f"\n  Datasets a processar: {total_datasets}")
    for i, (ds, st) in enumerate(datasets_strategies, 1):
        print(f"    {i:2d}. {ds}_{st}")
    
    # Verificar quais já foram treinados (para retomar)
    already_done = []
    for ds, st in datasets_strategies:
        summary_file = os.path.join(config.OUTPUT_DIR, f"{ds}_{st}_summary.json")
        if os.path.exists(summary_file):
            already_done.append((ds, st))
    
    if already_done:
        print(f"\n  [RETOMAR] {len(already_done)} datasets ja treinados:")
        for ds, st in already_done:
            print(f"    - {ds}_{st} (summary.json existe)")
        print(f"  Restam {total_datasets - len(already_done)} datasets por treinar.")
    
    # Treinar cada dataset num subprocesso
    results = {}
    for i, (dataset_name, strategy_name) in enumerate(datasets_strategies, 1):
        # Verificar se já foi treinado
        summary_file = os.path.join(config.OUTPUT_DIR, f"{dataset_name}_{strategy_name}_summary.json")
        if os.path.exists(summary_file):
            print(f"\n  [{i}/{total_datasets}] {dataset_name}_{strategy_name} - JA TREINADO (skip)")
            results[(dataset_name, strategy_name)] = ('skipped', 0)
            continue
        
        print(f"\n  {'━' * 70}")
        print(f"  [{i}/{total_datasets}] INICIANDO: {dataset_name} x {strategy_name}")
        print(f"  {'━' * 70}")
        
        success, elapsed = run_single_dataset_subprocess(
            dataset_name, strategy_name, train_script
        )
        
        results[(dataset_name, strategy_name)] = ('success' if success else 'failed', elapsed)
        
        # Checkpoint: mostrar progresso
        done_count = sum(1 for v in results.values() if v[0] in ('success', 'skipped'))
        failed_count = sum(1 for v in results.values() if v[0] == 'failed')
        print(f"\n  [PROGRESSO] {done_count}/{total_datasets} concluidos | {failed_count} falhados")
    
    # ================================================================
    # CONSOLIDAÇÃO FINAL
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print(f"{'CONSOLIDACAO DE RESULTADOS':^70}")
    print(f"{'=' * 70}")
    
    summary_df = consolidate_results()
    
    # Imprimir tabelas finais
    print_final_tables(summary_df)
    
    # Visualizações
    print("\n\n[VISUALIZACOES] Gerando graficos de treino...")
    try:
        from passo4_model_train_visualizer import TrainingVisualizer
        visualizer = TrainingVisualizer()
        visualizer.plot_real_training_metrics()
        visualizer.plot_predictions_vs_actual()
        visualizer.plot_best_model_analysis()
        visualizer.plot_predictions_comparison()
        print("  -> Visualizacoes geradas com sucesso!")
    except Exception as e:
        print(f"  AVISO: Nao foi possivel gerar visualizacoes: {e}")
        print(f"  Os modelos foram treinados e salvos com sucesso.")
    
    # Metadados
    print("\n[METADADOS] Gerando metadados do Passo 4...")
    total_time = time.time() - total_start
    try:
        from metadata_generator import generate_metadata_passo4
        all_summaries = summary_df.to_dict('records') if not summary_df.empty else []
        generate_metadata_passo4(
            all_summaries=all_summaries,
            training_times={"total_segundos": round(total_time, 1), "total_minutos": round(total_time/60, 2)},
            output_dir=config.OUTPUT_DIR
        )
        print("  -> Metadados gerados com sucesso!")
    except Exception as e:
        print(f"  AVISO: Nao foi possivel gerar metadados: {e}")
    
    # Resumo final
    print(f"\n\n{'=' * 70}")
    print(f"{'PIPELINE CONCLUIDO':^70}")
    print(f"{'=' * 70}")
    print(f"  Tempo total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Datasets processados:")
    for (ds, st), (status, elapsed) in results.items():
        emoji = "✓" if status == 'success' else "⟳" if status == 'skipped' else "✗"
        print(f"    {emoji} {ds}_{st}: {status} ({elapsed:.0f}s)")
    print(f"\n  Resultados em: {config.OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_model_training_pipeline()
