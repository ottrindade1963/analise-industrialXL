"""
Train Single Dataset - Subprocess Isolation
=============================================
Este script treina TODOS os 7 modelos para UM ÚNICO dataset.
É chamado pelo pipeline principal como subprocesso isolado,
garantindo que toda a memória é libertada quando o processo termina.

Uso:
    python3 train_single_dataset.py <dataset_name> <strategy_name>

Exemplo:
    python3 train_single_dataset.py inner A1_Direta
    python3 train_single_dataset.py nao_agregado A1_Direta

Output:
    - Ficheiros .pkl dos modelos treinados em modelos_treinados/
    - CSVs de métricas (globais e por país) em modelos_treinados/
    - Ficheiro JSON com resumo do dataset em modelos_treinados/
    - Exit code 0 = sucesso, 1 = erro
"""

import os
import sys
import json
import time
import traceback

# Garantir que o script funciona independentemente do diretório de execução
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

# Suprimir logs do TensorFlow ANTES de qualquer import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import passo4_model_train_config as config
from passo4_model_train_processor import UnifiedModelTrainer


def train_dataset(dataset_name, strategy_name):
    """
    Treina todos os 7 modelos para um único dataset.
    
    Returns:
        list: Lista de dicionários com resumo de métricas por modelo
    """
    # Carregar dataset
    filename = f"{dataset_name}_{strategy_name}.csv"
    filepath = os.path.join(config.DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"  ERRO: Ficheiro nao encontrado: {filepath}")
        return []
    
    df = pd.read_csv(filepath)
    print(f"\n  {'=' * 70}")
    print(f"  DATASET: {dataset_name} x {strategy_name} ({df.shape})")
    print(f"  {'=' * 70}")
    
    # Treinar todos os modelos
    trainer = UnifiedModelTrainer(df, dataset_name, strategy_name)
    trainer.train_all()
    trainer.save_results()
    
    # Obter resumo
    summaries = trainer.get_summary()
    
    # Salvar resumo individual deste dataset como JSON
    summary_file = os.path.join(
        config.OUTPUT_DIR, 
        f"{dataset_name}_{strategy_name}_summary.json"
    )
    
    # Converter tipos numpy para tipos Python nativos
    clean_summaries = []
    for s in summaries:
        clean = {}
        for k, v in s.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.ndarray,)):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        clean_summaries.append(clean)
    
    with open(summary_file, 'w') as f:
        json.dump(clean_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n  [OK] Resumo salvo em: {summary_file}")
    
    return summaries


def main():
    """Entry point do script."""
    if len(sys.argv) < 3:
        print("Uso: python3 train_single_dataset.py <dataset_name> <strategy_name>")
        print("Exemplo: python3 train_single_dataset.py inner A1_Direta")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    strategy_name = sys.argv[2]
    
    print(f"\n{'=' * 70}")
    print(f"  SUBPROCESSO ISOLADO: {dataset_name} x {strategy_name}")
    print(f"  PID: {os.getpid()} | Memoria sera libertada ao terminar")
    print(f"{'=' * 70}")
    
    start_time = time.time()
    
    try:
        summaries = train_dataset(dataset_name, strategy_name)
        elapsed = time.time() - start_time
        
        print(f"\n  {'─' * 70}")
        print(f"  CONCLUIDO: {dataset_name} x {strategy_name}")
        print(f"  Modelos treinados: {len(summaries)}")
        print(f"  Tempo: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  {'─' * 70}")
        
        sys.exit(0)  # Sucesso
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERRO no dataset {dataset_name}_{strategy_name}: {e}")
        traceback.print_exc()
        
        # Salvar erro para o orquestrador saber
        error_file = os.path.join(
            config.OUTPUT_DIR,
            f"{dataset_name}_{strategy_name}_ERROR.json"
        )
        with open(error_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'strategy': strategy_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'elapsed_seconds': elapsed
            }, f, indent=2)
        
        sys.exit(1)  # Erro


if __name__ == "__main__":
    main()
