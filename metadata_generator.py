"""
============================================================
GERADOR DE METADADOS AUTOMÁTICOS
Regista informações de cada passo do pipeline
============================================================
"""
import os
import json
import datetime
import hashlib
import pandas as pd


def gerar_metadados(passo, descricao, config, dados_entrada=None, dados_saida=None,
                    parametros=None, metricas=None, ficheiros_gerados=None, notas=None):
    """
    Gera metadados estruturados para um passo do pipeline.
    
    Args:
        passo: Nome do passo (ex: 'passo1_extracao')
        descricao: Descrição do que o passo faz
        config: Módulo de configuração (config_global)
        dados_entrada: Lista de ficheiros de entrada
        dados_saida: Lista de ficheiros de saída
        parametros: Dicionário de parâmetros usados
        metricas: Dicionário de métricas calculadas
        ficheiros_gerados: Lista de ficheiros gerados
        notas: Notas adicionais
    
    Returns:
        Dicionário de metadados
    """
    meta = {
        'passo': passo,
        'descricao': descricao,
        'projeto': getattr(config, 'PROJETO_NOME', 'Pipeline Industrial'),
        'versao': getattr(config, 'PROJETO_VERSAO', '2.0'),
        'timestamp': datetime.datetime.now().isoformat(),
        'ambiente': 'Google Colab' if getattr(config, '_IN_COLAB', False) else 'Local',
        'dados_entrada': dados_entrada or [],
        'dados_saida': dados_saida or [],
        'parametros': parametros or {},
        'metricas': metricas or {},
        'ficheiros_gerados': ficheiros_gerados or [],
        'notas': notas or '',
    }
    
    # Calcular checksums dos ficheiros de saída
    if dados_saida:
        checksums = {}
        for f in dados_saida:
            if os.path.exists(f):
                with open(f, 'rb') as fh:
                    checksums[os.path.basename(f)] = hashlib.md5(fh.read()).hexdigest()
        meta['checksums'] = checksums
    
    # Salvar metadados
    meta_dir = getattr(config, 'METADADOS_DIR', os.path.join(config.BASE_DIR, 'metadados'))
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f'{passo}_metadata.json')
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[METADADOS] {passo} → {meta_path}")
    return meta


def registar_dataframe_info(df, nome):
    """Retorna informações resumidas de um DataFrame para metadados."""
    info = {
        'nome': nome,
        'linhas': len(df),
        'colunas': len(df.columns),
        'colunas_lista': list(df.columns),
        'tipos': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_total': int(df.isnull().sum().sum()),
        'missing_percent': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        'memoria_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    if 'country_code' in df.columns:
        info['paises'] = sorted(df['country_code'].unique().tolist())
        info['n_paises'] = df['country_code'].nunique()
    if 'year' in df.columns:
        info['anos'] = [int(df['year'].min()), int(df['year'].max())]
    return info


def auto_save_drive(ficheiros, config):
    """Copia ficheiros para Google Drive se em ambiente Colab."""
    if not getattr(config, 'DRIVE_SAVE_DIR', None):
        return
    import shutil
    for f in ficheiros:
        if os.path.exists(f):
            dest = os.path.join(config.DRIVE_SAVE_DIR, os.path.basename(f))
            shutil.copy2(f, dest)
    print(f"[AUTO-SAVE] {len(ficheiros)} ficheiros copiados para Google Drive")
