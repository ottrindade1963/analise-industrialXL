"""
============================================================
PASSO 1: EXTRAÇÃO DE DADOS REAIS VIA API (WDI + WGI)
África e Médio Oriente
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importar configuração
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def extrair_wdi():
    """Extrai indicadores WDI via wbgapi."""
    import wbgapi as wb
    
    print("=" * 60)
    print("PASSO 1.1: Extração de dados WDI")
    print("=" * 60)
    
    indicadores = list(config.INDICADORES_WDI.keys())
    paises = config.PAISES_CODIGOS
    anos = range(config.ANO_INICIO, config.ANO_FIM + 1)
    
    print(f"  Países: {len(paises)}")
    print(f"  Indicadores: {len(indicadores)}")
    print(f"  Período: {config.ANO_INICIO}-{config.ANO_FIM}")
    
    all_data = []
    
    for ind_code in indicadores:
        ind_name = config.INDICADORES_WDI[ind_code]
        print(f"  Extraindo: {ind_name} ({ind_code})...")
        try:
            df_ind = wb.data.DataFrame(ind_code, paises, time=anos, labels=False)
            if df_ind is not None and not df_ind.empty:
                # Reformatar: linhas=países, colunas=anos → formato longo
                df_long = df_ind.reset_index().melt(
                    id_vars=['economy'],
                    var_name='year',
                    value_name=ind_name
                )
                df_long.rename(columns={'economy': 'country_code'}, inplace=True)
                # Limpar ano (formato YRxxxx → int)
                df_long['year'] = df_long['year'].astype(str).str.replace('YR', '').astype(int)
                all_data.append(df_long)
                print(f"    ✓ {len(df_long)} registos")
            else:
                print(f"    ✗ Sem dados")
        except Exception as e:
            print(f"    ✗ Erro: {e}")
    
    if not all_data:
        raise RuntimeError("Nenhum dado WDI extraído!")
    
    # Merge de todos os indicadores
    df_wdi = all_data[0]
    for df_extra in all_data[1:]:
        cols_merge = [c for c in df_extra.columns if c not in ['country_code', 'year']]
        df_wdi = df_wdi.merge(df_extra, on=['country_code', 'year'], how='outer')
    
    # Adicionar nome do país
    df_wdi['pais'] = df_wdi['country_code'].map(config.PAISES)
    
    # Ordenar
    df_wdi = df_wdi.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    print(f"\n  Dataset WDI final: {df_wdi.shape}")
    print(f"  Missing total: {df_wdi.isnull().sum().sum()} ({df_wdi.isnull().sum().sum()/(len(df_wdi)*len(df_wdi.columns))*100:.1f}%)")
    
    return df_wdi


def extrair_wgi():
    """Extrai indicadores WGI via wbgapi."""
    import wbgapi as wb
    
    print("\n" + "=" * 60)
    print("PASSO 1.2: Extração de dados WGI")
    print("=" * 60)
    
    indicadores = list(config.INDICADORES_WGI.keys())
    paises = config.PAISES_CODIGOS
    anos = range(config.ANO_INICIO, config.ANO_FIM + 1)
    
    print(f"  Indicadores WGI: {len(indicadores)}")
    
    all_data = []
    
    for ind_code in indicadores:
        ind_name = config.INDICADORES_WGI[ind_code]
        print(f"  Extraindo: {ind_name} ({ind_code})...")
        try:
            # WGI está na base 3 do Banco Mundial
            df_ind = wb.data.DataFrame(ind_code, paises, time=anos, 
                                       labels=False, db=3)
            if df_ind is not None and not df_ind.empty:
                df_long = df_ind.reset_index().melt(
                    id_vars=['economy'],
                    var_name='year',
                    value_name=ind_name
                )
                df_long.rename(columns={'economy': 'country_code'}, inplace=True)
                df_long['year'] = df_long['year'].astype(str).str.replace('YR', '').astype(int)
                all_data.append(df_long)
                print(f"    ✓ {len(df_long)} registos")
            else:
                print(f"    ✗ Sem dados")
        except Exception as e:
            print(f"    ✗ Erro: {e}")
    
    if not all_data:
        raise RuntimeError("Nenhum dado WGI extraído!")
    
    # Merge
    df_wgi = all_data[0]
    for df_extra in all_data[1:]:
        df_wgi = df_wgi.merge(df_extra, on=['country_code', 'year'], how='outer')
    
    df_wgi['pais'] = df_wgi['country_code'].map(config.PAISES)
    df_wgi = df_wgi.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    print(f"\n  Dataset WGI final: {df_wgi.shape}")
    print(f"  Missing total: {df_wgi.isnull().sum().sum()} ({df_wgi.isnull().sum().sum()/(len(df_wgi)*len(df_wgi.columns))*100:.1f}%)")
    
    return df_wgi


def executar_passo1():
    """Executa o Passo 1 completo."""
    print("\n" + "=" * 60)
    print("PASSO 1: EXTRAÇÃO DE DADOS - INÍCIO")
    print("=" * 60)
    
    # Extrair WDI
    df_wdi = extrair_wdi()
    wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_africa_mo_bruto.csv')
    df_wdi.to_csv(wdi_path, index=False)
    print(f"\n  ✓ WDI salvo: {wdi_path}")
    
    # Extrair WGI
    df_wgi = extrair_wgi()
    wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_africa_mo_bruto.csv')
    df_wgi.to_csv(wgi_path, index=False)
    print(f"  ✓ WGI salvo: {wgi_path}")
    
    # Gerar metadados
    ficheiros_saida = [wdi_path, wgi_path]
    gerar_metadados(
        passo='passo1_extracao',
        descricao='Extração de dados WDI e WGI via API do Banco Mundial para países da África e Médio Oriente',
        config=config,
        dados_saida=ficheiros_saida,
        parametros={
            'paises': len(config.PAISES_CODIGOS),
            'indicadores_wdi': len(config.INDICADORES_WDI),
            'indicadores_wgi': len(config.INDICADORES_WGI),
            'periodo': f'{config.ANO_INICIO}-{config.ANO_FIM}',
        },
        metricas={
            'wdi': registar_dataframe_info(df_wdi, 'WDI Bruto'),
            'wgi': registar_dataframe_info(df_wgi, 'WGI Bruto'),
        }
    )
    
    # Auto-save
    auto_save_drive(ficheiros_saida, config)
    
    print("\n" + "=" * 60)
    print("PASSO 1: EXTRAÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 60)
    
    return df_wdi, df_wgi


if __name__ == '__main__':
    executar_passo1()
