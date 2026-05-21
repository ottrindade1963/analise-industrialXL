"""
============================================================
PASSO 1: EXTRAÇÃO DE DADOS REAIS VIA API (WDI + WGI)
África e Médio Oriente
============================================================
Estratégia:
- WDI: Extração via wbgapi (funciona bem com source default)
- WGI: Extração via API REST directa (source 75) porque wbgapi
  com db=3 gera erros de JSON decoding na API do World Bank.
  A source 75 é a nova base de dados ESG que inclui WGI.
============================================================
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# Importar configuração
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def extrair_wdi():
    """Extrai indicadores WDI via wbgapi (World Bank API python package)."""
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
                print(f"    \u2713 {len(df_long)} registos")
            else:
                print(f"    \u2717 Sem dados")
        except Exception as e:
            print(f"    \u2717 Erro: {e}")
    
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


def _fetch_wgi_indicator(indicator_code, paises_str, ano_inicio, ano_fim, max_retries=3):
    """
    Extrai um indicador WGI via API REST directa do World Bank.
    Usa source=75 (ESG Data) que contém os indicadores WGI actualizados.
    
    Args:
        indicator_code: Código do indicador (ex: CC.EST)
        paises_str: String com códigos ISO3 separados por ;
        ano_inicio: Ano inicial
        ano_fim: Ano final
        max_retries: Número máximo de tentativas
    
    Returns:
        Lista de dicionários com {country_code, year, value}
    """
    url = (
        f"https://api.worldbank.org/v2/country/{paises_str}/indicator/{indicator_code}"
        f"?format=json&per_page=5000&date={ano_inicio}:{ano_fim}&source=75"
    )
    
    records = []
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar se a resposta é válida
            if not isinstance(data, list) or len(data) < 2:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return records
            
            # Verificar se há mensagem de erro
            if isinstance(data[0], dict) and 'message' in data[0]:
                error_msg = data[0]['message']
                if isinstance(error_msg, list) and len(error_msg) > 0:
                    print(f"      API Warning: {error_msg[0].get('value', 'Unknown')}")
                return records
            
            # Extrair dados
            total_records = data[0].get('total', 0)
            total_pages = data[0].get('pages', 1)
            
            # Processar primeira página
            for record in data[1]:
                if record.get('value') is not None:
                    records.append({
                        'country_code': record.get('countryiso3code', record['country']['id']),
                        'year': int(record['date']),
                        'value': float(record['value'])
                    })
            
            # Se há mais páginas, buscar todas
            for page in range(2, total_pages + 1):
                page_url = f"{url}&page={page}"
                time.sleep(0.5)  # Rate limiting
                page_response = requests.get(page_url, timeout=90)
                page_data = page_response.json()
                
                if isinstance(page_data, list) and len(page_data) > 1 and page_data[1]:
                    for record in page_data[1]:
                        if record.get('value') is not None:
                            records.append({
                                'country_code': record.get('countryiso3code', record['country']['id']),
                                'year': int(record['date']),
                                'value': float(record['value'])
                            })
            
            return records
            
        except requests.exceptions.Timeout:
            print(f"      Timeout (tentativa {attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"      Erro HTTP (tentativa {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
        except (ValueError, KeyError) as e:
            print(f"      Erro parsing (tentativa {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    
    return records


def extrair_wgi():
    """
    Extrai indicadores WGI via API REST directa do World Bank (source 75).
    
    NOTA: O wbgapi com db=3 gera erros 'JSON decoding error' porque a API
    do World Bank mudou o formato dos dados WGI. A source 75 (ESG Data)
    contém os mesmos indicadores WGI e funciona correctamente via REST.
    """
    print("\n" + "=" * 60)
    print("PASSO 1.2: Extração de dados WGI (via API REST, source 75)")
    print("=" * 60)
    
    indicadores = config.INDICADORES_WGI
    paises = config.PAISES_CODIGOS
    paises_str = ";".join(paises)
    
    print(f"  Indicadores WGI: {len(indicadores)}")
    print(f"  Países: {len(paises)}")
    print(f"  Período: {config.ANO_INICIO}-{config.ANO_FIM}")
    print(f"  Método: API REST directa (source 75)")
    
    all_dfs = []
    
    for ind_code, ind_name in indicadores.items():
        print(f"  Extraindo: {ind_name} ({ind_code})...")
        
        records = _fetch_wgi_indicator(
            ind_code, paises_str, config.ANO_INICIO, config.ANO_FIM
        )
        
        if records:
            df_ind = pd.DataFrame(records)
            df_ind.rename(columns={'value': ind_name}, inplace=True)
            all_dfs.append(df_ind)
            
            # Estatísticas
            n_paises = df_ind['country_code'].nunique()
            n_anos = df_ind['year'].nunique()
            print(f"    \u2713 {len(records)} registos ({n_paises} países, {n_anos} anos)")
        else:
            print(f"    \u2717 Sem dados (API pode estar indisponível)")
        
        # Rate limiting entre indicadores
        time.sleep(1)
    
    if not all_dfs:
        raise RuntimeError(
            "Nenhum dado WGI extraído! Verifique:\n"
            "  1. Conexão à internet\n"
            "  2. API do World Bank pode estar em manutenção\n"
            "  3. Tente novamente em alguns minutos"
        )
    
    # Merge de todos os indicadores WGI
    df_wgi = all_dfs[0]
    for df_extra in all_dfs[1:]:
        cols_merge = [c for c in df_extra.columns if c not in ['country_code', 'year']]
        df_wgi = df_wgi.merge(df_extra, on=['country_code', 'year'], how='outer')
    
    # Adicionar nome do país
    df_wgi['pais'] = df_wgi['country_code'].map(config.PAISES)
    
    # Ordenar
    df_wgi = df_wgi.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    print(f"\n  Dataset WGI final: {df_wgi.shape}")
    n_missing = df_wgi.isnull().sum().sum()
    total_cells = len(df_wgi) * len(df_wgi.columns)
    print(f"  Missing total: {n_missing} ({n_missing/total_cells*100:.1f}%)")
    
    # Resumo por indicador
    print("\n  Cobertura por indicador:")
    for col in df_wgi.columns:
        if col not in ['country_code', 'year', 'pais']:
            n_valid = df_wgi[col].notna().sum()
            pct = n_valid / len(df_wgi) * 100
            print(f"    {col}: {n_valid}/{len(df_wgi)} ({pct:.0f}%)")
    
    return df_wgi


def executar_passo1():
    """Executa o Passo 1 completo."""
    print("\n" + "=" * 70)
    print("  PASSO 1: EXTRAÇÃO DE DADOS - INÍCIO")
    print("=" * 70)
    
    t0 = time.time()
    
    # Criar directório de saída
    os.makedirs(config.DADOS_BRUTOS_DIR, exist_ok=True)
    
    # Extrair WDI
    df_wdi = extrair_wdi()
    wdi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wdi_africa_mo_bruto.csv')
    df_wdi.to_csv(wdi_path, index=False)
    print(f"\n  \u2713 WDI salvo: {wdi_path}")
    
    # Extrair WGI
    df_wgi = extrair_wgi()
    wgi_path = os.path.join(config.DADOS_BRUTOS_DIR, 'wgi_africa_mo_bruto.csv')
    df_wgi.to_csv(wgi_path, index=False)
    print(f"  \u2713 WGI salvo: {wgi_path}")
    
    tempo = time.time() - t0
    
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
            'metodo_wdi': 'wbgapi (source default)',
            'metodo_wgi': 'API REST directa (source 75)',
        },
        metricas={
            'wdi': registar_dataframe_info(df_wdi, 'WDI Bruto'),
            'wgi': registar_dataframe_info(df_wgi, 'WGI Bruto'),
            'tempo_execucao_segundos': round(tempo, 1),
        }
    )
    
    # Auto-save
    auto_save_drive(ficheiros_saida, config)
    
    print(f"\n  Tempo total: {tempo:.1f}s")
    print("\n" + "=" * 70)
    print("  PASSO 1: EXTRAÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 70)
    
    return df_wdi, df_wgi


if __name__ == '__main__':
    executar_passo1()
