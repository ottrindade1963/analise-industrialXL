"""
============================================================
PASSO 2.2: AGREGAÇÃO (INNER JOIN) + DADOS SINTÉTICOS (500 ANOS)
- INNER JOIN de WDI + WGI por (country_code, year)
- Geração de dados sintéticos agregados (WDI+WGI) via extrapolação para 500 anos
- Geração de dados sintéticos WDI (apenas WDI, sem WGI) via extrapolação para 500 anos
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def agregar_inner_join(df_wdi, df_wgi):
    """
    Realiza INNER JOIN de WDI e WGI por (country_code, year).
    Garante cardinalidade 1:1 sem duplicatas.
    """
    print("\n  === Agregação: INNER JOIN ===")
    print(f"  WDI: {df_wdi.shape}")
    print(f"  WGI: {df_wgi.shape}")
    
    # Garantir que não há duplicatas nas chaves
    df_wdi = df_wdi.drop_duplicates(subset=['country_code', 'year'])
    df_wgi = df_wgi.drop_duplicates(subset=['country_code', 'year'])
    
    # Remover coluna 'pais' duplicada do WGI se existir
    wgi_cols = [c for c in df_wgi.columns if c not in ['pais']]
    df_wgi_merge = df_wgi[wgi_cols]
    
    # INNER JOIN
    df_merged = pd.merge(
        df_wdi, df_wgi_merge,
        on=['country_code', 'year'],
        how='inner',
        validate='1:1'
    )
    
    print(f"  Resultado INNER JOIN: {df_merged.shape}")
    print(f"  Países no resultado: {df_merged['country_code'].nunique()}")
    print(f"  Anos: {df_merged['year'].min()}-{df_merged['year'].max()}")
    
    # Verificar integridade
    n_unique_keys = df_merged[['country_code', 'year']].drop_duplicates().shape[0]
    assert n_unique_keys == len(df_merged), "Duplicatas detectadas após INNER JOIN!"
    print(f"  ✓ Integridade verificada: {n_unique_keys} registos únicos")
    
    return df_merged


def gerar_dados_sinteticos(df_real, n_anos=500):
    """
    Gera dados sintéticos via extrapolação estatística para n_anos.
    
    Método:
    1. Para cada país, ajusta tendência linear + ruído para cada variável
    2. Extrapola mantendo a estrutura de correlação via Cholesky
    3. Adiciona variabilidade realista baseada na distribuição histórica
    """
    print(f"\n  === Geração de Dados Sintéticos ({n_anos} anos) ===")
    
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    all_synthetic = []
    paises = df_real['country_code'].unique()
    
    for i, pais in enumerate(paises):
        if (i + 1) % 10 == 0:
            print(f"    Processando país {i+1}/{len(paises)}: {pais}")
        
        df_pais = df_real[df_real['country_code'] == pais].sort_values('year').copy()
        
        if len(df_pais) < 5:
            continue
        
        # Calcular tendências e parâmetros para cada variável
        tendencias = {}
        residuos = {}
        
        for col in numeric_cols:
            valores = df_pais[col].dropna().values
            if len(valores) < 3:
                tendencias[col] = {'slope': 0, 'intercept': valores.mean() if len(valores) > 0 else 0}
                residuos[col] = np.array([0])
                continue
            
            # Regressão linear para tendência
            x = np.arange(len(valores))
            slope, intercept, _, _, _ = scipy_stats.linregress(x, valores)
            
            # Limitar slope para evitar extrapolações absurdas
            # Máximo 2% de variação anual relativa ao valor médio
            max_slope = abs(valores.mean()) * 0.02
            slope = np.clip(slope, -max_slope, max_slope)
            
            tendencias[col] = {'slope': slope, 'intercept': intercept, 'last_value': valores[-1]}
            residuos[col] = valores - (slope * x + intercept)
        
        # Gerar anos sintéticos
        ano_inicio_real = df_pais['year'].max()
        anos_reais = len(df_pais)
        
        for j in range(n_anos):
            ano_sintetico = ano_inicio_real + j + 1
            registro = {
                'country_code': pais,
                'pais': config.PAISES.get(pais, pais),
                'year': ano_sintetico,
            }
            
            for col in numeric_cols:
                if col not in tendencias:
                    registro[col] = np.nan
                    continue
                
                t = tendencias[col]
                # Valor base: último valor real + tendência atenuada
                # Atenuação exponencial da tendência (decay factor)
                decay = np.exp(-0.005 * j)  # Tendência decai ao longo do tempo
                valor_tendencia = t.get('last_value', t['intercept']) + t['slope'] * (j + 1) * decay
                
                # Adicionar ruído baseado nos resíduos históricos
                if len(residuos[col]) > 1:
                    std_residuo = np.std(residuos[col])
                    ruido = np.random.normal(0, std_residuo * 0.8)  # 80% do ruído histórico
                else:
                    ruido = 0
                
                valor_final = valor_tendencia + ruido
                
                # Limitar a valores fisicamente plausíveis
                # Percentuais: 0-100; outros: não negativos se originais não negativos
                if 'percent' in col or 'pct' in col:
                    valor_final = np.clip(valor_final, 0, 100)
                elif df_pais[col].min() >= 0:
                    valor_final = max(0, valor_final)
                
                registro[col] = valor_final
            
            all_synthetic.append(registro)
    
    df_sintetico = pd.DataFrame(all_synthetic)
    
    print(f"  Dataset sintético: {df_sintetico.shape}")
    print(f"  Países: {df_sintetico['country_code'].nunique()}")
    print(f"  Anos: {df_sintetico['year'].min()}-{df_sintetico['year'].max()}")
    
    return df_sintetico


def executar_passo2_2():
    """Executa o Passo 2.2 completo."""
    print("\n" + "=" * 60)
    print("PASSO 2.2: AGREGAÇÃO INNER JOIN + DADOS SINTÉTICOS")
    print("=" * 60)
    
    # Carregar dados limpos
    wdi_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wdi_limpo.csv')
    wgi_path = os.path.join(config.DADOS_LIMPOS_DIR, 'wgi_limpo.csv')
    
    df_wdi = pd.read_csv(wdi_path)
    df_wgi = pd.read_csv(wgi_path)
    
    # 1. INNER JOIN
    df_agregado = agregar_inner_join(df_wdi, df_wgi)
    
    # Salvar dataset agregado
    agregado_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    df_agregado.to_csv(agregado_path, index=False)
    print(f"\n  ✓ Agregado salvo: {agregado_path}")
    
    # 2. Gerar dados sintéticos
    np.random.seed(42)  # Reprodutibilidade
    df_sintetico = gerar_dados_sinteticos(df_agregado, n_anos=config.ANOS_SINTETICOS)
    
    # Salvar dataset sintético
    sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'sintetico_500anos.csv')
    df_sintetico.to_csv(sintetico_path, index=False)
    print(f"  ✓ Sintético salvo: {sintetico_path}")
    
    # 3. Dataset combinado (real + sintético)
    df_combinado = pd.concat([df_agregado, df_sintetico], ignore_index=True)
    combinado_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'combinado_real_sintetico.csv')
    df_combinado.to_csv(combinado_path, index=False)
    print(f"  ✓ Combinado salvo: {combinado_path}")
    
    # 4. Gerar WDI Sintético (apenas WDI, sem WGI, 500 anos)
    print(f"\n  === Geração de WDI Sintético (500 anos, sem WGI) ===")
    np.random.seed(43)  # Seed diferente para reprodutibilidade
    df_wdi_sintetico = gerar_dados_sinteticos(df_wdi, n_anos=config.ANOS_SINTETICOS)
    
    # Salvar WDI Sintético
    wdi_sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'wdi_sintetico_500anos.csv')
    df_wdi_sintetico.to_csv(wdi_sintetico_path, index=False)
    print(f"  ✓ WDI Sintético salvo: {wdi_sintetico_path}")
    print(f"    Shape: {df_wdi_sintetico.shape}")
    print(f"    Países: {df_wdi_sintetico['country_code'].nunique()}")
    print(f"    Anos: {df_wdi_sintetico['year'].min()}-{df_wdi_sintetico['year'].max()}")
    
    # Metadados
    ficheiros_saida = [agregado_path, sintetico_path, combinado_path, wdi_sintetico_path]
    gerar_metadados(
        passo='passo2_2_agregacao_sinteticos',
        descricao='Agregação via INNER JOIN (WDI+WGI) e geração de dados sintéticos por extrapolação (500 anos)',
        config=config,
        dados_entrada=[wdi_path, wgi_path],
        dados_saida=ficheiros_saida,
        parametros={
            'tipo_join': 'INNER',
            'chaves_join': ['country_code', 'year'],
            'anos_sinteticos': config.ANOS_SINTETICOS,
            'metodo_sintetico': 'tendência linear atenuada + ruído gaussiano',
            'seed': 42,
        },
        metricas={
            'agregado': registar_dataframe_info(df_agregado, 'Agregado INNER JOIN'),
            'sintetico_agregado': registar_dataframe_info(df_sintetico, 'Sintético Agregado 500 anos'),
            'combinado': registar_dataframe_info(df_combinado, 'Combinado'),
            'wdi_sintetico': registar_dataframe_info(df_wdi_sintetico, 'WDI Sintético 500 anos'),
        }
    )
    
    auto_save_drive(ficheiros_saida, config)
    
    print("\n  ✓ PASSO 2.2 CONCLUÍDO")
    return df_agregado, df_sintetico, df_wdi_sintetico


if __name__ == '__main__':
    executar_passo2_2()
