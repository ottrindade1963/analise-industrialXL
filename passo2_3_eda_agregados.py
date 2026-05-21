"""
============================================================
PASSO 2.3: EDA DOS DADOS AGREGADOS + SINTÉTICOS
============================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, registar_dataframe_info, auto_save_drive


def eda_agregado(df, nome, output_dir):
    """EDA completa do dataset agregado ou sintético."""
    print(f"\n  === EDA: {nome} ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    
    ficheiros = []
    
    # 1. Estatísticas descritivas
    stats = df[numeric_cols].describe().T
    stats_path = os.path.join(output_dir, f'{nome}_stats.csv')
    stats.to_csv(stats_path)
    ficheiros.append(stats_path)
    
    # 2. Correlação com variável alvo
    if config.TARGET_VAR in numeric_cols:
        corr_target = df[numeric_cols].corr()[config.TARGET_VAR].drop(config.TARGET_VAR).sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_target.plot(kind='barh', ax=ax, color=['coral' if v < 0 else 'steelblue' for v in corr_target])
        ax.set_title(f'Correlação com {config.TARGET_VAR}\n{nome}', fontsize=12)
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{nome}_corr_target.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        ficheiros.append(path)
    
    # 3. Boxplots das variáveis WGI
    wgi_cols = [c for c in numeric_cols if 'wgi' in c]
    if wgi_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        df[wgi_cols].boxplot(ax=ax, rot=45)
        ax.set_title(f'Distribuição dos Indicadores WGI - {nome}', fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{nome}_wgi_boxplot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        ficheiros.append(path)
    
    # 4. Evolução temporal da variável alvo por região
    if config.TARGET_VAR in df.columns and 'country_code' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Média por ano
        media_anual = df.groupby('year')[config.TARGET_VAR].agg(['mean', 'std'])
        ax.plot(media_anual.index, media_anual['mean'], 'b-', linewidth=2, label='Média')
        ax.fill_between(media_anual.index,
                       media_anual['mean'] - media_anual['std'],
                       media_anual['mean'] + media_anual['std'],
                       alpha=0.2, color='blue', label='±1 DP')
        ax.set_title(f'Evolução Temporal - {config.TARGET_VAR}\n{nome}', fontsize=12)
        ax.set_xlabel('Ano')
        ax.set_ylabel('Valor (%)')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, f'{nome}_evolucao_temporal.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        ficheiros.append(path)
    
    # 5. Pairplot (amostra de variáveis principais)
    key_vars = [config.TARGET_VAR, 'pib_per_capita_ppc', 'ied_percent_pib',
                'wgi_qualidade_regulatoria', 'wgi_controle_corrupcao']
    key_vars = [v for v in key_vars if v in df.columns]
    
    if len(key_vars) >= 3:
        sample = df[key_vars].dropna().sample(min(500, len(df)), random_state=42)
        fig = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20})
        fig.fig.suptitle(f'Pairplot - {nome}', y=1.02)
        path = os.path.join(output_dir, f'{nome}_pairplot.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        ficheiros.append(path)
    
    return ficheiros


def executar_passo2_3():
    """Executa o Passo 2.3 completo."""
    print("\n" + "=" * 60)
    print("PASSO 2.3: EDA AGREGADOS + SINTÉTICOS")
    print("=" * 60)
    
    output_dir = os.path.join(config.BASE_DIR, 'eda_agregados')
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dados
    agregado_path = os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv')
    sintetico_path = os.path.join(config.DADOS_SINTETICOS_DIR, 'sintetico_500anos.csv')
    
    df_agregado = pd.read_csv(agregado_path)
    df_sintetico = pd.read_csv(sintetico_path)
    
    # EDA Agregado
    ficheiros_agreg = eda_agregado(df_agregado, 'agregado_real', output_dir)
    
    # EDA Sintético
    ficheiros_sint = eda_agregado(df_sintetico, 'sintetico_500', output_dir)
    
    # Comparação real vs sintético
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Comparar distribuições de variáveis-chave
    compare_vars = [config.TARGET_VAR, 'pib_per_capita_ppc', 
                    'wgi_qualidade_regulatoria', 'ied_percent_pib']
    compare_vars = [v for v in compare_vars if v in df_agregado.columns and v in df_sintetico.columns]
    
    for idx, var in enumerate(compare_vars[:4]):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df_agregado[var].dropna(), bins=30, alpha=0.6, label='Real', density=True, color='steelblue')
        ax.hist(df_sintetico[var].dropna(), bins=30, alpha=0.6, label='Sintético', density=True, color='coral')
        ax.set_title(var, fontsize=10)
        ax.legend()
    
    plt.suptitle('Comparação: Dados Reais vs Sintéticos', fontsize=14)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'comparacao_real_vs_sintetico.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metadados
    todos_ficheiros = ficheiros_agreg + ficheiros_sint + [comp_path]
    gerar_metadados(
        passo='passo2_3_eda_agregados',
        descricao='EDA dos dados agregados (INNER JOIN) e sintéticos (500 anos)',
        config=config,
        dados_entrada=[agregado_path, sintetico_path],
        dados_saida=todos_ficheiros,
        metricas={
            'agregado': registar_dataframe_info(df_agregado, 'Agregado'),
            'sintetico': registar_dataframe_info(df_sintetico, 'Sintético'),
        }
    )
    
    auto_save_drive(todos_ficheiros, config)
    
    print("\n  ✓ PASSO 2.3 CONCLUÍDO")
    return df_agregado, df_sintetico


if __name__ == '__main__':
    executar_passo2_3()
