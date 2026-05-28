"""
============================================================
PASSO 10 — INOVAÇÕES METODOLÓGICAS (Nível Doutoramento)
Pipeline de Análise Industrial — África e Médio Oriente
============================================================

Implementa as 3 sugestões do orientador para elevar o trabalho:

  SUGESTÃO 1: Detecção de Quebras Estruturais via Ruptures (PELT)
    - Identifica múltiplos pontos de mudança de regime por país
    - Classifica períodos em: pré-crise, crise, recuperação
    - Gera feature categórica 'regime' para condicionar modelos

  SUGESTÃO 2: Simulação Contrafactual Localizada (Ceteris Paribus)
    - Pergunta: "Se o Estado de Direito de Moçambique subisse 0.5 std,
      qual seria o impacto no valor agregado industrial?"
    - Cruza com SHAP para mostrar COMO a estrutura de decisão muda
    - Gera waterfall plots comparativos (antes vs. depois da reforma)

  SUGESTÃO 3: Predição Conformal para Todos os Modelos
    - Calcula intervalos de predição válidos (distribution-free)
    - Aplica-se a RF, XGBoost, HistGBM, LSTM (não apenas Bayesiano)
    - Compara largura dos intervalos entre arquitecturas e países
    - Análise de quartis (p05, p25, p50, p75, p95)

Saídas
------
- inovacoes_mestrado/quebras_ruptures.csv
- inovacoes_mestrado/regimes_por_pais.csv
- inovacoes_mestrado/contrafactual_localizado.csv
- inovacoes_mestrado/contrafactual_shap_comparativo.csv
- inovacoes_mestrado/conformal_intervalos.csv
- inovacoes_mestrado/conformal_cobertura.csv
- inovacoes_mestrado/conformal_quartis.csv
- inovacoes_mestrado/*.png (10+ gráficos)

Dependências
------------
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, shap
Opcionais: ruptures (pip install ruptures)

Referências
-----------
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection
  of changepoints with a linear computational cost. JASA, 107(500).
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning
  in a Random World. Springer.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized
  Quantile Regression. NeurIPS.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to
  interpreting model predictions. NeurIPS.
"""

import os
import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Importar configuração global
try:
    import config_global as config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config_global as config

# Directório de saída
INOVACOES_DIR = os.path.join(config.BASE_DIR, 'inovacoes_mestrado')
os.makedirs(INOVACOES_DIR, exist_ok=True)


# ============================================================
# UTILIDADES COMUNS
# ============================================================

def carregar_dados_agregados():
    """
    Carrega o dataset agregado principal.
    
    FICHEIROS DE ENTRADA (por ordem de prioridade):
    ================================================
    
    1.º  dados_engenharia/agregado_features.csv  (PREFERENCIAL)
        → Gerado pelo Passo 3 (Engenharia de Features)
        → Contém features derivadas: lags, deltas, PCA, interações
        → MELHOR OPÇÃO: compatível com os modelos treinados no Passo 4
    
    2.º  dados_agregados/agregado_inner_join.csv  (FALLBACK)
        → Gerado pelo Passo 2.2 (Agregação)
        → Contém apenas variáveis originais WDI + WGI
        → AVISO: Pode ter menos features que os modelos esperam
    
    COLUNAS OBRIGATÓRIAS:
    - country_code: código ISO3 do país (ex: ZAF, NGA, ETH)
    - year: ano da observação (1996-2023)
    - valor_agregado_industrial_percent_pib: variável alvo (TARGET)
    - Colunas wgi_*: indicadores de governança (para contrafactual)
    
    CRITÉRIOS DE VALIDAÇÃO:
    - Mínimo 50 observações
    - Mínimo 3 países
    - Variável alvo presente e com <50% NaN
    """
    # Ordem de prioridade: features primeiro (compatível com modelos)
    caminhos = [
        ('dados_engenharia/agregado_features.csv',
         os.path.join(config.DADOS_ENGENHARIA_DIR, 'agregado_features.csv'),
         'Passo 3 — com features derivadas (lags, PCA, interações)'),
        ('dados_agregados/agregado_inner_join.csv',
         os.path.join(config.DADOS_AGREGADOS_DIR, 'agregado_inner_join.csv'),
         'Passo 2.2 — variáveis originais WDI+WGI'),
        ('dados_engenharia/dados_engenharia_agregado.csv',
         os.path.join(config.DADOS_ENGENHARIA_DIR, 'dados_engenharia_agregado.csv'),
         'Passo 3 — nome alternativo'),
    ]
    
    for nome_rel, caminho_abs, descricao in caminhos:
        if os.path.exists(caminho_abs):
            df = pd.read_csv(caminho_abs)
            
            # Validação de colunas obrigatórias
            colunas_obrigatorias = ['country_code', 'year', config.TARGET_VAR]
            colunas_faltam = [c for c in colunas_obrigatorias if c not in df.columns]
            
            if colunas_faltam:
                print(f"  AVISO: {nome_rel} não contém colunas obrigatórias: {colunas_faltam}")
                print(f"         A tentar próximo ficheiro...")
                continue
            
            # Validação de qualidade mínima
            n_obs = len(df)
            n_paises = df['country_code'].nunique()
            pct_nan_target = df[config.TARGET_VAR].isna().mean() * 100
            
            if n_obs < 50:
                print(f"  AVISO: {nome_rel} tem apenas {n_obs} obs (mínimo: 50). A tentar próximo...")
                continue
            
            print(f"  ✓ Dados carregados: {nome_rel}")
            print(f"    Fonte: {descricao}")
            print(f"    Dimensão: {n_obs} obs × {df.shape[1]} variáveis")
            print(f"    Países: {n_paises} | Período: {df['year'].min()}-{df['year'].max()}")
            print(f"    NaN no target: {pct_nan_target:.1f}%")
            
            # Verificar presença de WGI (necessário para Sugestão 2)
            wgi_cols = [c for c in df.columns if 'wgi' in c.lower()]
            if len(wgi_cols) == 0:
                print(f"    ⚠ Sem colunas WGI — Sugestão 2 (Contrafactual) não funcionará")
            else:
                print(f"    Colunas WGI: {len(wgi_cols)} encontradas")
            
            return df
    
    # Nenhum ficheiro encontrado — erro detalhado
    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  ERRO: NENHUM DATASET ENCONTRADO                         ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print(f"  O Passo 10 requer dados gerados pelos Passos 1-3.")
    print(f"  ")
    print(f"  Ficheiros procurados (nenhum encontrado):")
    for nome_rel, caminho_abs, descricao in caminhos:
        print(f"    ✗ {caminho_abs}")
        print(f"      ({descricao})")
    print(f"  ")
    print(f"  ACÇÃO: Execute os passos anteriores na ordem:")
    print(f"    1. from passo1_extracao import executar_passo1; executar_passo1()")
    print(f"    2. from passo2_1_limpeza import executar_passo2; executar_passo2()")
    print(f"    3. from passo2_2_agregacao_sinteticos import executar_passo2_2; executar_passo2_2()")
    print(f"    4. from passo3_engenharia_features import executar_passo3; executar_passo3()")
    return None


def carregar_modelo(nome_modelo, dataset='Agregado'):
    """Carrega um modelo treinado do directório de modelos."""
    path = os.path.join(config.MODELOS_DIR, f'modelo_{dataset}_{nome_modelo}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    return None


def detectar_colunas_wgi(df):
    """Detecta colunas WGI no dataframe."""
    wgi_cols = [c for c in df.columns if 'wgi' in c.lower() or 'pca1' in c.lower()]
    return wgi_cols


def auto_save_drive(ficheiros, config):
    """Copia ficheiros para Google Drive se disponível."""
    if config.DRIVE_SAVE_DIR:
        import shutil
        for f in ficheiros:
            if os.path.exists(f):
                try:
                    shutil.copy2(f, config.DRIVE_SAVE_DIR)
                except Exception:
                    pass


# ============================================================
# SUGESTÃO 1: DETECÇÃO DE QUEBRAS ESTRUTURAIS (RUPTURES)
# ============================================================

def executar_quebras_ruptures(df):
    """
    Detecta múltiplos pontos de mudança de regime por país usando
    o algoritmo PELT (Pruned Exact Linear Time) da biblioteca Ruptures.
    
    Algoritmo PELT (Killick et al., 2012):
    - Minimiza: sum_{i=0}^{m} [C(y_{tau_i+1:tau_{i+1}}) + beta]
    - Onde C é a função de custo (variância do segmento)
    - beta é a penalização (controla o número de quebras)
    - Utiliza poda para complexidade O(n) em vez de O(n^2)
    
    Classificação de Regimes:
    - Compara a média de cada segmento com a média global
    - Se média_segmento < média_global - 0.5*std: CRISE
    - Se média_segmento > média_global + 0.5*std: EXPANSÃO
    - Caso contrário: ESTÁVEL
    
    Returns:
        tuple: (df_quebras, df_regimes) DataFrames com resultados
    """
    print("\n" + "="*60)
    print("  SUGESTÃO 1: DETECÇÃO DE QUEBRAS ESTRUTURAIS (RUPTURES)")
    print("="*60)
    
    target = config.TARGET_VAR
    
    # Tentar importar ruptures
    try:
        import ruptures as rpt
        usar_ruptures = True
        print("  Biblioteca 'ruptures' disponível — usando PELT")
    except ImportError:
        usar_ruptures = False
        print("  AVISO: 'ruptures' não instalado — usando fallback CUSUM")
        print("  Para instalar: pip install ruptures")
    
    if 'country_code' not in df.columns or 'year' not in df.columns:
        print("  ERRO: Colunas 'country_code' e 'year' necessárias.")
        return pd.DataFrame(), pd.DataFrame()
    
    if target not in df.columns:
        print(f"  ERRO: Variável alvo '{target}' não encontrada.")
        return pd.DataFrame(), pd.DataFrame()
    
    resultados_quebras = []
    resultados_regimes = []
    paises = sorted(df['country_code'].unique())
    
    print(f"  Analisando {len(paises)} países...")
    
    for pais in paises:
        df_pais = df[df['country_code'] == pais].sort_values('year').copy()
        serie = df_pais[target].dropna().values
        anos = df_pais.loc[df_pais[target].notna(), 'year'].values
        
        if len(serie) < 8:  # Mínimo para detecção fiável
            continue
        
        # --- Detecção de Quebras ---
        breakpoints = []
        
        if usar_ruptures:
            # PELT com modelo de custo "rbf" (kernel gaussiano)
            # Penalização: pen = ln(n) * sigma^2 (critério BIC modificado)
            # Para séries curtas (<30), reduzir penalização para maior sensibilidade
            sigma2 = np.var(serie)
            if len(serie) < 30:
                pen = np.log(len(serie)) * sigma2 * 0.5  # Menos conservador para séries curtas
            else:
                pen = np.log(len(serie)) * sigma2 * 2.0  # Penalização conservadora
            
            try:
                # Algoritmo PELT com modelo l2 (mais sensível a mudanças de média)
                algo = rpt.Pelt(model="l2", min_size=3, jump=1)
                algo.fit(serie)
                bkps = algo.predict(pen=pen)
                # Remover o último ponto (é sempre n)
                breakpoints = [b for b in bkps if b < len(serie)]
                
                # Se não encontrou com PELT, tentar Binseg com n_bkps fixo
                if len(breakpoints) == 0:
                    algo2 = rpt.Binseg(model="l2", min_size=3, jump=1)
                    algo2.fit(serie)
                    # Tentar detectar até 2 quebras
                    bkps2 = algo2.predict(n_bkps=min(2, max(1, len(serie)//8)))
                    breakpoints = [b for b in bkps2 if b < len(serie)]
            except Exception as e:
                # Fallback: Binseg se PELT falhar
                try:
                    algo = rpt.Binseg(model="l2", min_size=3, jump=1)
                    algo.fit(serie)
                    bkps = algo.predict(n_bkps=min(3, len(serie)//5))
                    breakpoints = [b for b in bkps if b < len(serie)]
                except Exception:
                    breakpoints = []
        else:
            # Fallback: CUSUM (Cumulative Sum) manual
            # Detecta mudanças na média usando desvios acumulados
            media_global = np.mean(serie)
            cusum = np.cumsum(serie - media_global)
            # Pontos onde o CUSUM muda de direcção significativamente
            for i in range(2, len(serie) - 2):
                antes = np.mean(serie[:i])
                depois = np.mean(serie[i:])
                # Teste t simplificado
                diff = abs(depois - antes)
                se = np.sqrt(np.var(serie[:i])/i + np.var(serie[i:])/(len(serie)-i))
                if se > 0 and diff / se > 2.0:  # Limiar de 2 desvios padrão
                    breakpoints.append(i)
            # Manter apenas os mais significativos (máximo 3)
            if len(breakpoints) > 3:
                # Seleccionar os com maior diferença de médias
                diffs = []
                for bp in breakpoints:
                    d = abs(np.mean(serie[bp:]) - np.mean(serie[:bp]))
                    diffs.append(d)
                idx_top = np.argsort(diffs)[-3:]
                breakpoints = sorted([breakpoints[i] for i in idx_top])
        
        # Converter índices para anos
        anos_quebra = [int(anos[bp]) if bp < len(anos) else int(anos[-1]) 
                       for bp in breakpoints]
        
        # Registar quebras
        for i, (bp, ano_q) in enumerate(zip(breakpoints, anos_quebra)):
            # Calcular magnitude da mudança
            if bp > 0 and bp < len(serie):
                media_antes = np.mean(serie[max(0, bp-3):bp])
                media_depois = np.mean(serie[bp:min(len(serie), bp+3)])
                magnitude = media_depois - media_antes
            else:
                magnitude = 0.0
            
            resultados_quebras.append({
                'Pais': pais,
                'Pais_Nome': config.PAISES.get(pais, pais),
                'Quebra_Num': i + 1,
                'Ano_Quebra': ano_q,
                'Indice': bp,
                'Media_Antes': media_antes if bp > 0 else np.nan,
                'Media_Depois': media_depois if bp < len(serie) else np.nan,
                'Magnitude': magnitude,
                'Metodo': 'PELT' if usar_ruptures else 'CUSUM',
            })
        
        # --- Classificação de Regimes ---
        # Dividir a série em segmentos baseados nas quebras
        todos_bkps = [0] + breakpoints + [len(serie)]
        media_global = np.mean(serie)
        std_global = np.std(serie)
        
        for seg_idx in range(len(todos_bkps) - 1):
            inicio_idx = todos_bkps[seg_idx]
            fim_idx = todos_bkps[seg_idx + 1]
            
            if inicio_idx >= len(anos) or fim_idx > len(anos):
                continue
            
            segmento = serie[inicio_idx:fim_idx]
            media_seg = np.mean(segmento)
            
            # Classificação do regime
            if media_seg < media_global - 0.5 * std_global:
                regime = 'CRISE'
            elif media_seg > media_global + 0.5 * std_global:
                regime = 'EXPANSAO'
            else:
                regime = 'ESTAVEL'
            
            # Verificar se coincide com eventos conhecidos
            ano_inicio = int(anos[inicio_idx])
            ano_fim = int(anos[min(fim_idx - 1, len(anos) - 1)])
            
            # Eventos globais conhecidos para contextualização
            evento = ''
            if ano_inicio <= 2008 <= ano_fim:
                evento = 'Crise Financeira Global'
            elif ano_inicio <= 2014 <= ano_fim and media_seg < media_global:
                evento = 'Choque Commodities'
            elif ano_inicio <= 2020 <= ano_fim:
                evento = 'COVID-19'
            
            # Registar regime para cada ano do segmento
            for idx in range(inicio_idx, min(fim_idx, len(anos))):
                resultados_regimes.append({
                    'Pais': pais,
                    'Pais_Nome': config.PAISES.get(pais, pais),
                    'Year': int(anos[idx]),
                    'Regime': regime,
                    'Segmento': seg_idx + 1,
                    'Media_Segmento': media_seg,
                    'Evento_Associado': evento,
                })
    
    # Criar DataFrames
    df_quebras = pd.DataFrame(resultados_quebras)
    df_regimes = pd.DataFrame(resultados_regimes)
    
    # Exportar CSVs
    quebras_path = os.path.join(INOVACOES_DIR, 'quebras_ruptures.csv')
    regimes_path = os.path.join(INOVACOES_DIR, 'regimes_por_pais.csv')
    df_quebras.to_csv(quebras_path, index=False)
    df_regimes.to_csv(regimes_path, index=False)
    
    # Estatísticas
    n_paises_com_quebra = df_quebras['Pais'].nunique() if len(df_quebras) > 0 else 0
    n_total_quebras = len(df_quebras)
    print(f"  Quebras detectadas: {n_total_quebras} em {n_paises_com_quebra} países")
    
    if len(df_regimes) > 0:
        regime_counts = df_regimes['Regime'].value_counts()
        print(f"  Distribuição de regimes: {dict(regime_counts)}")
    
    # --- GRÁFICOS ---
    
    # Gráfico 1: Timeline de quebras por país
    if len(df_quebras) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        paises_plot = df_quebras['Pais'].unique()[:20]  # Top 20
        for i, pais in enumerate(paises_plot):
            quebras_pais = df_quebras[df_quebras['Pais'] == pais]
            for _, row in quebras_pais.iterrows():
                color = 'red' if row['Magnitude'] < 0 else 'green'
                ax.scatter(row['Ano_Quebra'], i, c=color, s=abs(row['Magnitude'])*20+50,
                          alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_yticks(range(len(paises_plot)))
        ax.set_yticklabels([config.PAISES.get(p, p) for p in paises_plot], fontsize=8)
        ax.set_xlabel('Ano da Quebra Estrutural', fontsize=11)
        ax.set_title('Quebras Estruturais Detectadas (PELT/Ruptures)\nVermelho = Declínio | Verde = Expansão',
                    fontsize=12, fontweight='bold')
        ax.axvline(2008, color='orange', linestyle='--', alpha=0.5, label='Crise 2008')
        ax.axvline(2014, color='purple', linestyle='--', alpha=0.5, label='Choque Commodities')
        ax.axvline(2020, color='brown', linestyle='--', alpha=0.5, label='COVID-19')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'quebras_timeline.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Gráfico 2: Distribuição de regimes ao longo do tempo
    if len(df_regimes) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        regime_por_ano = df_regimes.groupby(['Year', 'Regime']).size().unstack(fill_value=0)
        if len(regime_por_ano) > 0:
            cores_regime = {'CRISE': 'red', 'ESTAVEL': 'steelblue', 'EXPANSAO': 'green'}
            regime_por_ano.plot(kind='area', stacked=True, ax=ax,
                              color=[cores_regime.get(c, 'gray') for c in regime_por_ano.columns],
                              alpha=0.7)
            ax.set_xlabel('Ano', fontsize=11)
            ax.set_ylabel('Número de Países', fontsize=11)
            ax.set_title('Distribuição de Regimes Económicos ao Longo do Tempo\n(Classificação via Ruptures)',
                        fontsize=12, fontweight='bold')
            ax.legend(title='Regime', loc='upper left')
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'regimes_temporal.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Gráfico 3: Exemplo detalhado para 4 países
    if len(df_quebras) > 0:
        paises_exemplo = ['ZAF', 'NGA', 'TUR', 'MAR']
        paises_disp = [p for p in paises_exemplo if p in df['country_code'].unique()][:4]
        
        if len(paises_disp) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, pais in enumerate(paises_disp):
                ax = axes[idx]
                df_p = df[df['country_code'] == pais].sort_values('year')
                serie_p = df_p[target].values
                anos_p = df_p['year'].values
                
                ax.plot(anos_p, serie_p, 'b-o', markersize=3, linewidth=1.5, label='Observado')
                
                # Marcar quebras
                quebras_p = df_quebras[df_quebras['Pais'] == pais]
                for _, row in quebras_p.iterrows():
                    ax.axvline(row['Ano_Quebra'], color='red', linestyle='--', alpha=0.7)
                    ax.annotate(f"{int(row['Ano_Quebra'])}", 
                              xy=(row['Ano_Quebra'], ax.get_ylim()[1]*0.9),
                              fontsize=8, color='red', ha='center')
                
                # Colorir regimes
                regimes_p = df_regimes[df_regimes['Pais'] == pais]
                for _, row in regimes_p.iterrows():
                    cor_fundo = {'CRISE': 'red', 'ESTAVEL': 'lightblue', 'EXPANSAO': 'lightgreen'}
                    ax.axvspan(row['Year']-0.5, row['Year']+0.5, 
                              alpha=0.1, color=cor_fundo.get(row['Regime'], 'white'))
                
                ax.set_title(f"{config.PAISES.get(pais, pais)} — Quebras e Regimes", fontweight='bold')
                ax.set_xlabel('Ano')
                ax.set_ylabel('VAI (% PIB)')
                ax.grid(alpha=0.3)
            
            # Desactivar eixos não usados
            for idx in range(len(paises_disp), 4):
                axes[idx].set_visible(False)
            
            plt.suptitle('Detecção de Quebras Estruturais — Exemplos por País', 
                        fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(INOVACOES_DIR, 'quebras_exemplos_paises.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  Ficheiros exportados: {quebras_path}")
    print(f"                        {regimes_path}")
    print(f"  Gráficos: 3 PNG gerados")
    
    return df_quebras, df_regimes



# ============================================================
# SUGESTÃO 2: SIMULAÇÃO CONTRAFACTUAL LOCALIZADA (CETERIS PARIBUS)
# ============================================================

def executar_contrafactual_localizado(df):
    """
    Simulação contrafactual localizada por país com projecção inversa PCA.
    
    Pergunta central: "Se o índice de Estado de Direito de um país
    específico aumentasse 0.5 desvios-padrão, mantendo todo o resto
    constante (ceteris paribus), qual seria o impacto previsto no
    valor agregado industrial?"
    
    ENTRADAS NECESSÁRIAS:
    =====================
    1. Dados (features):  dados_engenharia/agregado_features.csv  ← Passo 3
    2. Modelo preditivo:  modelos/modelo_Agregado_*.pkl            ← Passo 4
    3. Transformador PCA: dados_engenharia/pca_models/agregado_pca_model.pkl  ← Passo 3
    4. Scaler WGI:        dados_engenharia/pca_models/agregado_pca_scaler.pkl ← Passo 3
    
    METODOLOGIA (Projecção Inversa PCA):
    =====================================
    O dataset de features contém colunas PCA (wgi_pca1_lag1, etc.),
    não os 6 indicadores WGI originais. Para perturbar um indicador
    individual (ex: Estado de Direito), o algoritmo:
    
    1. Carrega o PCA model e scaler salvos pelo Passo 3.
    2. Obtém o score PCA actual do país (wgi_pca1_lag1).
    3. Faz a projecção inversa: PCA_score → 6 WGI originais (aproximados).
    4. Perturba APENAS o indicador alvo (ex: rule_of_law += 0.5*std).
    5. Reprojecta para o espaço PCA: 6 WGI perturbados → novo PCA_score.
    6. Substitui wgi_pca1_lag1 pelo novo valor no vector de features.
    7. Prevê com o modelo treinado.
    8. Calcula valores SHAP antes e depois.
    
    Isto é matematicamente rigoroso porque usa a MESMA transformação
    (scaler + PCA) que foi aplicada durante o treino no Passo 3/4.
    
    Returns:
        tuple: (df_contrafactual, df_shap_comparativo) DataFrames
    """
    print("\n" + "="*60)
    print("  SUGESTÃO 2: SIMULAÇÃO CONTRAFACTUAL LOCALIZADA + SHAP")
    print("  (com Projecção Inversa PCA para indicadores individuais)")
    print("="*60)
    
    target = config.TARGET_VAR
    
    # ================================================================
    # CARREGAR ARTEFACTOS NECESSÁRIOS
    # ================================================================
    
    # 1. Modelo preditivo (Passo 4)
    model = None
    model_nome = None
    for nome in ['RandomForest', 'XGBoost', 'HistGBM']:
        model = carregar_modelo(nome)
        if model is not None:
            model_nome = nome
            print(f"  Modelo preditivo carregado: {nome} (Passo 4)")
            break
    
    if model is None:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  ERRO: NENHUM MODELO PREDITIVO ENCONTRADO              ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        print(f"  O Passo 10 NÃO treina modelos. Requer modelos do Passo 4.")
        print(f"  Ficheiros procurados:")
        for nome in ['RandomForest', 'XGBoost', 'HistGBM']:
            path = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{nome}.pkl')
            existe = '✓' if os.path.exists(path) else '✗'
            print(f"    {existe} {path}")
        print(f"  ACÇÃO: Execute primeiro o Passo 4 (Treino de Modelos).")
        return pd.DataFrame(), pd.DataFrame()
    
    # 2. Transformadores PCA (Passo 3)
    pca_model_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'pca_models', 'agregado_pca_model.pkl')
    pca_scaler_path = os.path.join(config.DADOS_ENGENHARIA_DIR, 'pca_models', 'agregado_pca_scaler.pkl')
    
    pca_disponivel = False
    pca_model = None
    pca_scaler = None
    wgi_originais_nomes = None
    
    if os.path.exists(pca_model_path) and os.path.exists(pca_scaler_path):
        try:
            import joblib
            pca_model = joblib.load(pca_model_path)
            pca_scaler = joblib.load(pca_scaler_path)
            pca_disponivel = True
            
            # Nomes dos 6 indicadores WGI originais (do scaler)
            if hasattr(pca_scaler, 'feature_names_in_'):
                wgi_originais_nomes = list(pca_scaler.feature_names_in_)
            else:
                wgi_originais_nomes = [
                    'wgi_control_corruption', 'wgi_government_effectiveness',
                    'wgi_political_stability', 'wgi_regulatory_quality',
                    'wgi_rule_of_law', 'wgi_voice_accountability'
                ]
            
            print(f"  Transformador PCA carregado (Passo 3):")
            print(f"    Scaler: {pca_scaler_path}")
            print(f"    PCA: {pca_model_path}")
            print(f"    Indicadores WGI originais: {wgi_originais_nomes}")
            print(f"    N. componentes PCA: {pca_model.n_components_}")
            print(f"    Variância explicada: {pca_model.explained_variance_ratio_.sum():.1%}")
            print(f"  → Modo: PROJECÇÃO INVERSA (perturba indicadores individuais)")
        except Exception as e:
            print(f"  AVISO: Erro ao carregar PCA: {e}")
            print(f"  → Modo: PERTURBAÇÃO DIRECTA (perturba features PCA compostas)")
    else:
        print(f"  Transformadores PCA não encontrados:")
        pca_m_status = '\u2713' if os.path.exists(pca_model_path) else '\u2717'
        pca_s_status = '\u2713' if os.path.exists(pca_scaler_path) else '\u2717'
        print(f"    {pca_m_status} {pca_model_path}")
        print(f"    {pca_s_status} {pca_scaler_path}")
        print(f"  → Modo: PERTURBAÇÃO DIRECTA (perturba features PCA compostas)")
        print(f"    (Para perturbar indicadores individuais, execute o Passo 3 primeiro)")
    
    # ================================================================
    # PREPARAR DADOS E FEATURES
    # ================================================================
    
    colunas_excluir = ['country_code', 'year', target, 'pais', 'country_name']
    feature_cols = [c for c in df.columns if c not in colunas_excluir and df[c].dtype in ['float64', 'int64', 'float32']]
    
    if len(feature_cols) == 0:
        print("  ERRO: Nenhuma feature numérica encontrada.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Identificar colunas PCA no dataset (para substituir após reprojectar)
    pca_feature_cols = [c for c in feature_cols if 'pca1' in c.lower() and 'wgi' in c.lower()]
    
    # ================================================================
    # DEFINIR VARIÁVEIS DE PERTURBAÇÃO
    # ================================================================
    
    # Os 6 indicadores WGI originais (para perturbação via PCA inverso)
    wgi_perturbacao_originais = [
        ('wgi_rule_of_law', 'Estado de Direito'),
        ('wgi_regulatory_quality', 'Qualidade Regulatória'),
        ('wgi_control_corruption', 'Controlo de Corrupção'),
        ('wgi_government_effectiveness', 'Eficácia do Governo'),
        ('wgi_political_stability', 'Estabilidade Política'),
        ('wgi_voice_accountability', 'Voz e Responsabilização'),
    ]
    
    # Filtrar apenas os que existem no scaler
    if pca_disponivel and wgi_originais_nomes:
        vars_perturbacao = [(nome, label) for nome, label in wgi_perturbacao_originais 
                           if nome in wgi_originais_nomes]
    else:
        # Fallback: perturbar directamente as features PCA no dataset
        vars_perturbacao = [(c, c) for c in pca_feature_cols[:6]]
        if len(vars_perturbacao) == 0:
            # Último recurso: qualquer coluna WGI
            wgi_any = [c for c in feature_cols if 'wgi' in c.lower() or 'pca1' in c.lower()]
            vars_perturbacao = [(c, c) for c in wgi_any[:6]]
    
    if len(vars_perturbacao) == 0:
        print("  ERRO: Nenhuma variável de governança identificada para perturbação.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"  Variáveis de perturbação ({len(vars_perturbacao)}):")
    for nome, label in vars_perturbacao:
        print(f"    → {nome} ({label})")
    
    # Magnitudes de perturbação (em desvios-padrão)
    magnitudes = [-1.0, -0.5, 0.0, +0.5, +1.0]
    
    # Configurar SHAP
    usar_shap = False
    try:
        import shap
        usar_shap = True
        print("  SHAP disponível — gerando explicações contrafactuais")
    except ImportError:
        print("  AVISO: SHAP não disponível. Apenas previsões pontuais.")
    
    # ================================================================
    # FUNÇÃO AUXILIAR: PROJECÇÃO INVERSA PCA
    # ================================================================
    
    def perturbar_via_pca_inverso(X_base_row, feature_cols, var_wgi_original, magnitude_std):
        """
        Perturba um indicador WGI individual e reprojecta para o espaço PCA.
        
        Algoritmo:
        1. Extrair o score PCA actual (wgi_pca1_lag1) do vector de features.
        2. Inverter: PCA_score → 6 WGI normalizados (via pca_model.inverse_transform).
        3. Des-normalizar: WGI_norm → WGI_original (via scaler.inverse_transform).
        4. Perturbar o indicador alvo: WGI_i += magnitude * std_original_i.
        5. Re-normalizar: WGI_perturbado → WGI_norm_perturbado.
        6. Reprojectar: WGI_norm_perturbado → novo PCA_score.
        7. Substituir no vector de features todas as colunas PCA.
        """
        X_cf = X_base_row.copy()
        
        # Encontrar a coluna pca1 principal (sem lag) ou lag1
        pca1_col = None
        for candidato in ['wgi_pca1_lag1', 'wgi_pca1', 'pca1_lag1']:
            if candidato in feature_cols:
                pca1_col = candidato
                break
        
        if pca1_col is None:
            return X_cf  # Não consegue fazer a projecção
        
        pca1_idx = feature_cols.index(pca1_col)
        pca1_actual = X_cf[0, pca1_idx]
        
        # Construir vector PCA completo (n_components dimensoes)
        n_comp = pca_model.n_components_
        pca_scores = np.zeros((1, n_comp))
        pca_scores[0, 0] = pca1_actual
        
        # Verificar se existem PC2, PC3 no dataset
        for pc_i in range(1, n_comp):
            for candidato in [f'wgi_pca{pc_i+1}_lag1', f'wgi_pca{pc_i+1}', f'pca{pc_i+1}_lag1']:
                if candidato in feature_cols:
                    pca_scores[0, pc_i] = X_cf[0, feature_cols.index(candidato)]
                    break
        
        # Projecção inversa: PCA → WGI normalizados
        wgi_normalizado = pca_model.inverse_transform(pca_scores)  # shape (1, 6)
        
        # Des-normalizar: WGI_norm → WGI_original
        wgi_original = pca_scaler.inverse_transform(wgi_normalizado)  # shape (1, 6)
        
        # Perturbar o indicador alvo
        idx_wgi = wgi_originais_nomes.index(var_wgi_original)
        std_original = pca_scaler.scale_[idx_wgi]  # std do indicador no espaço original
        wgi_original[0, idx_wgi] += magnitude_std * std_original
        
        # Re-normalizar
        wgi_norm_perturbado = pca_scaler.transform(wgi_original)  # shape (1, 6)
        
        # Reprojectar para PCA
        pca_scores_novo = pca_model.transform(wgi_norm_perturbado)  # shape (1, n_comp)
        
        # Substituir no vector de features
        X_cf[0, pca1_idx] = pca_scores_novo[0, 0]
        
        # Substituir também lag2, ma3, delta se existirem
        # (propagar a perturbação para features derivadas do PCA)
        delta_pca = pca_scores_novo[0, 0] - pca1_actual
        for col_derivada in pca_feature_cols:
            if col_derivada == pca1_col:
                continue
            if col_derivada in feature_cols:
                idx_d = feature_cols.index(col_derivada)
                # Para lag2, ma3: aplicar proporção do delta
                # (simplificação: assume que a perturbação é persistente)
                if 'lag2' in col_derivada:
                    X_cf[0, idx_d] += delta_pca * 0.8  # Atenuação temporal
                elif 'ma3' in col_derivada:
                    X_cf[0, idx_d] += delta_pca * 0.6  # Média móvel atenua
                elif 'delta' in col_derivada:
                    X_cf[0, idx_d] += delta_pca  # Delta capta a mudança
        
        # Propagar para interações (inter_pca1_*)
        for col_inter in feature_cols:
            if 'inter_pca1' in col_inter or 'inter_pca' in col_inter:
                idx_inter = feature_cols.index(col_inter)
                # Interação = pca1 * outra_var → recalcular
                # Proporção: novo = antigo * (novo_pca / antigo_pca)
                if pca1_actual != 0:
                    ratio = pca_scores_novo[0, 0] / pca1_actual
                    X_cf[0, idx_inter] = X_cf[0, idx_inter] * ratio
        
        return X_cf
    
    # ================================================================
    # LOOP PRINCIPAL: SIMULAÇÃO CONTRAFACTUAL
    # ================================================================
    
    resultados_cf = []
    resultados_shap = []
    
    paises = sorted(df['country_code'].unique())
    modo = "PROJECÇÃO INVERSA PCA" if pca_disponivel else "PERTURBAÇÃO DIRECTA"
    print(f"  Modo: {modo}")
    print(f"  Simulando choques para {len(paises)} países × {len(vars_perturbacao)} indicadores × {len(magnitudes)} magnitudes...")
    
    for pais in paises:
        df_pais = df[df['country_code'] == pais].sort_values('year')
        
        if len(df_pais) == 0:
            continue
        
        # Usar última observação disponível
        ultima_obs = df_pais.iloc[-1]
        ano_ref = int(ultima_obs['year']) if 'year' in df_pais.columns else 2023
        
        # Vector de features base
        X_base = ultima_obs[feature_cols].values.astype(float).reshape(1, -1)
        X_base = np.nan_to_num(X_base, nan=0.0)
        
        # Verificar compatibilidade de features com o modelo
        n_features_modelo = getattr(model, 'n_features_in_', None)
        if n_features_modelo is not None and X_base.shape[1] != n_features_modelo:
            continue  # Incompatibilidade — saltar este país
        
        # Previsão base
        try:
            y_base = float(model.predict(X_base)[0])
        except Exception:
            continue
        
        # Valor real (se disponível)
        y_real = float(ultima_obs[target]) if pd.notna(ultima_obs[target]) else np.nan
        
        for var_nome, var_label in vars_perturbacao:
            for mag in magnitudes:
                # ---- PERTURBAÇÃO ----
                if pca_disponivel and var_nome in wgi_originais_nomes:
                    # Modo A: Projecção inversa PCA (indicador individual)
                    X_cf = perturbar_via_pca_inverso(X_base, feature_cols, var_nome, mag)
                    metodo = 'PCA_Inverso'
                else:
                    # Modo B: Perturbação directa (feature PCA composta)
                    X_cf = X_base.copy()
                    if var_nome in feature_cols:
                        var_idx = feature_cols.index(var_nome)
                        var_std = df[var_nome].std()
                        if var_std > 0:
                            X_cf[0, var_idx] = X_base[0, var_idx] + mag * var_std
                    metodo = 'Directa'
                
                # Previsão contrafactual
                try:
                    y_cf = float(model.predict(X_cf)[0])
                except Exception:
                    continue
                
                delta_abs = y_cf - y_base
                delta_pct = (delta_abs / (abs(y_base) + 1e-10)) * 100
                
                resultados_cf.append({
                    'Pais': pais,
                    'Pais_Nome': config.PAISES.get(pais, pais),
                    'Ano_Referencia': ano_ref,
                    'Indicador_WGI': var_nome,
                    'Indicador_Label': var_label,
                    'Magnitude_Std': mag,
                    'Previsao_Base': y_base,
                    'Previsao_Contrafactual': y_cf,
                    'Delta_Absoluto': delta_abs,
                    'Delta_Percentual': delta_pct,
                    'Y_Real': y_real,
                    'Modelo': model_nome,
                    'Metodo_Perturbacao': metodo,
                })
    
    # ================================================================
    # SHAP COMPARATIVO (antes vs. depois da "reforma")
    # ================================================================
    
    paises_shap = ['ZAF', 'NGA', 'TUR', 'MAR', 'MOZ', 'ETH', 'KEN', 'EGY']
    paises_shap = [p for p in paises_shap if p in df['country_code'].unique()]
    
    if usar_shap and len(paises_shap) > 0 and len(vars_perturbacao) > 0:
        print(f"  Calculando SHAP contrafactual para {len(paises_shap)} países...")
        
        try:
            import shap
            X_all = df[feature_cols].fillna(0).values
            
            if hasattr(model, 'estimators_'):  # Random Forest / Ensemble
                explainer = shap.TreeExplainer(model)
            else:
                background = shap.kmeans(X_all, min(10, len(X_all)))
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Usar o primeiro indicador WGI (rule_of_law se disponível)
            var_shap_nome, var_shap_label = vars_perturbacao[0]
            
            for pais in paises_shap:
                df_pais = df[df['country_code'] == pais].sort_values('year')
                if len(df_pais) == 0:
                    continue
                
                ultima_obs = df_pais.iloc[-1]
                X_base_shap = ultima_obs[feature_cols].values.astype(float).reshape(1, -1)
                X_base_shap = np.nan_to_num(X_base_shap, nan=0.0)
                
                n_feat_m = getattr(model, 'n_features_in_', None)
                if n_feat_m and X_base_shap.shape[1] != n_feat_m:
                    continue
                
                # SHAP base
                shap_base = explainer.shap_values(X_base_shap)
                if isinstance(shap_base, list):
                    shap_base = shap_base[0]
                
                # Perturbação +0.5 std
                if pca_disponivel and var_shap_nome in wgi_originais_nomes:
                    X_cf_shap = perturbar_via_pca_inverso(X_base_shap, feature_cols, var_shap_nome, 0.5)
                else:
                    X_cf_shap = X_base_shap.copy()
                    if var_shap_nome in feature_cols:
                        idx_s = feature_cols.index(var_shap_nome)
                        X_cf_shap[0, idx_s] += 0.5 * df[var_shap_nome].std()
                
                # SHAP contrafactual
                shap_cf = explainer.shap_values(X_cf_shap)
                if isinstance(shap_cf, list):
                    shap_cf = shap_cf[0]
                
                # Registar diferenças SHAP (top 10 features)
                shap_diff = shap_cf[0] - shap_base[0]
                top_features_idx = np.argsort(np.abs(shap_diff))[-10:]
                
                for fi in top_features_idx:
                    resultados_shap.append({
                        'Pais': pais,
                        'Pais_Nome': config.PAISES.get(pais, pais),
                        'Feature': feature_cols[fi],
                        'SHAP_Base': float(shap_base[0][fi]),
                        'SHAP_Contrafactual': float(shap_cf[0][fi]),
                        'SHAP_Delta': float(shap_diff[fi]),
                        'Indicador_Perturbado': var_shap_nome,
                        'Indicador_Label': var_shap_label,
                        'Magnitude': 0.5,
                        'Metodo': 'PCA_Inverso' if pca_disponivel else 'Directa',
                    })
        
        except Exception as e:
            print(f"  AVISO: Erro no cálculo SHAP contrafactual: {e}")
    
    # ================================================================
    # EXPORTAR RESULTADOS
    # ================================================================
    
    df_contrafactual = pd.DataFrame(resultados_cf)
    df_shap_comp = pd.DataFrame(resultados_shap)
    
    cf_path = os.path.join(INOVACOES_DIR, 'contrafactual_localizado.csv')
    shap_path = os.path.join(INOVACOES_DIR, 'contrafactual_shap_comparativo.csv')
    df_contrafactual.to_csv(cf_path, index=False)
    df_shap_comp.to_csv(shap_path, index=False)
    
    print(f"  Simulações contrafactuais: {len(df_contrafactual)} registos")
    print(f"  Comparações SHAP: {len(df_shap_comp)} registos")
    if pca_disponivel:
        print(f"  Método: Projecção Inversa PCA (indicadores WGI individuais)")
    else:
        print(f"  Método: Perturbação Directa (features PCA compostas)")
    
    # --- GRÁFICOS ---
    
    # Gráfico 4: Impacto contrafactual por país (para +0.5 std na 1a variável WGI)
    if len(df_contrafactual) > 0 and len(vars_perturbacao) > 0:
        var_plot_nome, var_plot_label = vars_perturbacao[0]
        df_plot = df_contrafactual[
            (df_contrafactual['Indicador_WGI'] == var_plot_nome) & 
            (df_contrafactual['Magnitude_Std'] == 0.5)
        ].copy()
        
        if len(df_plot) > 0:
            df_plot = df_plot.sort_values('Delta_Absoluto', ascending=True)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            cores = ['green' if d > 0 else 'red' for d in df_plot['Delta_Absoluto']]
            ax.barh(range(len(df_plot)), df_plot['Delta_Absoluto'], color=cores, alpha=0.7)
            ax.set_yticks(range(len(df_plot)))
            ax.set_yticklabels(df_plot['Pais_Nome'], fontsize=8)
            ax.set_xlabel('Impacto no VAI (pontos percentuais)', fontsize=11)
            ax.set_title(f'Simulação Contrafactual: Impacto de +0.5σ em "{var_plot_label}"\n'
                        f'(Ceteris Paribus via Projecção Inversa PCA — Modelo: {model_nome})',
                        fontsize=12, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(INOVACOES_DIR, 'contrafactual_impacto_pais.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Gráfico 5: Curva dose-resposta (impacto vs. magnitude da perturbação)
    if len(df_contrafactual) > 0 and len(vars_perturbacao) > 0:
        var_plot_nome, var_plot_label = vars_perturbacao[0]
        fig, ax = plt.subplots(figsize=(10, 7))
        
        paises_curva = ['ZAF', 'NGA', 'TUR', 'MAR', 'MOZ', 'ETH']
        paises_curva = [p for p in paises_curva if p in df_contrafactual['Pais'].unique()]
        
        for pais in paises_curva[:6]:
            df_p = df_contrafactual[
                (df_contrafactual['Pais'] == pais) & 
                (df_contrafactual['Indicador_WGI'] == var_plot_nome)
            ].sort_values('Magnitude_Std')
            
            if len(df_p) > 0:
                ax.plot(df_p['Magnitude_Std'], df_p['Previsao_Contrafactual'], 
                       '-o', markersize=5, label=config.PAISES.get(pais, pais))
        
        ax.set_xlabel('Magnitude da Perturbação (desvios-padrão)', fontsize=11)
        ax.set_ylabel('Previsão VAI (% PIB)', fontsize=11)
        ax.set_title(f'Curva Dose-Resposta: "{var_plot_label}" → VAI\n'
                    f'(Simulação Ceteris Paribus por País — Projecção Inversa PCA)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'contrafactual_dose_resposta.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Gráfico 6: SHAP Waterfall comparativo (antes vs. depois)
    if len(df_shap_comp) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Escolher um país exemplo
        pais_ex = paises_shap[0] if len(paises_shap) > 0 else df_shap_comp['Pais'].iloc[0]
        df_ex = df_shap_comp[df_shap_comp['Pais'] == pais_ex].sort_values('SHAP_Base', key=abs, ascending=False).head(10)
        
        if len(df_ex) > 0:
            # Painel esquerdo: SHAP Base
            ax = axes[0]
            features_plot = df_ex['Feature'].values
            shap_base_vals = df_ex['SHAP_Base'].values
            cores_b = ['green' if v > 0 else 'red' for v in shap_base_vals]
            ax.barh(range(len(features_plot)), shap_base_vals, color=cores_b, alpha=0.7)
            ax.set_yticks(range(len(features_plot)))
            ax.set_yticklabels([f[:25] for f in features_plot], fontsize=8)
            ax.set_xlabel('Valor SHAP')
            ax.set_title(f'{config.PAISES.get(pais_ex, pais_ex)} — ANTES da Reforma', fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            # Painel direito: SHAP Contrafactual
            ax = axes[1]
            shap_cf_vals = df_ex['SHAP_Contrafactual'].values
            cores_c = ['green' if v > 0 else 'red' for v in shap_cf_vals]
            ax.barh(range(len(features_plot)), shap_cf_vals, color=cores_c, alpha=0.7)
            ax.set_yticks(range(len(features_plot)))
            ax.set_yticklabels([f[:25] for f in features_plot], fontsize=8)
            ax.set_xlabel('Valor SHAP')
            ax.set_title(f'{config.PAISES.get(pais_ex, pais_ex)} — DEPOIS da Reforma (+0.5σ)', fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Comparação SHAP: Estrutura de Decisão Antes vs. Depois da Reforma Institucional',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'contrafactual_shap_waterfall.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Ficheiros exportados: {cf_path}")
    print(f"                        {shap_path}")
    print(f"  Gráficos: 3 PNG gerados")
    
    return df_contrafactual, df_shap_comp



# ============================================================
# SUGESTÃO 3: PREDIÇÃO CONFORMAL PARA TODOS OS MODELOS
# ============================================================

def executar_predicao_conformal(df):
    """
    Predição Conformal (Split Conformal Prediction) para todos os modelos.
    
    Fundamentação Teórica (Vovk et al., 2005; Romano et al., 2019):
    A predição conformal fornece intervalos de predição com garantia
    de cobertura marginal finita, sem assumir qualquer distribuição
    dos resíduos. A única hipótese é a permutabilidade (exchangeability)
    das observações.
    
    Algoritmo (Split Conformal):
    1. Dividir dados em treino (D_train) e calibração (D_cal).
    2. Treinar modelo f_hat em D_train.
    3. Calcular resíduos de conformidade em D_cal:
       s_i = |y_i - f_hat(x_i)|  para i in D_cal
    4. Calcular o quantil empírico:
       q_hat = Quantil(s_1, ..., s_n_cal; (1-alpha)*(1 + 1/n_cal))
    5. Intervalo de predição para nova observação x_new:
       C(x_new) = [f_hat(x_new) - q_hat, f_hat(x_new) + q_hat]
    
    Garantia teórica:
       P(Y_new in C(X_new)) >= 1 - alpha
    
    Esta garantia é EXACTA para amostras finitas e independente da
    distribuição dos dados (distribution-free).
    
    Extensão: Conformalized Quantile Regression (CQR)
    Para intervalos adaptativos (mais estreitos onde o modelo é confiante):
    - Treinar regressão quantílica para q_lower e q_upper
    - Calibrar os quantis com resíduos de conformidade
    
    Returns:
        tuple: (df_intervalos, df_cobertura, df_quartis) DataFrames
    """
    print("\n" + "="*60)
    print("  SUGESTÃO 3: PREDIÇÃO CONFORMAL PARA TODOS OS MODELOS")
    print("="*60)
    
    target = config.TARGET_VAR
    
    # Preparar dados
    colunas_excluir = ['country_code', 'year', target, 'pais', 'country_name']
    feature_cols = [c for c in df.columns if c not in colunas_excluir and df[c].dtype in ['float64', 'int64', 'float32']]
    
    if target not in df.columns or len(feature_cols) == 0:
        print("  ERRO: Dados insuficientes para predição conformal.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Split temporal (consistente com o pipeline)
    df_sorted = df.sort_values(['country_code', 'year']).copy()
    
    # Usar TRAIN_END_YEAR e VAL_END_YEAR do config
    train_mask = df_sorted['year'] <= config.TRAIN_END_YEAR
    cal_mask = (df_sorted['year'] > config.TRAIN_END_YEAR) & (df_sorted['year'] <= config.VAL_END_YEAR)
    test_mask = df_sorted['year'] > config.VAL_END_YEAR
    
    # Se teste vazio (dados não vão além de VAL_END_YEAR), usar split percentual
    if test_mask.sum() == 0:
        n_total = len(df_sorted)
        n_train = int(n_total * 0.60)
        n_cal = int(n_total * 0.20)
        # Recriar masks por posição
        train_mask = pd.Series([False]*n_total, index=df_sorted.index)
        cal_mask = pd.Series([False]*n_total, index=df_sorted.index)
        test_mask = pd.Series([False]*n_total, index=df_sorted.index)
        train_mask.iloc[:n_train] = True
        cal_mask.iloc[n_train:n_train+n_cal] = True
        test_mask.iloc[n_train+n_cal:] = True
        print(f"  Split percentual (60/20/20): Treino={n_train}, Cal={n_cal}, Teste={n_total-n_train-n_cal}")
    
    X_train = df_sorted.loc[train_mask, feature_cols].fillna(0).values
    y_train = df_sorted.loc[train_mask, target].fillna(0).values
    X_cal = df_sorted.loc[cal_mask, feature_cols].fillna(0).values
    y_cal = df_sorted.loc[cal_mask, target].fillna(0).values
    X_test = df_sorted.loc[test_mask, feature_cols].fillna(0).values
    y_test = df_sorted.loc[test_mask, target].fillna(0).values
    
    # Metadados do teste
    test_paises = df_sorted.loc[test_mask, 'country_code'].values if 'country_code' in df_sorted.columns else np.array([])
    test_anos = df_sorted.loc[test_mask, 'year'].values if 'year' in df_sorted.columns else np.array([])
    
    print(f"  Split: Treino={len(X_train)} | Calibração={len(X_cal)} | Teste={len(X_test)}")
    
    if len(X_cal) < 5:
        print("  AVISO: Conjunto de calibração muito pequeno. Usando validação cruzada conformal.")
        # Fallback: usar últimos 20% do treino como calibração
        n_cal = max(5, int(len(X_train) * 0.2))
        X_cal = X_train[-n_cal:]
        y_cal = y_train[-n_cal:]
        X_train = X_train[:-n_cal]
        y_train = y_train[:-n_cal]
        print(f"  Split ajustado: Treino={len(X_train)} | Calibração={len(X_cal)} | Teste={len(X_test)}")
    
    # Carregar todos os modelos disponíveis
    modelos = {}
    nomes_modelos = ['RandomForest', 'XGBoost', 'HistGBM', 'SARIMAX', 'LSTM',
                     'Bayes_PartialPooling', 'Bayes_CompletePooling']
    
    for nome in nomes_modelos:
        model = carregar_modelo(nome)
        if model is not None:
            modelos[nome] = model
    
    # Verificar compatibilidade de features — NÃO treina modelos localmente
    modelos_validos = {}
    modelos_incompativeis = []
    for nome, model in modelos.items():
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_train.shape[1]:
            modelos_incompativeis.append((nome, model.n_features_in_, X_train.shape[1]))
        else:
            modelos_validos[nome] = model
    
    # Reportar modelos incompatíveis
    if modelos_incompativeis:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  AVISO: MODELOS COM INCOMPATIBILIDADE DE FEATURES       ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        for nome, n_esp, n_real in modelos_incompativeis:
            print(f"  → {nome}: espera {n_esp} features, dados têm {n_real}")
        print(f"  SOLUÇÃO: Execute o Passo 3 (Engenharia de Features) e depois")
        print(f"           o Passo 4 (Treino) com o dataset 'agregado_features.csv'")
        print(f"           para garantir consistência entre features e modelos.")
    
    modelos = modelos_validos
    
    if len(modelos) == 0:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  ERRO: NENHUM MODELO COMPATÍVEL ENCONTRADO               ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        print(f"  O Passo 10 NÃO treina modelos. Requer modelos do Passo 4.")
        print(f"  ")
        print(f"  Ficheiros necessários (pelo menos 1):")
        for nome in nomes_modelos:
            path = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{nome}.pkl')
            existe = '✓' if os.path.exists(path) else '✗'
            print(f"    {existe} {path}")
        print(f"  ")
        print(f"  ACÇÃO: Execute primeiro o Passo 4 (Treino de Modelos):")
        print(f"    from passo4_treino_modelos import executar_passo4")
        print(f"    executar_passo4()")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"  Modelos compatíveis para conformal: {list(modelos.keys())}")
    print(f"  (Nota: O Passo 10 usa APENAS modelos treinados no Passo 4)")
    
    
    # Níveis de confiança para análise
    alphas = [0.05, 0.10, 0.20]  # 95%, 90%, 80%
    
    resultados_intervalos = []
    resultados_cobertura = []
    resultados_quartis = []
    
    for nome_modelo, model in modelos.items():
        print(f"\n  --- {nome_modelo} ---")
        
        # Previsões no conjunto de calibração
        try:
            y_cal_pred = model.predict(X_cal).ravel()
        except Exception as e:
            print(f"    ERRO na previsão (calibração): {e}")
            continue
        
        # Previsões no conjunto de teste
        try:
            y_test_pred = model.predict(X_test).ravel()
        except Exception as e:
            print(f"    ERRO na previsão (teste): {e}")
            continue
        
        # Resíduos de conformidade (nonconformity scores)
        residuos_cal = np.abs(y_cal - y_cal_pred)
        
        # Para cada nível de confiança
        for alpha in alphas:
            nivel_conf = 1 - alpha
            
            # Quantil empírico corrigido (finite-sample correction)
            n_cal = len(residuos_cal)
            quantil_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
            quantil_level = min(quantil_level, 1.0)
            q_hat = np.quantile(residuos_cal, quantil_level)
            
            # Intervalos de predição para o teste
            lower = y_test_pred - q_hat
            upper = y_test_pred + q_hat
            
            # Cobertura empírica
            dentro = ((y_test >= lower) & (y_test <= upper))
            cobertura_empirica = np.mean(dentro) * 100 if len(y_test) > 0 else np.nan
            
            # Largura média do intervalo
            largura_media = np.mean(upper - lower)
            
            # Registar cobertura
            resultados_cobertura.append({
                'Modelo': nome_modelo,
                'Nivel_Confianca': f"{int(nivel_conf*100)}%",
                'Alpha': alpha,
                'Q_Hat': q_hat,
                'Cobertura_Empirica_Pct': cobertura_empirica,
                'Cobertura_Nominal_Pct': nivel_conf * 100,
                'Largura_Media_Intervalo': largura_media,
                'N_Calibracao': n_cal,
                'N_Teste': len(y_test),
                'Calibrado': 'Sim' if abs(cobertura_empirica - nivel_conf*100) < 10 else 'Nao',
            })
            
            # Registar intervalos individuais (para alpha=0.10, i.e., 90%)
            if alpha == 0.10:
                for i in range(len(y_test)):
                    pais_i = test_paises[i] if i < len(test_paises) else ''
                    ano_i = int(test_anos[i]) if i < len(test_anos) else 0
                    
                    resultados_intervalos.append({
                        'Modelo': nome_modelo,
                        'Pais': pais_i,
                        'Pais_Nome': config.PAISES.get(pais_i, pais_i),
                        'Ano': ano_i,
                        'Y_Real': float(y_test[i]),
                        'Y_Pred': float(y_test_pred[i]),
                        'IC_Lower_90': float(lower[i]),
                        'IC_Upper_90': float(upper[i]),
                        'Largura_IC': float(upper[i] - lower[i]),
                        'Dentro_IC': bool(dentro[i]),
                        'Residuo_Abs': float(abs(y_test[i] - y_test_pred[i])),
                    })
        
        # Análise de Quartis (p05, p25, p50, p75, p95)
        # Usando bootstrap dos resíduos para gerar distribuição preditiva empírica
        n_bootstrap = 1000
        
        for i in range(len(y_test)):
            # Bootstrap dos resíduos de calibração
            boot_residuos = np.random.choice(residuos_cal, size=n_bootstrap, replace=True)
            # Sinais aleatórios (+/-)
            sinais = np.random.choice([-1, 1], size=n_bootstrap)
            # Distribuição preditiva empírica
            dist_pred = y_test_pred[i] + sinais * boot_residuos
            
            # Quartis
            p05 = np.percentile(dist_pred, 5)
            p25 = np.percentile(dist_pred, 25)
            p50 = np.percentile(dist_pred, 50)
            p75 = np.percentile(dist_pred, 75)
            p95 = np.percentile(dist_pred, 95)
            
            pais_i = test_paises[i] if i < len(test_paises) else ''
            ano_i = int(test_anos[i]) if i < len(test_anos) else 0
            
            resultados_quartis.append({
                'Modelo': nome_modelo,
                'Pais': pais_i,
                'Pais_Nome': config.PAISES.get(pais_i, pais_i),
                'Ano': ano_i,
                'Y_Real': float(y_test[i]),
                'Y_Pred_Pontual': float(y_test_pred[i]),
                'P05': float(p05),
                'P25': float(p25),
                'P50': float(p50),
                'P75': float(p75),
                'P95': float(p95),
                'IQR': float(p75 - p25),
                'Amplitude_90': float(p95 - p05),
            })
        
        print(f"    Cobertura 90%: {resultados_cobertura[-2]['Cobertura_Empirica_Pct']:.1f}% "
              f"(nominal: 90%) | Largura: {resultados_cobertura[-2]['Largura_Media_Intervalo']:.3f}")
    
    # Criar DataFrames
    df_intervalos = pd.DataFrame(resultados_intervalos)
    df_cobertura = pd.DataFrame(resultados_cobertura)
    df_quartis = pd.DataFrame(resultados_quartis)
    
    # Exportar CSVs
    int_path = os.path.join(INOVACOES_DIR, 'conformal_intervalos.csv')
    cob_path = os.path.join(INOVACOES_DIR, 'conformal_cobertura.csv')
    qrt_path = os.path.join(INOVACOES_DIR, 'conformal_quartis.csv')
    df_intervalos.to_csv(int_path, index=False)
    df_cobertura.to_csv(cob_path, index=False)
    df_quartis.to_csv(qrt_path, index=False)
    
    # --- GRÁFICOS ---
    
    # Gráfico 7: Comparação de cobertura entre modelos
    if len(df_cobertura) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 7a: Cobertura empírica vs. nominal
        ax = axes[0]
        df_cob_90 = df_cobertura[df_cobertura['Alpha'] == 0.10]
        if len(df_cob_90) > 0:
            x_pos = range(len(df_cob_90))
            bars = ax.bar(x_pos, df_cob_90['Cobertura_Empirica_Pct'], 
                         color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Nominal (90%)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_cob_90['Modelo'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Cobertura Empírica (%)')
            ax.set_title('Cobertura dos Intervalos Conformais (90%)\nvs. Garantia Nominal', fontweight='bold')
            ax.legend()
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)
        
        # 7b: Largura média dos intervalos
        ax = axes[1]
        if len(df_cob_90) > 0:
            bars = ax.bar(x_pos, df_cob_90['Largura_Media_Intervalo'],
                         color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_cob_90['Modelo'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Largura Média do Intervalo (pp)')
            ax.set_title('Eficiência dos Intervalos Conformais\n(Menor = Mais Preciso)', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'conformal_cobertura_comparacao.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Gráfico 8: Intervalos de predição para um país exemplo
    if len(df_intervalos) > 0:
        pais_ex = 'ZAF'
        if pais_ex not in df_intervalos['Pais'].values:
            pais_ex = df_intervalos['Pais'].iloc[0] if len(df_intervalos) > 0 else ''
        
        modelos_plot = df_intervalos['Modelo'].unique()[:3]
        
        if len(modelos_plot) > 0:
            fig, axes = plt.subplots(len(modelos_plot), 1, figsize=(12, 4*len(modelos_plot)))
            if len(modelos_plot) == 1:
                axes = [axes]
            
            for idx, modelo_p in enumerate(modelos_plot):
                ax = axes[idx]
                df_p = df_intervalos[
                    (df_intervalos['Modelo'] == modelo_p) & 
                    (df_intervalos['Pais'] == pais_ex)
                ].sort_values('Ano')
                
                if len(df_p) > 0:
                    anos_p = df_p['Ano'].values
                    ax.plot(anos_p, df_p['Y_Real'], 'ko-', markersize=6, label='Real', linewidth=2)
                    ax.plot(anos_p, df_p['Y_Pred'], 'b^-', markersize=5, label='Previsão', linewidth=1.5)
                    ax.fill_between(anos_p, df_p['IC_Lower_90'], df_p['IC_Upper_90'],
                                   alpha=0.3, color='steelblue', label='IC 90% (Conformal)')
                    ax.set_title(f'{modelo_p} — {config.PAISES.get(pais_ex, pais_ex)}', fontweight='bold')
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('VAI (% PIB)')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(alpha=0.3)
            
            plt.suptitle('Intervalos de Predição Conformal (90%) — Comparação entre Modelos',
                        fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(INOVACOES_DIR, 'conformal_intervalos_exemplo.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Gráfico 9: Análise de quartis (fan chart)
    if len(df_quartis) > 0:
        pais_ex = 'ZAF'
        if pais_ex not in df_quartis['Pais'].values:
            pais_ex = df_quartis['Pais'].iloc[0]
        
        modelo_ex = df_quartis['Modelo'].iloc[0]
        df_q = df_quartis[
            (df_quartis['Modelo'] == modelo_ex) & 
            (df_quartis['Pais'] == pais_ex)
        ].sort_values('Ano')
        
        if len(df_q) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            anos_q = df_q['Ano'].values
            
            # Fan chart com camadas de incerteza
            ax.fill_between(anos_q, df_q['P05'], df_q['P95'], alpha=0.15, color='blue', label='P05-P95')
            ax.fill_between(anos_q, df_q['P25'], df_q['P75'], alpha=0.3, color='blue', label='P25-P75 (IQR)')
            ax.plot(anos_q, df_q['P50'], 'b--', linewidth=1.5, label='Mediana (P50)')
            ax.plot(anos_q, df_q['Y_Pred_Pontual'], 'b-', linewidth=2, label='Previsão Pontual')
            ax.plot(anos_q, df_q['Y_Real'], 'ko-', markersize=7, linewidth=2, label='Real')
            
            ax.set_xlabel('Ano', fontsize=11)
            ax.set_ylabel('VAI (% PIB)', fontsize=11)
            ax.set_title(f'Fan Chart de Incerteza Preditiva — {config.PAISES.get(pais_ex, pais_ex)}\n'
                        f'(Modelo: {modelo_ex} | Predição Conformal com Bootstrap)',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(INOVACOES_DIR, 'conformal_fan_chart.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Gráfico 10: Largura do intervalo por país (heterogeneidade da incerteza)
    if len(df_intervalos) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Largura média por país (primeiro modelo disponível)
        modelo_1 = df_intervalos['Modelo'].iloc[0]
        df_larg = df_intervalos[df_intervalos['Modelo'] == modelo_1].groupby('Pais_Nome')['Largura_IC'].mean()
        df_larg = df_larg.sort_values(ascending=True)
        
        if len(df_larg) > 0:
            cores_larg = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df_larg)))
            ax.barh(range(len(df_larg)), df_larg.values, color=cores_larg, alpha=0.8)
            ax.set_yticks(range(len(df_larg)))
            ax.set_yticklabels(df_larg.index, fontsize=8)
            ax.set_xlabel('Largura Média do IC 90% (pontos percentuais)', fontsize=11)
            ax.set_title(f'Heterogeneidade da Incerteza Preditiva por País\n'
                        f'(Modelo: {modelo_1} | Predição Conformal)',
                        fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(INOVACOES_DIR, 'conformal_incerteza_por_pais.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n  Ficheiros exportados: {int_path}")
    print(f"                        {cob_path}")
    print(f"                        {qrt_path}")
    print(f"  Gráficos: 4 PNG gerados")
    
    return df_intervalos, df_cobertura, df_quartis



# ============================================================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ============================================================

def executar_passo10():
    """
    Executa as 3 inovações metodológicas sequencialmente.
    
    PRÉ-REQUISITOS (o Passo 10 NÃO treina modelos):
    ================================================
    
    FICHEIROS DE DADOS (pelo menos 1 obrigatório):
      1.º dados_engenharia/agregado_features.csv  ← Passo 3 (PREFERENCIAL)
      2.º dados_agregados/agregado_inner_join.csv  ← Passo 2.2 (FALLBACK)
    
    MODELOS TREINADOS (necessários para Sugestões 2 e 3):
      modelos/modelo_Agregado_RandomForest.pkl    ← Passo 4
      modelos/modelo_Agregado_XGBoost.pkl         ← Passo 4
      modelos/modelo_Agregado_HistGBM.pkl         ← Passo 4
      modelos/modelo_Agregado_LSTM.pkl            ← Passo 4 (opcional)
      modelos/modelo_Agregado_SARIMAX.pkl         ← Passo 4 (opcional)
      modelos/modelo_Agregado_Bayes_*.pkl         ← Passo 4 (opcional)
    
    CRITÉRIOS POR SUGESTÃO:
    
    Sugestão 1 (Ruptures): Apenas dados. Sem modelos.
      → Entrada: dataset com [country_code, year, TARGET]
      → Mínimo: 8 obs por país, 3+ países
    
    Sugestão 2 (Contrafactual): Dados + 1 modelo + colunas WGI.
      → Entrada: dataset com colunas wgi_* + modelo .pkl
      → O modelo deve ser compatível com as features do dataset
    
    Sugestão 3 (Conformal): Dados + 1+ modelos compatíveis.
      → Entrada: dataset + modelos .pkl com n_features_in_ correcto
      → Mínimo: 5 obs no conjunto de calibração
    
    Ordem de execução:
    1. Detecção de Quebras Estruturais (Ruptures/PELT)
    2. Simulação Contrafactual Localizada (Ceteris Paribus + SHAP)
    3. Predição Conformal (Split Conformal + Quartis)
    
    Saídas: 7 CSV + 10 PNG + 1 JSON em inovacoes_mestrado/
    """
    t_inicio = time.time()
    
    print("\n" + "="*70)
    print("  PASSO 10 — INOVAÇÕES METODOLÓGICAS (Nível Mestrado)")
    print("  Pipeline de Análise Industrial — África e Médio Oriente")
    print("="*70)
    print(f"  Directório de saída: {INOVACOES_DIR}")
    
    # Verificar modelos disponíveis com detalhe
    print(f"\n  Modelos disponíveis no Passo 4:")
    modelos_encontrados = 0
    nomes_modelos_check = ['RandomForest', 'XGBoost', 'HistGBM', 'SARIMAX', 'LSTM',
                           'Bayes_PartialPooling', 'Bayes_CompletePooling']
    for nome in nomes_modelos_check:
        path = os.path.join(config.MODELOS_DIR, f'modelo_Agregado_{nome}.pkl')
        if os.path.exists(path):
            tamanho = os.path.getsize(path) / 1024
            print(f"    ✓ {nome} ({tamanho:.0f} KB)")
            modelos_encontrados += 1
        else:
            print(f"    ✗ {nome} — não encontrado")
    
    if modelos_encontrados == 0:
        print(f"\n  ⚠ NENHUM MODELO ENCONTRADO. As Sugestões 2 e 3 não funcionarão.")
        print(f"    Execute primeiro: from passo4_treino_modelos import executar_passo4; executar_passo4()")
    
    # Carregar dados
    print("\n  [0/3] Carregando dados...")
    df = carregar_dados_agregados()
    
    if df is None or len(df) == 0:
        print("  ERRO FATAL: Não foi possível carregar os dados agregados.")
        print("  Certifique-se de que os Passos 1-3 foram executados com sucesso.")
        return
    
    print(f"  Dataset: {df.shape[0]} observações × {df.shape[1]} variáveis")
    print(f"  Países: {df['country_code'].nunique() if 'country_code' in df.columns else '?'}")
    print(f"  Período: {df['year'].min()}-{df['year'].max() if 'year' in df.columns else '?'}")
    
    # ============================================================
    # SUGESTÃO 1: QUEBRAS ESTRUTURAIS
    # ============================================================
    print("\n  [1/3] Executando Sugestão 1: Quebras Estruturais...")
    try:
        df_quebras, df_regimes = executar_quebras_ruptures(df)
        print(f"  ✓ Sugestão 1 concluída: {len(df_quebras)} quebras, {len(df_regimes)} registos de regime")
    except Exception as e:
        print(f"  ✗ ERRO na Sugestão 1: {e}")
        import traceback
        traceback.print_exc()
        df_quebras, df_regimes = pd.DataFrame(), pd.DataFrame()
    
    # ============================================================
    # SUGESTÃO 2: SIMULAÇÃO CONTRAFACTUAL
    # ============================================================
    print("\n  [2/3] Executando Sugestão 2: Simulação Contrafactual...")
    try:
        df_contrafactual, df_shap_comp = executar_contrafactual_localizado(df)
        print(f"  ✓ Sugestão 2 concluída: {len(df_contrafactual)} simulações, {len(df_shap_comp)} comparações SHAP")
    except Exception as e:
        print(f"  ✗ ERRO na Sugestão 2: {e}")
        import traceback
        traceback.print_exc()
        df_contrafactual, df_shap_comp = pd.DataFrame(), pd.DataFrame()
    
    # ============================================================
    # SUGESTÃO 3: PREDIÇÃO CONFORMAL
    # ============================================================
    print("\n  [3/3] Executando Sugestão 3: Predição Conformal...")
    try:
        df_intervalos, df_cobertura, df_quartis = executar_predicao_conformal(df)
        print(f"  ✓ Sugestão 3 concluída: {len(df_intervalos)} intervalos, {len(df_cobertura)} coberturas, {len(df_quartis)} quartis")
    except Exception as e:
        print(f"  ✗ ERRO na Sugestão 3: {e}")
        import traceback
        traceback.print_exc()
        df_intervalos, df_cobertura, df_quartis = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # ============================================================
    # RESUMO E METADADOS
    # ============================================================
    t_total = time.time() - t_inicio
    
    # Listar ficheiros gerados
    ficheiros_gerados = []
    for f in os.listdir(INOVACOES_DIR):
        ficheiros_gerados.append(os.path.join(INOVACOES_DIR, f))
    
    n_csv = len([f for f in ficheiros_gerados if f.endswith('.csv')])
    n_png = len([f for f in ficheiros_gerados if f.endswith('.png')])
    
    # Metadados
    metadados = {
        'passo': 'passo10_inovacoes_mestrado',
        'descricao': 'Inovações metodológicas: Ruptures, Contrafactual SHAP, Predição Conformal',
        'tempo_execucao_s': round(t_total, 1),
        'ficheiros_csv': n_csv,
        'ficheiros_png': n_png,
        'sugestao_1': {
            'nome': 'Quebras Estruturais (Ruptures/PELT)',
            'n_quebras': len(df_quebras),
            'n_regimes': len(df_regimes),
            'status': 'OK' if len(df_quebras) > 0 else 'SEM_DADOS',
        },
        'sugestao_2': {
            'nome': 'Simulação Contrafactual Localizada + SHAP',
            'n_simulacoes': len(df_contrafactual),
            'n_shap_comparacoes': len(df_shap_comp),
            'status': 'OK' if len(df_contrafactual) > 0 else 'SEM_DADOS',
        },
        'sugestao_3': {
            'nome': 'Predição Conformal (Split Conformal + Quartis)',
            'n_intervalos': len(df_intervalos),
            'n_modelos_avaliados': df_cobertura['Modelo'].nunique() if len(df_cobertura) > 0 else 0,
            'status': 'OK' if len(df_intervalos) > 0 else 'SEM_DADOS',
        },
    }
    
    # Salvar metadados
    import json
    meta_path = os.path.join(INOVACOES_DIR, 'metadados_passo10.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    
    # Auto-save para Drive
    auto_save_drive(ficheiros_gerados, config)
    
    # Resumo final
    print("\n" + "="*70)
    print("  RESUMO PASSO 10 — INOVAÇÕES METODOLÓGICAS")
    print("="*70)
    print(f"  Tempo total: {t_total:.1f}s")
    print(f"  Ficheiros gerados: {n_csv} CSV + {n_png} PNG")
    print(f"  Directório: {INOVACOES_DIR}")
    print(f"\n  Sugestão 1 (Ruptures):      {metadados['sugestao_1']['status']} — {len(df_quebras)} quebras detectadas")
    print(f"  Sugestão 2 (Contrafactual): {metadados['sugestao_2']['status']} — {len(df_contrafactual)} simulações")
    print(f"  Sugestão 3 (Conformal):     {metadados['sugestao_3']['status']} — {len(df_intervalos)} intervalos")
    print("\n  ✓ PASSO 10 CONCLUÍDO COM SUCESSO")
    print("="*70)


# ============================================================
# EXECUÇÃO DIRECTA
# ============================================================

if __name__ == '__main__':
    executar_passo10()
