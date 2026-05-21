"""
============================================================
PASSO 7: INTERPRETABILIDADE - SHAP E FEATURE IMPORTANCE
============================================================
Analises Completas:
  1. SHAP TreeExplainer (RF, XGBoost, GB) para cada dataset
  2. Permutation Importance (todos os modelos)
  3. Summary Plot (top 20 features)
  4. Dependence Plots (top 5 features)
  5. WGI vs WDI: contribuicao relativa
  6. Force Plots (top/bottom observacoes)
  7. SHAP Interaction Effects
  8. Heatmap importancia por dataset
============================================================
"""
import os, sys, pickle, time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_global as config
from metadata_generator import gerar_metadados, auto_save_drive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def executar_passo7():
    print("\n" + "=" * 70)
    print("  PASSO 7: INTERPRETABILIDADE (SHAP + FEATURE IMPORTANCE)")
    print("=" * 70)
    os.makedirs(config.SHAP_DIR, exist_ok=True)
    t_inicio = time.time()

    # CARREGAR DADOS E MODELOS
    print("\n  [1/8] Carregando modelos e dados...")
    datasets = {}
    for ds_nome in ['WDI_Limpo', 'Agregado', 'Sintetico_Agregado', 'WDI_Sintetico']:
        for fname in [f'{ds_nome.lower()}_features.csv', f'{ds_nome}_features.csv']:
            fpath = os.path.join(config.DADOS_ENGENHARIA_DIR, fname)
            if os.path.exists(fpath):
                datasets[ds_nome] = pd.read_csv(fpath)
                break
    if not datasets:
        print("  ERRO: Nenhum dataset encontrado.")
        return
    target = config.TARGET_VAR
    tree_models = ['RandomForest', 'XGBoost', 'GradientBoosting']
    modelos_carregados = {}
    for ds_nome in datasets.keys():
        for modelo_nome in tree_models:
            path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    modelos_carregados[f'{ds_nome}_{modelo_nome}'] = pickle.load(f)
    print(f"    Datasets: {len(datasets)} | Modelos: {len(modelos_carregados)}")

    # SHAP VALUES
    print("\n  [2/8] Calculando SHAP values...")
    try:
        import shap
        shap_disponivel = True
    except ImportError:
        shap_disponivel = False
        print("    AVISO: SHAP nao disponivel. Usando Permutation Importance.")
    shap_resultados = {}
    for key, modelo in modelos_carregados.items():
        ds_nome = None
        for d in datasets.keys():
            if key.startswith(d):
                ds_nome = d
                break
        if ds_nome is None or ds_nome not in datasets:
            continue
        df_ds = datasets[ds_nome]
        if target not in df_ds.columns:
            continue
        X = df_ds.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        n_sample = min(500, len(X))
        X_sample = X.sample(n=n_sample, random_state=42) if len(X) > n_sample else X
        if shap_disponivel:
            try:
                explainer = shap.TreeExplainer(modelo)
                shap_values = explainer.shap_values(X_sample)
                shap_importance = pd.DataFrame({'Feature': X_sample.columns, 'SHAP_Mean_Abs': np.abs(shap_values).mean(axis=0)}).sort_values('SHAP_Mean_Abs', ascending=False)
                shap_resultados[key] = {'shap_values': shap_values, 'X_sample': X_sample, 'importance': shap_importance}
                print(f"    SHAP: {key} ({n_sample} obs)")
            except Exception as e:
                print(f"    SHAP falhou: {key} - {str(e)[:40]}")
        if key not in shap_resultados:
            from sklearn.inspection import permutation_importance
            try:
                y_s = df_ds.loc[X_sample.index, target].values
                perm = permutation_importance(modelo, X_sample, y_s, n_repeats=10, random_state=42)
                shap_importance = pd.DataFrame({'Feature': X_sample.columns, 'SHAP_Mean_Abs': perm.importances_mean}).sort_values('SHAP_Mean_Abs', ascending=False)
                shap_resultados[key] = {'shap_values': None, 'X_sample': X_sample, 'importance': shap_importance}
                print(f"    Perm Importance: {key}")
            except Exception:
                pass

    # PERMUTATION IMPORTANCE COMPLETA
    print("\n  [3/8] Permutation Importance completa...")
    from sklearn.inspection import permutation_importance as perm_imp
    perm_resultados = []
    for ds_nome, df_ds in datasets.items():
        if target not in df_ds.columns:
            continue
        X = df_ds.drop(columns=['country_code', 'year', target, 'pais'], errors='ignore').fillna(0)
        y = df_ds[target].values
        n_s = min(300, len(X))
        idx_s = np.random.RandomState(42).choice(len(X), n_s, replace=False)
        X_s, y_s = X.iloc[idx_s], y[idx_s]
        for modelo_nome in tree_models:
            path = os.path.join(config.MODELOS_DIR, f'modelo_{ds_nome}_{modelo_nome}.pkl')
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                modelo = pickle.load(f)
            try:
                perm = perm_imp(modelo, X_s, y_s, n_repeats=10, random_state=42)
                for j, feat in enumerate(X.columns):
                    perm_resultados.append({'Dataset': ds_nome, 'Modelo': modelo_nome, 'Feature': feat, 'Importance_Mean': perm.importances_mean[j], 'Importance_Std': perm.importances_std[j], 'Tipo': 'WGI' if 'wgi' in feat.lower() else 'WDI'})
            except Exception:
                pass
    df_perm = pd.DataFrame(perm_resultados)
    perm_path = os.path.join(config.SHAP_DIR, 'permutation_importance_completa.csv')
    df_perm.to_csv(perm_path, index=False)

    # CONTRIBUICAO WGI vs WDI
    print("\n  [4/8] Contribuicao relativa WGI vs WDI...")
    contribuicao = []
    for key, resultado in shap_resultados.items():
        imp = resultado['importance']
        wgi_imp = imp[imp['Feature'].str.contains('wgi', case=False)]['SHAP_Mean_Abs'].sum()
        wdi_imp = imp[~imp['Feature'].str.contains('wgi', case=False)]['SHAP_Mean_Abs'].sum()
        total = wgi_imp + wdi_imp + 1e-10
        contribuicao.append({'Modelo_Dataset': key, 'WGI_Abs': wgi_imp, 'WDI_Abs': wdi_imp, 'WGI_Pct': wgi_imp/total*100, 'WDI_Pct': wdi_imp/total*100, 'WGI_Top5': ', '.join(imp[imp['Feature'].str.contains('wgi', case=False)].head(5)['Feature'].tolist())})
    df_contrib = pd.DataFrame(contribuicao)
    contrib_path = os.path.join(config.SHAP_DIR, 'contribuicao_wgi_vs_wdi.csv')
    df_contrib.to_csv(contrib_path, index=False)
    if len(df_contrib) > 0:
        print(f"    WGI media: {df_contrib['WGI_Pct'].mean():.1f}% | WDI media: {df_contrib['WDI_Pct'].mean():.1f}%")

    # TOP 20 FEATURES
    print("\n  [5/8] Tabela top 20 features...")
    top_all = []
    for key, resultado in shap_resultados.items():
        imp = resultado['importance'].head(20).copy()
        imp['Modelo_Dataset'] = key
        imp['Rank'] = range(1, len(imp)+1)
        imp['Tipo'] = imp['Feature'].apply(lambda x: 'WGI' if 'wgi' in x.lower() else 'WDI')
        top_all.append(imp)
    if top_all:
        df_top = pd.concat(top_all, ignore_index=True)
        top_path = os.path.join(config.SHAP_DIR, 'top20_features_por_modelo.csv')
        df_top.to_csv(top_path, index=False)

    # VISUALIZACOES
    print("\n  [6/8] Gerando visualizacoes...")
    best_key = None
    for key in shap_resultados:
        if 'Agregado' in key and 'RandomForest' in key:
            best_key = key
            break
    if best_key is None and shap_resultados:
        best_key = list(shap_resultados.keys())[0]

    # GRAFICO 1: SHAP Summary beeswarm
    if best_key and shap_resultados[best_key]['shap_values'] is not None:
        resultado = shap_resultados[best_key]
        sv = resultado['shap_values']
        Xs = resultado['X_sample']
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-20:][::-1]
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, fi in enumerate(top_idx):
            y_pos = len(top_idx) - 1 - i
            vals = sv[:, fi]
            fv = Xs.iloc[:, fi].values
            vmin, vmax = np.percentile(fv, [5, 95])
            norm_fv = (fv - vmin) / (vmax - vmin + 1e-10)
            colors = plt.cm.RdBu_r(norm_fv)
            ax.scatter(vals, np.full_like(vals, y_pos) + np.random.normal(0, 0.1, len(vals)), c=colors, s=5, alpha=0.5)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([Xs.columns[i][:35] for i in top_idx][::-1], fontsize=8)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'SHAP Summary - {best_key} (Top 20)', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 2: Bar comparacao datasets
    if shap_resultados:
        n_plots = min(4, len(shap_resultados))
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 8))
        if n_plots == 1:
            axes = [axes]
        for i, (key, resultado) in enumerate(list(shap_resultados.items())[:n_plots]):
            ax = axes[i]
            imp = resultado['importance'].head(15)
            colors = ['red' if 'wgi' in f.lower() else 'steelblue' for f in imp['Feature']]
            ax.barh(range(len(imp)), imp['SHAP_Mean_Abs'].values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels([f[:25] for f in imp['Feature']], fontsize=7)
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title(key[:25], fontsize=9, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        plt.suptitle('Feature Importance (Vermelho=WGI, Azul=WDI)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'shap_bar_comparacao.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 3: Dependence Plots top 5
    if best_key and shap_resultados[best_key]['shap_values'] is not None:
        resultado = shap_resultados[best_key]
        sv = resultado['shap_values']
        Xs = resultado['X_sample']
        imp = resultado['importance']
        top5 = imp.head(5)['Feature'].tolist()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes_flat = axes.flatten()
        for i, feat in enumerate(top5):
            ax = axes_flat[i]
            if feat in Xs.columns:
                fi = list(Xs.columns).index(feat)
                ax.scatter(Xs[feat], sv[:, fi], alpha=0.4, s=10, color='steelblue')
                ax.set_xlabel(feat[:30], fontsize=8)
                ax.set_ylabel('SHAP value')
                ax.set_title(f'{feat[:25]}', fontsize=9, fontweight='bold')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.grid(alpha=0.3)
                try:
                    z = np.polyfit(Xs[feat].values, sv[:, fi], 2)
                    xl = np.linspace(Xs[feat].min(), Xs[feat].max(), 50)
                    ax.plot(xl, np.polyval(z, xl), 'r-', linewidth=2, alpha=0.7)
                except Exception:
                    pass
        axes_flat[5].set_visible(False)
        plt.suptitle(f'SHAP Dependence - {best_key} (Top 5)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'shap_dependence_top5.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 4: Permutation Importance top 20
    if len(df_perm) > 0:
        perm_media = df_perm.groupby('Feature')['Importance_Mean'].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if 'wgi' in f.lower() else 'steelblue' for f in perm_media.index]
        ax.barh(range(len(perm_media)), perm_media.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(perm_media)))
        ax.set_yticklabels([f[:35] for f in perm_media.index], fontsize=8)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Top 20 Features (Vermelho=WGI, Azul=WDI)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'permutation_importance_top20.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 5: WGI vs WDI pie + bar
    if len(df_contrib) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        wgi_m = df_contrib['WGI_Pct'].mean()
        wdi_m = df_contrib['WDI_Pct'].mean()
        ax.pie([wgi_m, wdi_m], labels=['WGI (Governanca)', 'WDI (Economico)'], autopct='%1.1f%%', colors=['red', 'steelblue'], startangle=90, textprops={'fontsize': 11})
        ax.set_title('Contribuicao Relativa Media', fontweight='bold')
        ax = axes[1]
        x = range(len(df_contrib))
        w = 0.35
        ax.bar([i-w/2 for i in x], df_contrib['WGI_Pct'], w, label='WGI', color='red', alpha=0.7)
        ax.bar([i+w/2 for i in x], df_contrib['WDI_Pct'], w, label='WDI', color='steelblue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([k[:18] for k in df_contrib['Modelo_Dataset']], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('%')
        ax.set_title('WGI vs WDI por Modelo', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.suptitle('Contribuicao Governanca vs Economica', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'contribuicao_wgi_vs_wdi.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 6: Force Plots
    if best_key and shap_resultados[best_key]['shap_values'] is not None:
        resultado = shap_resultados[best_key]
        sv = resultado['shap_values']
        Xs = resultado['X_sample']
        shap_sum = sv.sum(axis=1)
        top3 = np.argsort(shap_sum)[-3:]
        bot3 = np.argsort(shap_sum)[:3]
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        for i, idx in enumerate(top3):
            ax = axes[0, i]
            fs = pd.Series(sv[idx], index=Xs.columns).sort_values()
            top_f = pd.concat([fs.head(5), fs.tail(5)])
            cols = ['red' if v < 0 else 'green' for v in top_f.values]
            ax.barh(range(len(top_f)), top_f.values, color=cols, alpha=0.7)
            ax.set_yticks(range(len(top_f)))
            ax.set_yticklabels([f[:20] for f in top_f.index], fontsize=7)
            ax.set_title(f'Top #{i+1} (sum={shap_sum[idx]:.2f})', fontsize=9)
            ax.axvline(x=0, color='gray', linestyle='--')
        for i, idx in enumerate(bot3):
            ax = axes[1, i]
            fs = pd.Series(sv[idx], index=Xs.columns).sort_values()
            top_f = pd.concat([fs.head(5), fs.tail(5)])
            cols = ['red' if v < 0 else 'green' for v in top_f.values]
            ax.barh(range(len(top_f)), top_f.values, color=cols, alpha=0.7)
            ax.set_yticks(range(len(top_f)))
            ax.set_yticklabels([f[:20] for f in top_f.index], fontsize=7)
            ax.set_title(f'Bottom #{i+1} (sum={shap_sum[idx]:.2f})', fontsize=9)
            ax.axvline(x=0, color='gray', linestyle='--')
        plt.suptitle('Force Plots: Top 3 vs Bottom 3', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'force_plots.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # GRAFICO 7: Interaction
    if best_key and shap_resultados[best_key]['shap_values'] is not None:
        resultado = shap_resultados[best_key]
        sv = resultado['shap_values']
        Xs = resultado['X_sample']
        imp = resultado['importance']
        top2 = imp.head(2)['Feature'].tolist()
        if len(top2) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for i, feat in enumerate(top2):
                ax = axes[i]
                fi = list(Xs.columns).index(feat) if feat in Xs.columns else None
                other = top2[1-i]
                if fi is not None and other in Xs.columns:
                    sc = ax.scatter(Xs[feat], sv[:, fi], c=Xs[other], cmap='RdBu_r', alpha=0.5, s=15)
                    ax.set_xlabel(feat[:30])
                    ax.set_ylabel(f'SHAP({feat[:15]})')
                    ax.set_title(f'Interacao (cor={other[:15]})', fontsize=9, fontweight='bold')
                    plt.colorbar(sc, ax=ax, label=other[:20])
                    ax.grid(alpha=0.3)
            plt.suptitle('SHAP Interaction Effects', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(config.SHAP_DIR, 'shap_interaction.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # GRAFICO 8: Heatmap por dataset
    if len(df_perm) > 0:
        top_feats = df_perm.groupby('Feature')['Importance_Mean'].mean().sort_values(ascending=False).head(15).index
        pivot_data = []
        for ds in df_perm['Dataset'].unique():
            ds_data = df_perm[df_perm['Dataset'] == ds].groupby('Feature')['Importance_Mean'].mean()
            for feat in top_feats:
                if feat in ds_data.index:
                    pivot_data.append({'Feature': feat, 'Dataset': ds, 'Importance': ds_data[feat]})
        if pivot_data:
            df_pv = pd.DataFrame(pivot_data).pivot(index='Feature', columns='Dataset', values='Importance')
            df_pv = df_pv.reindex(top_feats)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_pv, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, linewidths=0.5)
            ax.set_title('Importancia por Dataset (Top 15)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(config.SHAP_DIR, 'heatmap_importancia_datasets.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # METADADOS
    ficheiros_saida = [perm_path, contrib_path]
    gerar_metadados(passo='passo7_shap', descricao='SHAP TreeExplainer, Permutation Importance, Dependence, Force Plots, Interaction', config=config, dados_entrada=list(datasets.keys()), dados_saida=ficheiros_saida, parametros={'n_modelos': len(shap_resultados), 'shap_disponivel': shap_disponivel}, metricas={'wgi_pct_media': df_contrib['WGI_Pct'].mean() if len(df_contrib) > 0 else 0})
    auto_save_drive(ficheiros_saida, config)
    t_total = time.time() - t_inicio
    print(f"\n  RESUMO PASSO 7: {len(shap_resultados)} modelos | 8 graficos | {t_total:.1f}s")
    print("  OK PASSO 7 CONCLUIDO")


if __name__ == '__main__':
    executar_passo7()
