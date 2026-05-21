"""
============================================================
PIPELINE COMPLETO: ANÁLISE INDUSTRIAL - ÁFRICA E MÉDIO ORIENTE
============================================================
Passos 1 a 9 com geração automática de metadados em cada passo.

Para executar no Google Colab:
1. Fazer upload do ZIP para o Google Drive
2. Montar o Drive e extrair o ZIP
3. Executar este ficheiro ou o notebook .ipynb

Requisitos:
    pip install wbgapi pmdarima xgboost shap geopandas pymc arviz tensorflow
============================================================
"""
import os
import sys
import time

# Configurar path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

print("=" * 70)
print("  PIPELINE COMPLETO: ANÁLISE INDUSTRIAL - ÁFRICA E MÉDIO ORIENTE")
print("  Passos 1 a 9 com geração automática de metadados")
print("=" * 70)

# ============================================================
# PASSO 1: EXTRAÇÃO DE DADOS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 1 " + "▶" * 30)
t0 = time.time()
from passo1_extracao import executar_passo1
executar_passo1()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 2: EDA DOS DADOS BRUTOS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 2 " + "▶" * 30)
t0 = time.time()
from passo2_eda_brutos import executar_passo2
executar_passo2()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 2.1: LIMPEZA
# ============================================================
print("\n" + "▶" * 30 + " PASSO 2.1 " + "▶" * 30)
t0 = time.time()
from passo2_1_limpeza import executar_passo2_1
executar_passo2_1()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 2.2: AGREGAÇÃO + SINTÉTICOS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 2.2 " + "▶" * 30)
t0 = time.time()
from passo2_2_agregacao_sinteticos import executar_passo2_2
executar_passo2_2()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 2.3: EDA AGREGADOS + SINTÉTICOS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 2.3 " + "▶" * 30)
t0 = time.time()
from passo2_3_eda_agregados import executar_passo2_3
executar_passo2_3()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 3: ENGENHARIA DE FEATURES
# ============================================================
print("\n" + "▶" * 30 + " PASSO 3 " + "▶" * 30)
t0 = time.time()
from passo3_engenharia_features import executar_passo3
executar_passo3()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 4: TREINAMENTO DE MODELOS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 4 " + "▶" * 30)
t0 = time.time()
from passo4_treino_modelos import executar_passo4
executar_passo4()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 5: AVALIAÇÃO
# ============================================================
print("\n" + "▶" * 30 + " PASSO 5 " + "▶" * 30)
t0 = time.time()
from passo5_avaliacao import executar_passo5
executar_passo5()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 6: ESTRATÉGIAS
# ============================================================
print("\n" + "▶" * 30 + " PASSO 6 " + "▶" * 30)
t0 = time.time()
from passo6_estrategias import executar_passo6
executar_passo6()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 7: SHAP
# ============================================================
print("\n" + "▶" * 30 + " PASSO 7 " + "▶" * 30)
t0 = time.time()
from passo7_shap import executar_passo7
executar_passo7()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 8: GEOGRÁFICA
# ============================================================
print("\n" + "▶" * 30 + " PASSO 8 " + "▶" * 30)
t0 = time.time()
from passo8_geografica import executar_passo8
executar_passo8()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# PASSO 9: AVANÇADA
# ============================================================
print("\n" + "▶" * 30 + " PASSO 9 " + "▶" * 30)
t0 = time.time()
from passo9_avancada import executar_passo9
executar_passo9()
print(f"  ⏱ Tempo: {time.time()-t0:.1f}s")

# ============================================================
# CONCLUSÃO
# ============================================================
print("\n" + "=" * 70)
print("  ✓ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
print("=" * 70)
print("\n  Ficheiros gerados:")
for d in os.listdir(SCRIPT_DIR):
    full = os.path.join(SCRIPT_DIR, d)
    if os.path.isdir(full) and not d.startswith('.'):
        n_files = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
        if n_files > 0:
            print(f"    📁 {d}/  ({n_files} ficheiros)")
