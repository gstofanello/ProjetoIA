# Análise Exploratória de Dados (EDA)
# ============================
# Gabriel Sanches Tofanello RA: 10410281
# Hyandra Marjory RA:
# Felipe Damasceno RA: 
# Rodrigo Pampolin RA:
# ============================
# O arquivo 'base_vendas_pdv_IA.xlsx' foi gerado a partir do script 'GeradorBase.py'
# que simula dados de vendas, custos, lucros e satisfação de pontos de venda (PDVs)
# ao longo do tempo, incorporando fatores sazonais e econômicos.
# O objetivo deste notebook é realizar uma análise exploratória desses dados
# para entender suas características principais e identificar padrões relevantes.
# ============================
# Atualização - Rodrigo Pampolin
# Importação de bibliotecas e carregamento dos dados
# Data: 17/09/2025
# ===========================
# Atualização - Gabriel Tofanello
# Criação da estrutura básica do e das estatisticas descritivas
# Data: 17/09/2025
# ===========================
# Atualização - Felipe Damasceno
# Distribuição temporal e matriz de correlação
# Data: 19/09/2025
# ===========================
# # Atualização - Hyandra Marjory
# Testes, validação dos dados e resultados
# Data: 20/09/2025
# ===========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nome_produto = 'Sheet1'
base = pd.read_excel('..\Data\base_vendas_pdv_IA.xlsx', sheet_name=nome_produto)

# Estrutura básica
print("Dimensões:", base.shape)
print("\nColunas:", base.columns)
print(base.head())

# Estatísticas descritivas
print("\nEstatísticas descritivas:\n",base.describe())

# Valores ausentes
print("\nValores ausentes por coluna:\n")
print(base.isnull().sum())

# Distribuição temporal
base["Ano"] = base["Data"].dt.year
base["Mes"] = base["Data"].dt.month

base.groupby("Ano")["Vendas"].sum().plot(kind="bar", figsize=(10,5))
plt.title("Evolução anual das vendas")
plt.ylabel("Total de Vendas")
plt.show()

# Correlação entre variáveis numéricas
corr = base[["Vendas","Custo","Lucro","Satisfacao","Meta"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()