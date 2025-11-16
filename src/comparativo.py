import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar as métricas individuais
xgbr_df = pd.read_excel("metricas_modelos_por_pdv.xlsx")
lstm_df = pd.read_excel("metricas_lstm_por_pdv.xlsx")

# Conferir as primeiras linhas
print(xgbr_df.head())
print(lstm_df.head())

# Renomear colunas do LSTM para evitar conflito
lstm_df = lstm_df.rename(columns={
    "MAE": "MAE_LSTM",
    "RMSE": "RMSE_LSTM"
})

# Renomear colunas do XGBR (caso necessário)
xgbr_df = xgbr_df.rename(columns={
    "MAE": "MAE_XGBR",
    "RMSE": "RMSE_XGBR"
})

# Juntar os dois dataframes pelo PDV
comparativo = xgbr_df.merge(lstm_df, on="PDV_ID", how="inner")

# Mostrar o resultado final
print(comparativo.head())


# Merge
comparativo = xgbr_df.merge(lstm_df, on="PDV_ID", how="inner")

# --- Plot 1: MAE curve ---
plt.figure()
plt.plot(comparativo["PDV_ID"], comparativo["MAE_XGBR"], label="MAE XGBR")
plt.plot(comparativo["PDV_ID"], comparativo["MAE_LSTM"], label="MAE LSTM")
plt.xlabel("PDV")
plt.ylabel("MAE")
plt.title("MAE por PDV")
plt.legend()
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.show()

# --- Plot 2: RMSE curve ---
plt.figure()
plt.plot(comparativo["PDV_ID"], comparativo["RMSE_XGBR"], label="RMSE XGBR")
plt.plot(comparativo["PDV_ID"], comparativo["RMSE_LSTM"], label="RMSE LSTM")
plt.xlabel("PDV")
plt.ylabel("RMSE")
plt.title("RMSE por PDV")
plt.legend()
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.show()

# --- Plot 3: Histogram MAE ---
plt.figure()
plt.hist(comparativo["MAE_XGBR"], bins=20, alpha=0.6, label="XGBR")
plt.hist(comparativo["MAE_LSTM"], bins=20, alpha=0.6, label="LSTM")
plt.xlabel("MAE")
plt.ylabel("Frequência")
plt.title("Distribuição do MAE")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# GRÁFICO DE BARRAS SOBREPOSTAS - MAE
# ------------------------------
plt.figure(figsize=(14,6))

plt.bar(comparativo["PDV_ID"], comparativo["MAE_XGBR"], alpha=0.6, label="MAE XGBR")
plt.bar(comparativo["PDV_ID"], comparativo["MAE_LSTM"], alpha=0.6, label="MAE LSTM")

plt.title("Comparação de MAE por PDV - XGBR vs LSTM")
plt.xlabel("PDV")
plt.ylabel("MAE")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.grid(axis="y")

plt.show()

# ------------------------------
# GRÁFICO DE BARRAS SOBREPOSTAS - RMSE
# ------------------------------
plt.figure(figsize=(14,6))

plt.bar(comparativo["PDV_ID"], comparativo["RMSE_XGBR"], alpha=0.6, label="RMSE XGBR")
plt.bar(comparativo["PDV_ID"], comparativo["RMSE_LSTM"], alpha=0.6, label="RMSE LSTM")

plt.title("Comparação de RMSE por PDV - XGBR vs LSTM")
plt.xlabel("PDV")
plt.ylabel("RMSE")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.grid(axis="y")

plt.show()

# --- Plot 4: Scatter comparativo ---
comparativo["DIF"] = comparativo["MAE_LSTM"] - comparativo["MAE_XGBR"]

plt.figure(figsize=(8,8))

# Cores: azul = XGBR melhor, vermelho = LSTM melhor
cores = np.where(comparativo["DIF"] > 0, "red", "blue")

plt.scatter(
    comparativo["MAE_XGBR"],
    comparativo["MAE_LSTM"],
    c=cores,
    s=80,
    alpha=0.8
)

# Linha de igualdade
min_val = min(comparativo["MAE_XGBR"].min(), comparativo["MAE_LSTM"].min())
max_val = max(comparativo["MAE_XGBR"].max(), comparativo["MAE_LSTM"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")

plt.title("Correlação MAE XGBR vs LSTM")
plt.xlabel("MAE XGBR")
plt.ylabel("MAE LSTM")
plt.grid(True)

# Legenda manual
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='XGBR Pior', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='LSTM Pior', markersize=10)
]
plt.legend(handles=legend_elements)

plt.show()

comparativo.head()