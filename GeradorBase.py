import pandas as pd
import numpy as np

# ============================
# Configurações iniciais
# ============================
np.random.seed(42)
dates = pd.date_range(start="2015-01-01", end="2025-08-01", freq="MS")
n_pdv = 53

# Estados para os PDVs
estados = ["SP", "RJ", "MG", "RS", "BA", "PE", "PR", "DF"]
prob_estados = [0.25, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05]

# Estado e N_Funcionarios fixos por PDV
np.random.seed(99)
pdv_estados = np.random.choice(estados, size=n_pdv, p=prob_estados)
np.random.seed(100)
pdv_funcionarios = np.random.randint(5, 51, size=n_pdv)

data_list = []

# ============================
# Geração da base mês a mês
# ============================
for date in dates:
    dias_uteis = np.random.randint(18, 23)

    # -------------------
    # META com curva logística + sazonalidade
    # -------------------
    meta_base = 3000 / (1 + np.exp(-0.3 * (date.year - 2018)))  # curva logística

    # Sazonalidade anual + trimestral
    sazonalidade = (
        200 * np.sin(2 * np.pi * date.month / 12) +
        80 * np.sin(2 * np.pi * date.month / 3)
    )
    meta_base += sazonalidade

    # Choques econômicos
    if date.year == 2020:  # crise
        meta_base *= np.random.uniform(0.6, 0.8)
    elif date.year == 2023:  # boom
        meta_base *= np.random.uniform(1.2, 1.4)

    metas = np.abs(np.random.normal(meta_base, 150, size=n_pdv))

    # -------------------
    # Selic e inflação
    # -------------------
    selic = np.random.normal(10 - 0.5*(date.year-2015), 0.3) if date.year < 2021 else np.random.normal(9 + 1.0*(date.year-2019), 0.5)
    selic = np.clip(selic, 0, 20)
    inflacao = np.random.normal(0.5, 0.3)
    inflacao = np.clip(inflacao, -0.5, 1.5)

    # -------------------
    # Vendas dependem de meta + macro
    # -------------------
    fator = np.random.uniform(0.7, 1.7, size=n_pdv)
    fator_macro = 1 - 0.02 * (selic / 10) + 0.03 * (1 - inflacao)
    vendas = metas * fator * fator_macro + np.random.normal(0, 300, size=n_pdv)
    vendas = np.clip(vendas, 0, None)

    # Choques globais

    if date.month in [11, 12]:  # Natal / fim de ano
        vendas *= np.random.uniform(1.3, 1.6)

    for year in range(2015, 2026):
        if year == 2020 or year == 2021:  # pandemia
            vendas *= np.random.uniform(0.75, 0.9)
        elif year > 2020:
            vendas *= np.random.uniform(1.5, 2.0)
        else:
            vendas *= np.random.uniform(0.95, 1.05)

    # -------------------
    # Custos e Lucro
    # -------------------
    custo_fixo = np.random.uniform(500, 1500, size=n_pdv)
    custo_variavel = vendas * np.random.uniform(0.4, 0.8, size=n_pdv)
    custo = custo_fixo + custo_variavel + np.random.normal(0, 60, size=n_pdv)

    lucro = vendas - custo + np.random.normal(0, 120, size=n_pdv)

    # -------------------
    # Satisfação
    # -------------------
    base_satisfacao = 60 + 0.3 * pdv_funcionarios + np.random.normal(0, 8, size=n_pdv)
    ajuste_estado = np.where(pdv_estados == "SP", -5, np.random.randint(-2, 3, size=n_pdv))
    satisfacao = base_satisfacao + (vendas / metas - 1) * 30 + ajuste_estado
    satisfacao = np.clip(satisfacao, 0, 100)

    # -------------------
    # Eventos locais (aleatórios)
    # -------------------
    for i in range(n_pdv):
        if np.random.rand() < 0.01:  # campanha forte
            vendas[i] *= np.random.uniform(1.5, 2.5)
            custo[i] *= np.random.uniform(1.2, 1.8)
            satisfacao[i] += np.random.uniform(5, 15)
        elif np.random.rand() < 0.005:  # greve/problema
            vendas[i] *= np.random.uniform(0.3, 0.6)
            custo[i] *= np.random.uniform(0.5, 0.8)
            satisfacao[i] -= np.random.uniform(10, 20)

    # -------------------
    # Montagem do DataFrame do mês
    # -------------------
    df_month = pd.DataFrame({
        "Data": [date]*n_pdv,
        "PDV_ID": [f"PDV_{i+1}" for i in range(n_pdv)],
        "Estado": pdv_estados,
        "N_Funcionarios": pdv_funcionarios,
        "Dias_Uteis": dias_uteis,
        "Meta": metas,
        "Vendas": vendas,
        "Custo": custo,
        "Lucro": lucro,
        "Satisfacao": satisfacao,
        "Selic": selic,
        "Inflacao": inflacao,
        "Qtd_Cotacoes": np.random.randint(1, 21, size=n_pdv)
    })

    data_list.append(df_month)

# ============================
# Concatenar
# ============================
df_final = pd.concat(data_list, ignore_index=True)

# ============================
# Inserir valores ausentes
# ============================
np.random.seed(123)
df_final.loc[np.random.rand(len(df_final)) < 0.08, "Satisfacao"] = np.nan
df_final.loc[np.random.rand(len(df_final)) < 0.03, "Lucro"] = np.nan
df_final.loc[np.random.rand(len(df_final)) < 0.02, "Meta"] = np.nan

# ============================
# Recalcular Atingimento
# ============================
df_final["Atingimento"] = (df_final["Vendas"] / df_final["Meta"] * 100).round(2)

# ============================
# Arredondar valores
# ============================
cols_round = ["Vendas", "Meta", "Custo", "Lucro", "Satisfacao", "Selic", "Inflacao"]
df_final[cols_round] = df_final[cols_round].round(2)

# ============================
# Resumo
# ============================
print(df_final.head())
print("\nTamanho da base:", df_final.shape)
print("\nResumo Atingimento (%):")
print(df_final["Atingimento"].describe())

# ============================
# Salvar
# ============================
df_final.to_excel("base_vendas_pdv_2015_2025_v3.xlsx", index=False)
