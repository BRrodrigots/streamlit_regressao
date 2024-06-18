import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f

st.title("Análise de Regressão Múltipla e Tabela ANOVA")

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Escolha um arquivo Excel", type="xlsx")

if uploaded_file is not None:
    # Leitura do arquivo Excel
    df = pd.read_excel(uploaded_file)

    # Verificação se as colunas necessárias estão presentes
    if not set(['y', 'x1', 'x2']).issubset(df.columns):
        st.error("O arquivo Excel deve conter as colunas: 'y', 'x1', e 'x2'.")
    else:
        # Extraindo os dados
        y = df['y'].values
        x1 = df['x1'].values
        x2 = df['x2'].values

        n = len(y)

        # Calculando as médias
        media_x1 = np.mean(x1)
        media_x2 = np.mean(x2)
        media_Y = np.mean(y)

        # Calculando as somas e produtos
        soma_Y = np.sum(y)
        soma_x1 = np.sum(x1)
        soma_x2 = np.sum(x2)
        soma_x1Y = np.sum(x1 * y)
        soma_x2Y = np.sum(x2 * y)
        soma_x1_quadrado = np.sum(x1**2)
        soma_x2_quadrado = np.sum(x2**2)
        soma_y_quadrado = np.sum(y**2)
        soma_x1x2 = np.sum(x1 * x2)

        # Construindo a matriz X
        X = np.column_stack((np.ones(n), x1, x2))

        # Calculando os coeficientes b0, b1, b2 usando o método de mínimos quadrados
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        b0, b1, b2 = coefficients

        # Calculando a ANOVA
        y_mean = np.mean(y)
        SST = np.sum((y - y_mean)**2)
        y_pred = b0 + b1 * x1 + b2 * x2
        SSR = np.sum((y_pred - y_mean)**2)
        SSE = np.sum((y - y_pred)**2)

        df_model = 2
        df_error = n - df_model - 1

        MSR = SSR / df_model
        MSE = SSE / df_error

        F_value = MSR / MSE
        p_value = 1 - f.cdf(F_value, df_model, df_error)

        F_tab = f.ppf(0.95, df_model, df_error)

        R2 = SSR / SST
        R2_adjusted = 1 - (SSE / df_error) / (SST / (n - 1))

        alpha = 0.05

        decisao = ("Rejeita-se H0: Existem evidências estatísticas de que pelo menos um dos coeficientes é diferente de zero (H1 é verdadeira)."
                   if p_value < alpha else
                   "Aceita-se H0: Não há evidências estatísticas suficientes para concluir que pelo menos um dos coeficientes é diferente de zero (H0 é verdadeira).")

        anova_table = pd.DataFrame({
            "FV": ['Regressão', 'Residuo', 'Total'],
            "GL": [df_model, df_error, n - 1],
            "SQ": [SSR, SSE, SST],
            "QM": [MSR, MSE, np.nan],
            "Fcal": [F_value, np.nan, np.nan],
            "Ftab": [F_tab, np.nan, np.nan],
            "R²": [R2, np.nan, np.nan],
            "R²ajust": [R2_adjusted, np.nan, np.nan],
            "Teste de Hipótese": [decisao, np.nan, np.nan]
        })

        st.write("Tabela ANOVA")
        st.dataframe(anova_table)

        st.write("Resultados")
        st.write(f"b0 (intercepto): {b0}")
        st.write(f"b1 (coeficiente para 'x1'): {b1}")
        st.write(f"b2 (coeficiente para 'x2'): {b2}")
        st.write(f"Soma de Y: {soma_Y}")
        st.write(f"Soma de x1: {soma_x1}")
        st.write(f"Soma de x2: {soma_x2}")
        st.write(f"Média de x1: {media_x1}")
        st.write(f"Média de x2: {media_x2}")
        st.write(f"Média de Y: {media_Y}")
        st.write(f"Soma x1²: {soma_x1_quadrado - ((soma_x1)**2 / n)}")
        st.write(f"Soma x2²: {soma_x2_quadrado - ((soma_x2)**2 / n)}")
        st.write(f"Soma Y²: {soma_y_quadrado - ((soma_Y)**2 / n)}")
        st.write(f"Soma x1y: {soma_x1Y - ((soma_x1)*(soma_Y)/n)}")
        st.write(f"Soma x2y: {soma_x2Y - ((soma_x2)*(soma_Y)/n)}")
        st.write(f"Soma x1x2: {soma_x1x2 - ((soma_x1 * soma_x2)/n)}")
