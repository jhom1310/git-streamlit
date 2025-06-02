import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import re
import numpy as np

# Função para ajustar um polinômio à curva de potência e prever valores de potência esperados
def fit_polynomial_curve(filtered_data, degree=3):
    """ Ajusta uma curva polinomial para a relação entre velocidade do vento e potência gerada """
        
    # Ajustando um modelo de regressão polinomial para os dados
    X = filtered_data[col_vel_vento].values.reshape(-1, 1)
    y = filtered_data[col_pot].values
        
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
        
    model = LinearRegression()
    model.fit(X_poly, y)
        
    # Prevendo a potência com base no modelo ajustado
    y_pred = model.predict(X_poly)
        
    return model, X, y, y_pred

# Função para filtrar dados baseado na previsão da curva de potência ajustada
def filter_data_based_on_curve(filtered_data, model, X, y_pred, tolerance=50):
        """ Filtra os dados de forma suave, mantendo apenas os pontos próximos à curva ajustada """
        
        # Definir uma margem de erro (tolerância) em torno da curva de potência
        valid_data = []
        
        for i, wind_speed in enumerate(X.flatten()):
            actual_power = filtered_data.iloc[i][col_pot]
            predicted_power = y_pred[i]
            
            # Se a potência real estiver dentro da tolerância da potência prevista pela curva
            if abs(actual_power - predicted_power) <= tolerance:
                valid_data.append(filtered_data.iloc[i])
        
        # Criar um DataFrame com os dados válidos
        valid_df = pd.DataFrame(valid_data)
        
        return valid_df

    

def print_curva_potencia(df):
    # Filtrar os dados para garantir que estamos considerando apenas as potências maiores que zero
    filtered_data_power = df[df[col_pot] > 0]

    # Gráfico de dispersão de velocidade do vento (x) vs potência gerada (y)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(filtered_data_power[col_vel_vento], filtered_data_power[col_pot], 
                color='b', alpha=0.3, marker='o', s=10)

    ax.set_title("Dispersão: Velocidade do Vento vs Potência Gerada", fontsize=14)
    ax.set_xlabel("Velocidade do Vento (m/s)", fontsize=12)
    ax.set_ylabel("Potência Gerada (kW)", fontsize=12)
    ax.set_xlim(0, 20)
    ax.grid(True)

    st.pyplot(fig)


def meio_intervalo(intervalo):
    numeros = re.findall(r"-?\d+\.?\d*", str(intervalo))
    if len(numeros) == 2:
        return (float(numeros[0]) + float(numeros[1])) / 2
    return float(numeros[0]) if numeros else None

def extremos_intervalo(intervalo):
    numeros = re.findall(r"-?\d+\.?\d*", str(intervalo))
    if len(numeros) == 2:
        return f"{numeros[0]}-{numeros[1]}"
    return numeros[0] if numeros else None

def met_pz(filtered_data):
    m = filtered_data[col_vel_vento].mean()
    st.write(f'Média de velocidade de vento = {m:.2f} m/s')
    

    mean_power_data = (
        filtered_data.groupby(["WTUR_WindDir_intervalo", "WNAC_AneWdSpd_intervalo"])[col_pot]
        .mean()
        .reset_index()
    )

    mean_power_data["WTUR_WindDir_intervalo"] = mean_power_data["WTUR_WindDir_intervalo"].apply(meio_intervalo)
    mean_power_data["WNAC_AneWdSpd_intervalo_legenda"] = mean_power_data["WNAC_AneWdSpd_intervalo"].apply(extremos_intervalo)

    mean_power_data = mean_power_data.sort_values("WTUR_WindDir_intervalo")

    max_points = mean_power_data.loc[mean_power_data.groupby("WNAC_AneWdSpd_intervalo")[col_pot].idxmax()]

    vertices = []

    fig, ax = plt.subplots(figsize=(14, 10))
    palette = sns.color_palette("husl", len(mean_power_data["WNAC_AneWdSpd_intervalo"].unique()))

    for i, speed in enumerate(mean_power_data["WNAC_AneWdSpd_intervalo"].unique()):
        subset = mean_power_data[mean_power_data["WNAC_AneWdSpd_intervalo"] == speed]
        label_legenda = subset["WNAC_AneWdSpd_intervalo_legenda"].iloc[0]

        sns.lineplot(
            data=subset,
            x="WTUR_WindDir_intervalo",
            y=col_pot,
            marker="o",
            label=label_legenda,
            color=palette[i],
            ax=ax
        )

        coef = np.polyfit(subset["WTUR_WindDir_intervalo"], subset[col_pot], 2)
        p = np.poly1d(coef)

        x_vals = np.linspace(subset["WTUR_WindDir_intervalo"].min(), subset["WTUR_WindDir_intervalo"].max(), 100)
        y_vals = p(x_vals)

        y_pred = p(subset["WTUR_WindDir_intervalo"].values)
        r2 = 1 - np.sum((subset[col_pot].values - y_pred) ** 2) / np.sum((subset[col_pot].values - np.mean(subset[col_pot].values)) ** 2)

        x_vertex = -coef[1] / (2 * coef[0])
        y_vertex = p(x_vertex)

        vertices.append((speed, x_vertex, y_vertex, r2))

        cor_linha = "green" if r2 >= 0.4 else "gray"
        largura = 2
        alpha = 0.8 if r2 >= 0.4 else 0.4

        ax.plot(x_vals, y_vals, linestyle="--", color=cor_linha, linewidth=largura, alpha=alpha)
        ax.scatter(x_vertex, y_vertex, color=cor_linha, edgecolors="black", s=70, zorder=5)

    def distancia(speed_str):
        try:
            return abs(meio_intervalo(speed_str) - m)
        except:
            return np.inf

    vertices_validos = [v for v in vertices if v[3] >= 0.4]

    if vertices_validos:
        mais_proximo = min(vertices_validos, key=lambda v: distancia(v[0]))
        if not (-10 <= mais_proximo[1] <= 10):
            candidatos = [v for v in vertices_validos if -10 <= v[1] <= 10]
            if candidatos:
                mais_proximo = max(candidatos, key=lambda v: v[3])

        speed_str, x_v, y_v, r2_v = mais_proximo
        ax.scatter(x_v, y_v, color="blue", edgecolors="black", s=150, zorder=10, label="Ponto Zero")
        ax.text(
            0.99, 0.99,
            f"Ponto Zero\nVelocidade: {speed_str}\nVértice: ({x_v:.2f}, {y_v:.2f})\nR²: {r2_v:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            color="blue",
            weight="bold",
            ha='right',
            va='top',
            bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3', alpha=0.8)
        )
    else:
        st.warning("Nenhum vértice com R² ≥ 0.5 foi encontrado.")

    ax.scatter(
        max_points["WTUR_WindDir_intervalo"],
        max_points[col_pot],
        color="red",
        edgecolors="black",
        s=100,
        label="Ponto Máximo de Potência"
    )

    for _, row in max_points.iterrows():
        ax.annotate(
            f"{row[col_pot]:.0f}",
            (row["WTUR_WindDir_intervalo"], row[col_pot]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            color="black"
        )

    ax.text(
        0.01, 0.99,
        f'Média da Velocidade do Vento: {m:.2f} m/s',
        transform=ax.transAxes,
        fontsize=12,
        color='darkred',
        weight='bold',
        va='top',
        ha='left'
    )

    ax.set_title("Média da Potência Gerada com Regressão Quadrática", fontsize=14)
    ax.set_xlabel("Direção do Vento (graus)", fontsize=12)
    ax.set_ylabel("Média da Potência Gerada", fontsize=12)
    ax.set_xticks(ax.get_xticks())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(-15, 15)
    ax.set_ylim(200, 3500)
    ax.legend(title="Velocidade (m/s)", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title_fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Vértices das parábolas")
    vertices_df = pd.DataFrame(vertices, columns=["Velocidade", "Vértice X", "Vértice Y", "R²"])
    st.dataframe(vertices_df)

    return vertices


st.title("Análise de Dados: Erro de Ponto Zero em Aerogeradores")
st.markdown("#### Ferramenta desenvolvida na pesquisa de Mestrado de [Aleff Souza](https://www.linkedin.com/in/aleffsouza/)" \
" com objetivo de diagnosticar desvios de ponto zero em aerogeradores utilizado somente dados SCADA.")
st.markdown("Para uso da ferramenta é necesario um arquivo no formato CSV com dados SCADA de um aerogerador.")


uploaded_file = st.file_uploader("Envie o arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";")
    except:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.success("Arquivo carregado com sucesso!")

    st.markdown("### Relacione suas colunas com as funções esperadas")

    """ default=["TS", "WST_Wtg_State2", "WTUR_TotPwr", "WTUR_WindDir", "WNAC_AneWdSpd", "WYAW_YwAngGr"] """

    # Lista de funções esperadas
    funcoes_esperadas = {
        "timestamp": "Timestamp",
        "status": "Status da turbina",
        "potencia": "Potência gerada",
        "direcao_vento": "Direção do vento",
        "velocidade_vento": "Velocidade do vento",
        "angulo_yaw": "Ângulo de YAW"
    }

    # Mapeamento: função -> coluna
    mapeamento_colunas = {}

    for chave, descricao in funcoes_esperadas.items():
        col_selecionada = st.selectbox(
            f"Selecione a coluna para: {descricao}",
            options=df.columns.tolist(),
            key=chave
        )
        mapeamento_colunas[chave] = col_selecionada

    st.markdown("### Mapeamento final:")
    st.write(mapeamento_colunas)

    if all(mapeamento_colunas.values()) and len(set(mapeamento_colunas.values())) == len(mapeamento_colunas):
        col_ts = mapeamento_colunas["timestamp"]
        col_status = mapeamento_colunas["status"]
        col_pot = mapeamento_colunas["potencia"]
        col_dir_vento = mapeamento_colunas["direcao_vento"]
        col_vel_vento = mapeamento_colunas["velocidade_vento"]
        col_yaw = mapeamento_colunas["angulo_yaw"]

        df = df[list(mapeamento_colunas.values())]
        df[col_ts] = pd.to_datetime(df[col_ts], errors='coerce')

        valor_ok = st.number_input("Informe o valor considerado OK para STATUS:", value=None)

        filter_1 = df[col_status] == valor_ok
        filter_3 = df[col_pot] > 300
        filter_4 = df[col_dir_vento].between(-15, 15)
        filter_5 = df[col_vel_vento].between(4, 10)

        filtered_data = df[filter_1 & filter_3 & filter_4 & filter_5]

        # Criar intervalos
        intervalos_velocidade = pd.interval_range(start=4, end=10, freq=0.5)
        intervalos_direcao = pd.interval_range(start=-12, end=12, freq=1)
        filtered_data["WNAC_AneWdSpd_intervalo"] = pd.cut(filtered_data[col_vel_vento], bins=intervalos_velocidade)
        filtered_data["WTUR_WindDir_intervalo"] = pd.cut(filtered_data[col_dir_vento], bins=intervalos_direcao)

        st.subheader("Dados filtrados")
        st.write(filtered_data.describe())

        # Gráfico de dispersão
        st.subheader("Gráfico de Dispersão: Velocidade vs Direção do Vento")
        plt.figure(figsize=(8, 5))
        plt.scatter(filtered_data[col_vel_vento], filtered_data[col_dir_vento], color='b', alpha=0.5)
        plt.xlabel("Velocidade do Vento (m/s)")
        plt.ylabel("Direção do Vento (°)")
        plt.title("Gráfico de Dispersão: Velocidade vs Direção do Vento")
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        # Novo gráfico de dispersão
        st.subheader("Dispersão: Velocidade do Vento vs Potência")
        power_data = filtered_data[filtered_data[col_pot] > 0]
        plt.figure(figsize=(10, 6))
        plt.scatter(power_data[col_vel_vento], power_data[col_pot], color='b', alpha=0.5, s=10)
        plt.title("Dispersão: Velocidade do Vento vs Potência Gerada")
        plt.xlabel("Velocidade do Vento (m/s)")
        plt.ylabel("Potência Gerada (kW)")
        plt.xlim(0, 20)
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        # Calcular a média de potência para cada intervalo de velocidade do vento (WNAC_AneWdSpd_intervalo_SPIN)
        mean_potencia_wind_speed_interval = filtered_data.groupby("WTUR_WindDir_intervalo")[col_pot].mean()

        # Calcular a média geral de potência
        mean_potencia_total = filtered_data[col_pot].mean()

        if valor_ok is not None:
            st.subheader("Limpeza dos dados:")
            # Suponha que seu DataFrame se chame "filtered_data"
            with st.spinner("Filtrando dados e gerando gráfico..."):

                model, X, y, y_pred = fit_polynomial_curve(filtered_data, degree=3)
                filtered_data_cleaned = filter_data_based_on_curve(filtered_data, model, X, y_pred, tolerance=200)

                # Visualizar os dados limpos
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(filtered_data_cleaned[col_vel_vento], filtered_data_cleaned[col_pot], color="red", alpha=0.6, label='Dados Filtrados')
                # ax.scatter(filtered_data_cleaned['YA_corr_avg'], filtered_data_cleaned[col_pot], color="blue", alpha=0.6, label='Dados Filtrados')
                # ax.plot(X, y_pred, color='blue', label='Curva de Potência Ajustada (Polinomial)')
                ax.set_title("Dados Filtrados: Potência vs Velocidade do Vento")
                ax.set_xlabel("Velocidade do Vento (m/s)")
                ax.set_ylabel("Potência Gerada (kW)")
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 3500)
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                # Mostrar os dados limpos
                st.subheader("Amostra dos Dados Filtrados")
                st.dataframe(filtered_data_cleaned.describe())

                # Mostrar quantidade de linhas
                st.write("Número de linhas:", len(filtered_data_cleaned))

            filtered_data = filtered_data_cleaned
            filtered_data['semana'] = filtered_data[col_ts].dt.to_period('M').apply(lambda r: r.start_time)
            #df['semana'] = df['TS'].dt.to_period('Q').apply(lambda r: r.start_time)
            dfs= []
            dfs.append(filtered_data)

            for i, df in enumerate(dfs):
                diferencas = []
                vertices_n = []
                teste_ver_n = []
                agrupamento_mensal = df.groupby('semana')
                lista_de_grupos = list(agrupamento_mensal)

                # Cria uma coluna de rótulo vazia no DataFrame
                df['rótulo'] = np.nan

                for i in range(len(lista_de_grupos) - 1):
                    mes_atual, grupo = lista_de_grupos[i]

                    #grupo = grupo[grupo['WTUR_WindDir'] - grupo['YA_corr_avg'] > 0.5]
                    st.write("### Período ", i+1)
                    st.write("Data:", mes_atual)
                    print('######### CATAVENTO ############')
                    print(f'Semana {mes_atual}')
                    print_curva_potencia(grupo)
                    #my = dif_sensor(grupo)
                    met_pz(grupo)
    else:
        st.warning("Por favor, selecione todas as colunas corretamente e sem repetições.")

    

    


    


    