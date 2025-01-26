import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 📌 Cargar dataset mejorado
df_logs = pd.read_csv("log_anomalies_v2.csv")

# 📌 Título del dashboard
st.title("📊 Monitoreo de Anomalías en Servidores - Quantum AI")

# 📌 Mostrar datos en tiempo real
st.subheader("Datos en Tiempo Real")
st.write(df_logs.sample(10))  # Muestra aleatoria de datos

# 📌 Visualización de anomalías
st.subheader("Distribución de Anomalías")
fig, ax = plt.subplots()
df_logs["anomaly"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
ax.set_xticklabels(["Normal", "Anomalía"], rotation=0)
st.pyplot(fig)

# 📌 Visualizar tendencias en CPU y Memoria
st.subheader("Uso de CPU y Memoria en Servidores")
fig, ax = plt.subplots()
df_logs.plot(x="cpu_usage", y="memory_usage", kind="scatter", ax=ax, alpha=0.5)
st.pyplot(fig)
