import numpy as np
import pandas as pd
import random

# 📌 Generar un dataset de logs con anomalías
np.random.seed(42)
num_samples = 1000

# Variables: Uso de CPU, Memoria, Latencia de Red, Errores en Logs
cpu_usage = np.random.uniform(10, 100, num_samples)  # %
memory_usage = np.random.uniform(1, 32, num_samples)  # GB
latency = np.random.uniform(10, 200, num_samples)  # ms
log_errors = np.random.uniform(0, 50, num_samples)  # Cantidad de errores

# Generar etiquetas (0 = normal, 1 = anomalía)
labels = np.array([0] * 950 + [1] * 50)  # 5% anomalías
np.random.shuffle(labels)

# Crear DataFrame
df_logs = pd.DataFrame({
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "latency": latency,
    "log_errors": log_errors,
    "anomaly": labels
})

# Guardar el dataset
df_logs.to_csv("log_anomalies.csv", index=False)
print("Dataset generado y guardado en log_anomalies.csv")
