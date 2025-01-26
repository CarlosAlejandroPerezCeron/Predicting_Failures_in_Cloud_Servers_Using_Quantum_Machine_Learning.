import numpy as np
import pandas as pd

# 📌 Generar un dataset mejorado de logs con más métricas
np.random.seed(42)
num_samples = 2000

# Variables nuevas agregadas
cpu_usage = np.random.uniform(10, 100, num_samples)  # %
memory_usage = np.random.uniform(1, 64, num_samples)  # GB
latency = np.random.uniform(5, 300, num_samples)  # ms
log_errors = np.random.uniform(0, 100, num_samples)  # Cantidad de errores
disk_usage = np.random.uniform(10, 100, num_samples)  # % de uso de disco
server_temp = np.random.uniform(30, 90, num_samples)  # °C
active_connections = np.random.randint(10, 1000, num_samples)  # Número de conexiones activas
peak_load = np.random.uniform(0.5, 5.0, num_samples)  # Picos de carga en servidores

# Generar etiquetas (0 = normal, 1 = anomalía)
labels = np.array([0] * 1800 + [1] * 200)  # 10% anomalías
np.random.shuffle(labels)

# Crear DataFrame con las nuevas métricas
df_logs = pd.DataFrame({
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "latency": latency,
    "log_errors": log_errors,
    "disk_usage": disk_usage,
    "server_temp": server_temp,
    "active_connections": active_connections,
    "peak_load": peak_load,
    "anomaly": labels
})

# Guardar el dataset
df_logs.to_csv("log_anomalies_v2.csv", index=False)
print("Dataset mejorado generado y guardado en log_anomalies_v2.csv")
