from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 📌 Cargar dataset de logs
df_logs = pd.read_csv("log_anomalies.csv")
X = df_logs[["cpu_usage", "memory_usage", "latency", "log_errors"]].values
y = df_logs["anomaly"].values

# 📌 Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Configurar el simulador cuántico
simulator = Aer.get_backend("statevector_simulator")
quantum_kernel = QuantumKernel(quantum_instance=simulator)

# 📌 Crear y entrenar el clasificador cuántico
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# 📌 Evaluación del modelo
y_pred = qsvc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo cuántico para detección de anomalías: {accuracy:.4f}")
