from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pennylane as qml

# 📌 Inicializar la API
app = FastAPI()

# 📌 Configurar dispositivo cuántico
dev = qml.device("default.qubit", wires=6)

# 📌 Definir el circuito cuántico
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(6):
        qml.RX(inputs[i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=list(range(6)))
    return qml.expval(qml.PauliZ(0))

# 📌 Modelo cuántico
def quantum_model(inputs):
    weights = np.random.random((3, 6, 3))  # Simulación de pesos pre-entrenados
    return float(quantum_circuit(inputs, weights))

# 📌 Definir esquema de datos de entrada
class ServerMetrics(BaseModel):
    cpu: float
    ram: float
    temp: float
    latency: float
    disk: float
    errors: float

# 📌 Endpoint de predicción
@app.post("/predict")
async def predict_failure(metrics: ServerMetrics):
    inputs = np.array([metrics.cpu, metrics.ram, metrics.temp, metrics.latency, metrics.disk, metrics.errors]) / 100
    risk = quantum_model(inputs)
    return {"server_failure_risk": risk}
# 📌 Cargar dataset mejorado
df_logs = pd.read_csv("log_anomalies_v2.csv")

X = df_logs.drop(columns=["anomaly"]).values
y = df_logs["anomaly"].values

# 📌 Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Entrenar el clasificador cuántico con el nuevo dataset
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)
