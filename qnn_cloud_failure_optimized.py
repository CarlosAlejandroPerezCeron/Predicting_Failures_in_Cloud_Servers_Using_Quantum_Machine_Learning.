import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 📌 Generar un dataset simulado de métricas de servidores con más variables
np.random.seed(42)
num_samples = 500

# Variables: CPU (%), RAM (GB), Temp (°C), Latencia (ms), Uso de Disco (%), Errores en logs
X = np.random.rand(num_samples, 6) * [100, 32, 90, 200, 100, 50]  # Valores realistas
y = (
    0.4 * X[:, 0] + 0.2 * X[:, 1] + 0.2 * X[:, 2]
    - 0.1 * X[:, 3] + 0.3 * X[:, 4] + 0.5 * X[:, 5]
)  # Nueva fórmula para riesgo de fallo

# Normalización
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Configurar el dispositivo cuántico con más qubits
dev = qml.device("default.qubit", wires=6)

# 📌 Definir el circuito cuántico
@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, weights):
    for i in range(6):
        qml.RX(inputs[i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=list(range(6)))
    return qml.expval(qml.PauliZ(0))

# 📌 Modelo cuántico optimizado
def quantum_model(X, weights):
    return np.array([quantum_circuit(x, weights) for x in X], dtype=float)

# 📌 Inicializar pesos
num_layers = 3
weights = np.random.random((num_layers, 6, 3), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.01)

# 📌 Función de costo
def cost_function(w):
    y_pred = quantum_model(X_train, w)
    return np.mean((y_train - y_pred) ** 2)

# 📌 Entrenar la QNN optimizada
for epoch in range(200):
    weights, loss = opt.step_and_cost(cost_function, weights)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 📌 Evaluación del modelo
y_pred = quantum_model(X_test, weights)
mse_test = np.mean((y_test - y_pred) ** 2)
print("MSE en prueba (modelo optimizado):", mse_test)
