import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 📌 Generar un dataset simulado de métricas de servidores
np.random.seed(42)
num_samples = 200

# Variables: CPU (%), RAM (GB), Temperatura (°C), Latencia (ms)
X = np.random.rand(num_samples, 4) * [100, 32, 90, 200]  # Valores realistas
y = 0.5 * X[:, 0] + 0.2 * X[:, 1] + 0.3 * X[:, 2] - 0.1 * X[:, 3]  # Fórmula para riesgo de fallo

# Normalización
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Configurar el dispositivo cuántico
dev = qml.device("default.qubit", wires=4)

# 📌 Definir el circuito cuántico
@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, weights):
    for i in range(4):
        qml.RX(inputs[i], wires=i)
    qml.RY(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RZ(weights[3], wires=3)
    return qml.expval(qml.PauliZ(0))

# 📌 Modelo cuántico
def quantum_model(X, weights):
    return np.array([quantum_circuit(x, weights) for x in X], dtype=float)

# 📌 Inicializar pesos y optimizador
weights = np.random.random(4, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# 📌 Función de costo
def cost_function(w):
    y_pred = quantum_model(X_train, w)
    return np.mean((y_train - y_pred) ** 2)

# 📌 Entrenar la QNN
for epoch in range(100):
    weights, loss = opt.step_and_cost(cost_function, weights)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 📌 Evaluación
y_pred = quantum_model(X_test, weights)
mse_test = np.mean((y_test - y_pred) ** 2)
print("MSE en prueba:", mse_test)
