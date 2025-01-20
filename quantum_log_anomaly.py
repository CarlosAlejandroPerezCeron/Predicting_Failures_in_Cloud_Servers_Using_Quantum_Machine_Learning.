from qiskit_machine_learning.algorithms import QSVC
from qiskit import Aer
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 📌 Generar datos de logs de servidores
X, y = make_classification(n_samples=500, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 📌 Configurar el kernel cuántico
simulator = Aer.get_backend("statevector_simulator")
quantum_kernel = QuantumKernel(quantum_instance=simulator)

# 📌 Crear y entrenar el clasificador cuántico
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# 📌 Evaluación del modelo
accuracy = qsvc.score(X_test, y_test)
print("Precisión del clasificador cuántico:", accuracy)
