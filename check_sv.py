import numpy as np
from scipy.optimize import minimize

# Dataset 10 item
X = np.array([
    [0.8, 12, 58, 10, 0], [1.2, 15, 45, 12, 0], [3.5, 25, 33, 10, 0],
    [4.3, 30, 31, 15, 1], [6.0, 40, 28, 18, 0], [0.9, 10, 70, 9, 0],
    [5.0, 20, 56, 12, 1], [1.5, 15, 52, 14, 0], [1.2, 12, 58, 11, 0],
    [2.8, 18, 40, 13, 0]
])
y = np.array([1, 1, -1, -1, -1, 1, -1, 1, 1, -1])

def objective(alpha):
    return -(np.sum(alpha) - 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(y, y) * np.dot(X, X.T)))

constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]
bounds = [(0, 10) for _ in range(len(y))]
res = minimize(objective, np.zeros(len(y)), bounds=bounds, constraints=constraints)

alphas = res.x
sv_indices = np.where(alphas > 1e-4)[0]
print(f"SV Indices: {sv_indices}")
print(f"SV Alphas: {alphas[sv_indices]}")
