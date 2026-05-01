import numpy as np
from scipy.optimize import minimize

# 1. Dataset (10 Sampel)
X_raw = np.array([
    [1702743, 20, 64, 10, 1], [4314572, 25, 31, 8, 1], [4936484, 26, 56, 6, 0],
    [1200000, 15, 45, 12, 0], [6000000, 40, 28, 19, 1], [800000, 10, 70, 9, 0],
    [3500000, 22, 33, 15, 1], [1500000, 18, 52, 11, 0], [2800000, 30, 40, 16, 0],
    [950000, 12, 58, 10, 1]
])
y = np.array([1, -1, -1, 1, -1, 1, -1, 1, -1, 1])

# 2. Preprocessing (Normalisasi Total Belanja ke Juta)
X = X_raw.astype(float)
X[:, 0] = X[:, 0] / 1000000

# 3. Solusi SVM Dual Problem
# Fungsi yang akan di-MINIMASI (Negatif dari Lagrange Dual)
def objective(alpha):
    # L(alpha) = sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
    # Karena kita menggunakan 'minimize', kita kalikan -1 agar menjadi maksimasi
    part1 = np.sum(alpha)
    part2 = 0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            part2 += alpha[i] * alpha[j] * y[i] * y[j] * np.dot(X[i], X[j])
    return -(part1 - 0.5 * part2)

# Kendala sum(alpha * y) = 0
cons = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
# Batasan alpha >= 0
bounds = [(0, None) for _ in range(len(y))]

# Optimasi
res = minimize(objective, np.zeros(len(y)), bounds=bounds, constraints=cons)
alphas = res.x

# 4. Hitung Bobot (W)
# Rumus: w = sum(alpha_i * y_i * x_i)
w = np.zeros(X.shape[1])
for i in range(len(alphas)):
    w += alphas[i] * y[i] * X[i]

# 5. Hitung Bias (B)
# b = y_k - w . x_k (untuk data di mana alpha > 0)
support_vectors = np.where(alphas > 1e-5)[0]
b = 0
for i in support_vectors:
    b += y[i] - np.dot(w, X[i])
b /= len(support_vectors)

print("HASIL TRAINING SVM")
print("-" * 20)
print(f"Bobot (W) : {np.round(w, 4)}")
print(f"Bias (B)  : {round(b, 4)}")
print(f"Support Vectors (ID): {support_vectors + 1}")
