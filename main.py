import numpy as np
from sklearn.linear_model import LinearRegression

# Надані дані
data = np.array([
    [10600, 5070, 25071],
    [10550, 5000, 25064],
    [10630, 5000, 25226],
    [10790, 5150, 25461],
    [10860, 5260, 25935],
    [10380, 4950, 24700],
    [10630, 5015, 25015],
    [10600, 5020, 25018],
    [10800, 5170, 25626],
    [10740, 5195, 25580]
])

# Розділити дані на K, L, F
K = data[:, 0]
L = data[:, 1]
Q = data[:, 2]

# Взяти логарифми
log_L = np.log(L)
log_K = np.log(K)
log_Q = np.log(Q)

# Побудувати матрицю функцій (X) та вектор результатів (y) для регресії
X = np.column_stack((np.ones_like(log_L), log_L, log_K))
y = log_Q

# Побудувати лінійну регресію
regression = LinearRegression().fit(X, y)

# Вивести параметри регресії (A, alpha, beta)
A = np.exp(regression.intercept_)
alpha = regression.coef_[1]
beta = regression.coef_[2]

print("A:", A)
print("alpha:", alpha)
print("beta:", beta)
print(f"Q = {A:.2f} * L^{alpha:.2f} * K^{beta:.2f}")

ES = alpha + beta
E = -beta / alpha

print("Ефект масштабу:", ES)
print("Еластичність заміщення:", E)

# Тепер, як ми маємо коефіцієнти регресії, можемо використати їх для розрахунку оптимальних витрат та цін на працю та капітал.
# Задані ціни ресурсів (праці та капіталу)
price_L = 10  # Ціна праці (ваші дані)
price_K = 20  # Ціна капіталу (ваші дані)

# Розрахунок множників Лагранжа
lambda_L = alpha / price_L
lambda_K = beta / price_K

# Оптимальні витрати виробничих факторів
optimal_L = lambda_L / A
optimal_K = lambda_K / A

# Ціни ресурсів (витрати на працю та капітал)
resource_price_L = lambda_L / optimal_L
resource_price_K = lambda_K / optimal_K

# Ціна та обсяг продукції
price_Q = A
quantity_Q = np.exp(regression.predict([[1, np.mean(log_L), np.mean(log_K)]]))

# Вивід результатів
print("Оптимальні витрати на працю та капітал:", optimal_L, optimal_K)
print("Ціна праці та капіталу:", resource_price_L, resource_price_K)
print("Ціна та обсяг продукції:", price_Q, quantity_Q)


# Розрахунок множників Лагранжа
lambda_L = alpha / price_L
lambda_K = beta / price_K

# Задані ціни ресурсів для короткострокового та довгострокового періодів (довільні значення)
price_L_short_term = 15  # Ціна праці для короткострокового періоду
price_K_long_term = 25  # Ціна капіталу для довгострокового періоду

# Розрахунок оптимальних витрат виробничих факторів в короткостроковому періоді
optimal_L_short_term = lambda_L / price_L_short_term
optimal_K_short_term = lambda_K / price_K

# Розрахунок оптимальних витрат виробничих факторів в довгостроковому періоді
optimal_L_long_term = lambda_L / price_L
optimal_K_long_term = lambda_K / price_K_long_term

# Вивід результатів
print("Оптимальні витрати на працю та капітал у короткостроковому періоді:", optimal_L_short_term, optimal_K_short_term)
print("Оптимальні витрати на працю та капітал у довгостроковому періоді:", optimal_L_long_term, optimal_K_long_term)
