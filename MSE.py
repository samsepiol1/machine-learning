from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dados de exemplo
X = [[1], [2], [3], [4], [5]]  # Variáveis independentes
y = [2, 4, 6, 8, 10]  # Variável dependente (rótulos)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação e treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o modelo usando a métrica de erro quadrático médio, previsão para erro (Mean Squared Error - MSE)
mse = mean_squared_error(y_test, y_pred)

print("Erro quadrático médio (MSE):", mse)

# Fazendo uma previsão com um novo valor
new_X = [[5]]
new_y = model.predict(new_X)
print("Previsão para X =", new_X[0][0], ":", new_y[0])