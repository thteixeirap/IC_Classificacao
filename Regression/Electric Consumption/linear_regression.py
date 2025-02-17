import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Importando a base de dados
from ucimlrepo import fetch_ucirepo

# Fetch dataset
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

# Data (as pandas dataframes)
X = individual_household_electric_power_consumption.data.features

# Verificando as variáveis
print("Variáveis disponíveis:")
print(X.columns)

# Verificando as primeiras linhas da base de dados
print("\nPrimeiras linhas da base de dados:")
print(X.head())

# Verificando informações sobre a base de dados
print("\nInformações sobre a base de dados:")
print(X.info())

# Pré-processamento dos dados
# Convertendo 'Date' e 'Time' para datetime
X['DateTime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'])
X = X.drop(['Date', 'Time'], axis=1)  # Removendo as colunas 'Date' e 'Time'

# Verificando valores ausentes
print("\nValores ausentes na base de dados:")
print(X.isnull().sum())

# Removendo valores ausentes (se houver)
X = X.dropna()

# Definindo features (X) e target (y)
# Vamos prever 'Global_active_power' com base nas outras variáveis
y = X['Global_active_power']  # Target
X = X.drop(['Global_active_power', 'DateTime'], axis=1)  # Features (removendo 'DateTime' e 'Global_active_power')

# Convertendo todas as colunas para numéricas (caso haja problemas de tipo)
X = X.apply(pd.to_numeric, errors='coerce')

# Verificando e removendo valores nulos após a conversão
X = X.dropna()
y = y[X.index]

# Normalizando as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criando o modelo de Regressão Linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\nMSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# Validação cruzada com 10 folds
cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print(f'\nRMSE médio da validação cruzada com 10 folds: {cv_rmse_scores.mean():.4f}')
print(f'Desvio padrão do RMSE: {cv_rmse_scores.std():.4f}')

# Visualização 1: Gráfico de dispersão entre valores reais e previstos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reais (Global_active_power)')
plt.ylabel('Valores Previstos (Global_active_power)')
plt.title('Valores Reais vs. Previstos')
plt.show()

# Visualização 2: Distribuição dos erros
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Resíduos (Erros)')
plt.ylabel('Frequência')
plt.title('Distribuição dos Resíduos')
plt.show()