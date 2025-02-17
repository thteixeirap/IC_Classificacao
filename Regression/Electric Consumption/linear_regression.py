import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from ucimlrepo import fetch_ucirepo

# Configurações globais
plt.style.use('seaborn')
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def load_data():
    """
    Carrega e prepara os dados iniciais
    """
    # Fetch dataset
    dataset = fetch_ucirepo(id=235)
    X = dataset.data.features
    
    print("Informações iniciais do dataset:")
    print("\nVariáveis disponíveis:")
    print(X.columns)
    print("\nPrimeiras linhas:")
    print(X.head())
    print("\nInformações gerais:")
    print(X.info())
    
    return X

def preprocess_data(X):
    """
    Realiza o pré-processamento dos dados
    """
    # Convertendo DateTime
    X['DateTime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'])
    X = X.drop(['Date', 'Time'], axis=1)
    
    # Verificando e tratando valores ausentes
    print("\nValores ausentes antes do tratamento:")
    print(X.isnull().sum())
    
    X = X.dropna()
    
    # Preparando features e target
    y = X['Global_active_power']
    X_features = X.drop(['Global_active_power', 'DateTime'], axis=1)
    
    # Convertendo para numérico
    X_features = X_features.apply(pd.to_numeric, errors='coerce')
    X_features = X_features.dropna()
    y = y[X_features.index]
    
    return X_features, y

def analyze_data(X, y):
    """
    Realiza análise exploratória dos dados
    """
    # Estatísticas descritivas
    print("\nEstatísticas descritivas das features:")
    print(X.describe())
    
    # Correlações
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação das Features')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Distribuição da variável target
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True)
    plt.title('Distribuição da Global Active Power')
    plt.savefig(os.path.join(results_dir, 'target_distribution.png'))
    plt.close()

def train_model(X, y):
    """
    Treina o modelo e retorna as métricas de avaliação
    """
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Treinamento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Validação cruzada
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    return model, metrics, (y_test, y_pred), cv_rmse

def analyze_residuals(y_test, y_pred):
    """
    Realiza análise detalhada dos resíduos
    """
    residuals = y_test - y_pred
    
    # Criar figura com subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Distribuição dos resíduos
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title('Distribuição dos Resíduos')
    ax1.set_xlabel('Resíduo')
    ax1.set_ylabel('Contagem')
    
    # 2. Resíduos vs Valores Preditos
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Resíduos vs Valores Preditos')
    ax2.set_xlabel('Valores Preditos')
    ax2.set_ylabel('Resíduos')
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot dos Resíduos')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'residuals_analysis.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print("\nEstatísticas dos Resíduos:")
    print(f"Média: {np.mean(residuals):.4f}")
    print(f"Desvio Padrão: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")

def plot_predictions(y_test, y_pred):
    """
    Plota gráficos de predição
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'predictions.png'))
    plt.close()

def main():
    # Carregar dados
    data = load_data()
    
    # Pré-processamento
    X, y = preprocess_data(data)
    
    # Análise exploratória
    analyze_data(X, y)
    
    # Treinamento e avaliação
    model, metrics, predictions, cv_rmse = train_model(X, y)
    
    # Mostrar métricas
    print("\nMétricas de Avaliação:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"\nValidação Cruzada (10-fold):")
    print(f"RMSE médio: {cv_rmse.mean():.4f}")
    print(f"Desvio padrão RMSE: {cv_rmse.std():.4f}")
    
    # Análise de resíduos
    analyze_residuals(predictions[0], predictions[1])
    
    # Plot de predições
    plot_predictions(predictions[0], predictions[1])

if __name__ == "__main__":
    main()
