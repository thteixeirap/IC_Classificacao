import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ucimlrepo import fetch_ucirepo

# Configurações globais
plt.style.use('seaborn')
results_dir = 'Regression/Electric Consumption/results_random_forest'
os.makedirs(results_dir, exist_ok=True)

def load_data():
    """
    Carrega os dados diretamente do repositório UCI
    """
    # Fetch dataset 235: Individual household electric power consumption
    dataset = fetch_ucirepo(id=235) 
    
    # Data (as pandas dataframes)
    data = dataset.data.features
    
    print("Informações iniciais do dataset:")
    print("\nVariáveis disponíveis:")
    print(data.columns)
    print("\nPrimeiras linhas:")
    print(data.head())
    print("\nInformações gerais:")
    print(data.info())
    
    return data

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
    
    # Convertendo para numérico explicitamente
    for col in X_features.columns:
        X_features[col] = pd.to_numeric(X_features[col], errors='coerce')
    
    y = pd.to_numeric(y, errors='coerce')
    
    # Removendo linhas com valores NA após conversão
    mask = ~(X_features.isna().any(axis=1) | y.isna())
    X_features = X_features[mask]
    y = y[mask]
    
    return X_features, y

def plot_feature_importance(model, feature_names):
    """
    Plota a importância das features do Random Forest
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Importância das Features')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

def train_model(X, y, n_repetitions=30):
    """
    Treina o Random Forest n_repetitions vezes e retorna as métricas
    """
    all_metrics = []
    feature_names = X.columns
    
    for i in range(n_repetitions):
        print(f"\nExecutando repetição {i+1}/{n_repetitions}")
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=i)
        
        # Treinamento - Random Forest com parâmetros otimizados
        model = RandomForestRegressor(
            n_estimators=100,  # número de árvores
            max_depth=10,      # profundidade máxima das árvores
            min_samples_split=5,  # mínimo de amostras para split
            min_samples_leaf=2,   # mínimo de amostras em cada folha
            random_state=i
        )
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
        
        all_metrics.append(metrics)
        
        # Se for a última iteração, salvar as predições e plotar importância das features
        if i == n_repetitions - 1:
            final_predictions = (y_test, y_pred)
            plot_feature_importance(model, feature_names)

    # Calcular médias e desvios das métricas
    avg_metrics = {
        'mse_mean': np.mean([m['mse'] for m in all_metrics]),
        'mse_std': np.std([m['mse'] for m in all_metrics]),
        'rmse_mean': np.mean([m['rmse'] for m in all_metrics]),
        'rmse_std': np.std([m['rmse'] for m in all_metrics]),
        'mae_mean': np.mean([m['mae'] for m in all_metrics]),
        'mae_std': np.std([m['mae'] for m in all_metrics]),
        'r2_mean': np.mean([m['r2'] for m in all_metrics]),
        'r2_std': np.std([m['r2'] for m in all_metrics])
    }
    
    # Plot da distribuição das métricas
    plt.figure(figsize=(15, 5))
    
    # RMSE ao longo das repetições
    plt.subplot(1, 2, 1)
    rmse_values = [m['rmse'] for m in all_metrics]
    plt.plot(range(1, n_repetitions + 1), rmse_values, 'b-')
    plt.axhline(y=avg_metrics['rmse_mean'], color='r', linestyle='--', 
                label=f'Média: {avg_metrics["rmse_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('RMSE')
    plt.title('RMSE ao longo das repetições')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    r2_values = [m['r2'] for m in all_metrics]
    plt.plot(range(1, n_repetitions + 1), r2_values, 'g-')
    plt.axhline(y=avg_metrics['r2_mean'], color='r', linestyle='--', 
                label=f'Média: {avg_metrics["r2_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('R²')
    plt.title('R² ao longo das repetições')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_over_runs.png'))
    plt.close()
    
    return model, avg_metrics, final_predictions, all_metrics

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

def main():
    """
    Função principal que executa todo o pipeline de análise
    """
    # Carregar dados
    print("Carregando dados...")
    data = load_data()
    
    # Pré-processamento
    print("\nRealizando pré-processamento...")
    X, y = preprocess_data(data)
    
    # Treinamento e avaliação
    print("\nIniciando treinamento do Random Forest...")
    model, avg_metrics, predictions, all_metrics = train_model(X, y)
    
    # Mostrar métricas médias
    print("\nMétricas médias das 10 repetições:")
    print(f"MSE médio: {avg_metrics['mse_mean']:.4f} (±{avg_metrics['mse_std']:.4f})")
    print(f"RMSE médio: {avg_metrics['rmse_mean']:.4f} (±{avg_metrics['rmse_std']:.4f})")
    print(f"MAE médio: {avg_metrics['mae_mean']:.4f} (±{avg_metrics['mae_std']:.4f})")
    print(f"R² médio: {avg_metrics['r2_mean']:.4f} (±{avg_metrics['r2_std']:.4f})")
    
    # Análise de resíduos da última execução
    print("\nRealizando análise de resíduos...")
    analyze_residuals(predictions[0], predictions[1])
    
    print("\nAnálise completa! Os resultados foram salvos na pasta 'results'")

if __name__ == "__main__":
    main()