import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ucimlrepo import fetch_ucirepo

# Configurações globais
plt.style.use('seaborn')
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def load_data():
    """
    Carrega os dados diretamente do repositório UCI
    """
    dataset = fetch_ucirepo(id=235)
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
    # Criar coluna DateTime
    X['DateTime'] = pd.to_datetime(X['Date'] + ' ' + X['Time'])
    
    # Verificando e tratando valores ausentes
    print("\nValores ausentes antes do tratamento:")
    print(X.isnull().sum())
    
    X = X.dropna()
    
    # Preparando features e target
    y = pd.to_numeric(X['Global_active_power'], errors='coerce')
    X_features = X.drop(['Global_active_power', 'DateTime', 'Date', 'Time'], axis=1)
    
    # Convertendo para numérico explicitamente
    for col in X_features.columns:
        X_features[col] = pd.to_numeric(X_features[col], errors='coerce')
    
    # Removendo linhas com valores NA após conversão
    mask = ~(X_features.isna().any(axis=1) | y.isna())
    X_features = X_features[mask]
    y = y[mask]
    
    return X_features, y

def plot_feature_importance(model, feature_names):
    """
    Plota a importância das features do XGBoost
    """
    # Garantir que o diretório existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Pegar importância das features
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importance_df['importance'])
    plt.xticks(range(len(importances)), importance_df['feature'], rotation=45)
    plt.title('Importância das Features (XGBoost)')
    plt.xlabel('Features')
    plt.ylabel('Importância')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()

def analyze_residuals(y_test, y_pred, model_name="XGBoost"):
    """
    Realiza análise detalhada dos resíduos
    """
    residuals = y_test - y_pred
    
    # Criar figura com subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Distribuição dos resíduos
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title(f'Distribuição dos Resíduos - {model_name}')
    ax1.set_xlabel('Resíduo')
    ax1.set_ylabel('Contagem')
    
    # 2. Resíduos vs Valores Preditos
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(f'Resíduos vs Valores Preditos - {model_name}')
    ax2.set_xlabel('Valores Preditos')
    ax2.set_ylabel('Resíduos')
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot dos Resíduos - {model_name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_residuals.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print(f"\nEstatísticas dos Resíduos - {model_name}:")
    print(f"Média: {np.mean(residuals):.4f}")
    print(f"Desvio Padrão: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")

def train_model(X, y, n_repetitions=10):
    """
    Treina o XGBoost n_repetitions vezes e retorna as métricas
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
        
        # Treinamento - XGBoost com parâmetros otimizados
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
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
        
        # Se for a última iteração, salvar as predições e plotar importância
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
    
    # Boxplot das métricas
    plt.figure(figsize=(10, 6))
    plt.boxplot([rmse_values, r2_values], labels=['RMSE', 'R²'])
    plt.title('Distribuição das Métricas nas 10 Repetições')
    plt.savefig(os.path.join(results_dir, 'metrics_boxplot.png'))
    plt.close()
    
    return model, avg_metrics, final_predictions, all_metrics

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
    print("\nIniciando treinamento do XGBoost...")
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
    
    # Plot de predições da última execução
    print("\nGerando gráficos de predições...")
    plot_predictions(predictions[0], predictions[1])
    
    print("\nAnálise completa! Os resultados foram salvos na pasta 'results'")

if __name__ == "__main__":
    main()