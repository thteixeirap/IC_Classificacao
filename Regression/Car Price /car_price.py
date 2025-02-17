import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Definir o caminho do dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'car_price_dataset.csv')  
results_dir = os.path.join(current_dir, 'results_car_price')
os.makedirs(results_dir, exist_ok=True)

def load_and_explore_data():
    """
    Carrega e realiza análise exploratória dos dados
    """
    # Carregar dados
    df = pd.read_csv(dataset_path)
    
    # Informações básicas
    print("\nInformações do Dataset:")
    print(df.info())
    
    print("\nEstatísticas Descritivas:")
    print(df.describe())
    
    # Contagem de valores para variáveis categóricas
    categorical_cols = ['Brand', 'Fuel_Type', 'Transmission']
    for col in categorical_cols:
        print(f"\nDistribuição de {col}:")
        print(df[col].value_counts())
        
        # Criar gráficos de contagem
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{col.lower()}_distribution.png'))
        plt.close()
    
    # Análises gráficas adicionais
    create_additional_plots(df)
    
    return df

def create_additional_plots(df):
    """
    Cria visualizações adicionais dos dados
    """
    # Boxplot do preço
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df['Price'])
    plt.title('Distribuição do Preço')
    plt.savefig(os.path.join(results_dir, 'price_boxplot.png'))
    plt.close()
    
    # Scatter plot: Year vs Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Year', y='Price')
    plt.title('Relação entre Ano e Preço')
    plt.savefig(os.path.join(results_dir, 'year_price_scatter.png'))
    plt.close()
    
    # Boxplot: Transmission vs Price
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Transmission', y='Price')
    plt.title('Preço por Tipo de Transmissão')
    plt.savefig(os.path.join(results_dir, 'transmission_price_boxplot.png'))
    plt.close()

def plot_detailed_residuals(y_true, y_pred, model_name):
    """
    Plota análise detalhada dos resíduos para um modelo específico
    """
    residuals = y_true - y_pred
    
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
    plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_detailed_residuals.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print(f"\nEstatísticas dos Resíduos - {model_name}:")
    print(f"Média: {np.mean(residuals):.2f}")
    print(f"Desvio Padrão: {np.std(residuals):.2f}")
    print(f"Skewness: {stats.skew(residuals):.2f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.2f}")

def prepare_data(df):
    """
    Prepara os dados para modelagem
    """
    # Tratar outliers de preço
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    print(f"\nRemovidos {len(df) - len(df_clean)} outliers de preço")
    
    # Converter variáveis categóricas
    categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
    df_encoded = pd.get_dummies(df_clean, columns=categorical_columns)
    
    # Separar features e target
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    
    return X, y

def train_and_evaluate_models(X, y):
    """
    Treina e avalia os modelos com análise detalhada de resíduos
    """
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Treinar e avaliar cada modelo
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Fazer predições
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calcular métricas
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"R² (Treino): {train_r2:.4f}")
        print(f"R² (Teste): {test_r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Preço Real')
        plt.ylabel('Preço Previsto')
        plt.title(f'Preço Real vs Previsto - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_predictions.png'))
        plt.close()
        
        # Plot análise detalhada de resíduos
        plot_detailed_residuals(y_test, y_pred_test, name)

def main():
    # Carregar e explorar dados
    df = load_and_explore_data()
    
    # Preparar dados
    X, y = prepare_data(df)
    
    # Treinar e avaliar modelos
    train_and_evaluate_models(X, y)

if __name__ == "__main__":
    main()