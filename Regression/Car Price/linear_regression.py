import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configurações globais
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

# Criar diretório para resultados
results_dir = 'Regression/Car Price/results_linear_regression'
os.makedirs(results_dir, exist_ok=True)

def carregar_dados():
    """
    Carrega e realiza análise exploratória dos dados
    """
    df = pd.read_csv('Regression/Car Price/car_price_dataset.csv')
    
    print("\nInformações do Dataset:")
    print(df.info())
    
    print("\nEstatísticas Descritivas:")
    print(df.describe())
    
    # Análise de variáveis categóricas
    categoricas = ['Brand', 'Fuel_Type', 'Transmission']
    for col in categoricas:
        print(f"\nDistribuição de {col}:")
        print(df[col].value_counts())
        
        # Gráficos de contagem
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'lr_distribuicao_{col.lower()}.png'))
        plt.close()
    
    # Boxplot do preço
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df['Price'])
    plt.title('Distribuição do Preço')
    plt.savefig(os.path.join(results_dir, 'lr_preco_boxplot.png'))
    plt.close()
    
    # Scatter plot: Ano vs Preço
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Year', y='Price')
    plt.title('Relação entre Ano e Preço')
    plt.savefig(os.path.join(results_dir, 'lr_ano_preco_scatter.png'))
    plt.close()
    
    # Matriz de correlação
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numericas].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação - Variáveis Numéricas')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lr_matriz_correlacao.png'))
    plt.close()
    
    return df

def preprocessar_dados(df):
    """
    Prepara os dados para modelagem
    """
    # Tratamento de outliers no preço
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    df_limpo = df[(df['Price'] >= limite_inferior) & (df['Price'] <= limite_superior)]
    print(f"\nRemovidos {len(df) - len(df_limpo)} outliers de preço")
    
    # Converter variáveis categóricas
    categoricas = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
    df_encoded = pd.get_dummies(df_limpo, columns=categoricas)
    
    # Separar features e target
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    
    return X, y

def analisar_residuos(y_test, y_pred):
    """
    Plota análise detalhada dos resíduos
    """
    residuos = y_test - y_pred
    
    # Criar figura com subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Distribuição dos resíduos
    sns.histplot(residuos, kde=True, ax=ax1)
    ax1.set_title('Distribuição dos Resíduos')
    ax1.set_xlabel('Resíduo')
    ax1.set_ylabel('Contagem')
    
    # 2. Resíduos vs Valores Preditos
    sns.scatterplot(x=y_pred, y=residuos, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Resíduos vs Valores Preditos')
    ax2.set_xlabel('Valores Preditos')
    ax2.set_ylabel('Resíduos')
    
    # 3. Q-Q plot
    stats.probplot(residuos, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot dos Resíduos')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lr_analise_residuos.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print("\nEstatísticas dos Resíduos:")
    print(f"Média: {np.mean(residuos):.2f}")
    print(f"Desvio Padrão: {np.std(residuos):.2f}")
    print(f"Skewness: {stats.skew(residuos):.2f}")
    print(f"Kurtosis: {stats.kurtosis(residuos):.2f}")

def treinar_modelo(X, y, n_repeticoes=30):
    """
    Treina e avalia o modelo de Regressão Linear com múltiplas repetições
    """
    todas_metricas = []

    for i in range(n_repeticoes):
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treinar modelo
        modelo = LinearRegression()
        modelo.fit(X_train_scaled, y_train)

        # Fazer predições
        y_pred_test = modelo.predict(X_test_scaled)

        # Calcular métricas
        metricas = {
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        todas_metricas.append(metricas)

    # Cálculo das métricas médias
    metricas_medias = {
        'rmse_mean': np.mean([m['rmse'] for m in todas_metricas]),
        'r2_mean': np.mean([m['r2'] for m in todas_metricas])
    }

    # Criar figura
    plt.figure(figsize=(12, 5))

    # RMSE ao longo das repetições
    plt.subplot(1, 2, 1)
    rmse_values = [m['rmse'] for m in todas_metricas]
    plt.plot(range(1, n_repeticoes + 1), rmse_values, 'b-', label='RMSE')
    plt.axhline(y=metricas_medias['rmse_mean'], color='r', linestyle='--', label=f'Média: {metricas_medias["rmse_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('RMSE')
    plt.title('RMSE ao longo das repetições')
    plt.legend()

    # R² ao longo das repetições
    plt.subplot(1, 2, 2)
    r2_values = [m['r2'] for m in todas_metricas]
    plt.plot(range(1, n_repeticoes + 1), r2_values, 'g-', label='R²')
    plt.axhline(y=metricas_medias['r2_mean'], color='r', linestyle='--', label=f'Média: {metricas_medias["r2_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('R²')
    plt.title('R² ao longo das repetições')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metricas_repeticoes.png'))
    plt.close()

    print("\nMédias das métricas ao longo das repetições:")
    print(f"RMSE Médio: {metricas_medias['rmse_mean']:.2f}")
    print(f"R² Médio: {metricas_medias['r2_mean']:.4f}")

    return modelo, metricas_medias


def main():
    print("Carregando e preparando dados...")
    df = carregar_dados()
    X, y = preprocessar_dados(df)
    
    print("\nIniciando treinamento...")
    modelo, metricas = treinar_modelo(X, y)
    
    print("\nAnálise completa! Os resultados foram salvos na pasta 'resultados'")

if __name__ == "__main__":
    main()