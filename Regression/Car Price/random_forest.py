import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configurações globais
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

# Criar diretório para resultados
results_dir = 'Regression/Car Price/results_random_forest'
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
        plt.savefig(os.path.join(results_dir, f'rf_distribuicao_{col.lower()}.png'))
        plt.close()
    
    # Boxplot do preço
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df['Price'])
    plt.title('Distribuição do Preço')
    plt.savefig(os.path.join(results_dir, 'rf_preco_boxplot.png'))
    plt.close()
    
    # Scatter plot: Ano vs Preço
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Year', y='Price')
    plt.title('Relação entre Ano e Preço')
    plt.savefig(os.path.join(results_dir, 'rf_ano_preco_scatter.png'))
    plt.close()
    
    # Matriz de correlação
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numericas].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação - Variáveis Numéricas')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rf_matriz_correlacao.png'))
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
    plt.savefig(os.path.join(results_dir, 'rf_analise_residuos.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print("\nEstatísticas dos Resíduos:")
    print(f"Média: {np.mean(residuos):.2f}")
    print(f"Desvio Padrão: {np.std(residuos):.2f}")
    print(f"Skewness: {stats.skew(residuos):.2f}")
    print(f"Kurtosis: {stats.kurtosis(residuos):.2f}")

def plotar_importancia_features(modelo, X, n_top=20):
    """
    Plota a importância das features do Random Forest
    """
    importancias = pd.DataFrame({
        'Feature': X.columns,
        'Importância': modelo.feature_importances_
    }).sort_values('Importância', ascending=True)
    
    # Plotar as N features mais importantes
    plt.figure(figsize=(10, 8))
    plt.barh(y=range(min(n_top, len(importancias))), 
            width=importancias['Importância'][-n_top:],
            tick_label=importancias['Feature'][-n_top:])
    plt.title(f'Top {n_top} Features Mais Importantes - Random Forest')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rf_importancia_features.png'))
    plt.close()
    
    return importancias

def treinar_modelo(X, y):
    """
    Treina e avalia o modelo Random Forest
    """
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo
    print("\nTreinando Random Forest...")
    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    modelo.fit(X_train_scaled, y_train)
    
    # Fazer predições
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)
    
    # Calcular métricas
    metricas = {
        'r2_treino': r2_score(y_train, y_pred_train),
        'r2_teste': r2_score(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    print("\nMétricas do Modelo:")
    print(f"R² (Treino): {metricas['r2_treino']:.4f}")
    print(f"R² (Teste): {metricas['r2_teste']:.4f}")
    print(f"RMSE: {metricas['rmse']:.2f}")
    print(f"MAE: {metricas['mae']:.2f}")
    
    # Plotar predições
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Preço Real')
    plt.ylabel('Preço Previsto')
    plt.title('Preço Real vs Previsto - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rf_predicoes.png'))
    plt.close()
    
    # Análise de resíduos
    analisar_residuos(y_test, y_pred_test)
    
    # Análise de importância das features
    importancias = plotar_importancia_features(modelo, X)
    print("\nTop 10 Features mais importantes:")
    print(importancias.tail(10))
    
    return modelo, metricas

def main():
    print("Carregando e preparando dados...")
    df = carregar_dados()
    X, y = preprocessar_dados(df)
    
    print("\nIniciando treinamento...")
    modelo, metricas = treinar_modelo(X, y)
    
    print("\nAnálise completa! Os resultados foram salvos na pasta 'resultados'")

if __name__ == "__main__":
    main()