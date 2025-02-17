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

# Configurações globais
# Configurações de estilo do matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
results_dir = 'Regression/Loan Approval/results_xgboost'
os.makedirs(results_dir, exist_ok=True)

def carregar_dados():
    """
    Carrega os dados do arquivo CSV
    """
    df = pd.read_csv('loan_approval_dataset.csv')
    
    print("Informações iniciais do dataset:")
    print("\nVariáveis disponíveis:")
    print(df.columns)
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nInformações gerais:")
    print(df.info())
    
    return df

def preprocessar_dados(df):
    """
    Realiza o pré-processamento dos dados
    """
    # Limpar nomes das colunas (remover espaços extras)
    df.columns = df.columns.str.strip()
    
    # Limpar valores das colunas string (remover espaços extras)
    for col in ['education', 'self_employed', 'loan_status']:
        df[col] = df[col].str.strip()
    
    # Convertendo variáveis categóricas
    mapa_education = {
        'Graduate': 1,
        'Not Graduate': 0
    }
    
    mapa_self_employed = {
        'Yes': 1,
        'No': 0
    }
    
    mapa_loan_status = {
        'Approved': 1,
        'Rejected': 0
    }
    
    # Aplicando os mapeamentos
    df['education'] = df['education'].map(mapa_education)
    df['self_employed'] = df['self_employed'].map(mapa_self_employed)
    df['loan_status'] = df['loan_status'].map(mapa_loan_status)
    
    # Verificando valores ausentes
    print("\nValores ausentes:")
    print(df.isnull().sum())
    
    # Verificar valores únicos nas colunas categóricas
    print("\nValores únicos em education:", df['education'].unique())
    print("Valores únicos em self_employed:", df['self_employed'].unique())
    print("Valores únicos em loan_status:", df['loan_status'].unique())
    
    # Separando features e target
    X = df.drop(['loan_status', 'loan_id'], axis=1)
    y = df['loan_status']
    
    return X, y, df

def analisar_dados(X, y, df):
    """
    Realiza análise exploratória dos dados
    """
    # Estatísticas descritivas
    print("\nEstatísticas descritivas das features:")
    print(X.describe())
    
    # Matriz de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação das Features')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'xgb_correlacao_features.png'))
    plt.close()
    
    # Distribuição da variável target
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='loan_status')
    plt.title('Distribuição da Variável Target (Aprovação de Empréstimo)')
    plt.xlabel('Status do Empréstimo')
    plt.ylabel('Contagem')
    plt.savefig(os.path.join(results_dir, 'xgb_distribuicao_target.png'))
    plt.close()
    
    # Distribuição das features numéricas
    for coluna in X.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(X[coluna], kde=True)
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Contagem')
        plt.savefig(os.path.join(results_dir, f'xgb_distribuicao_{coluna}.png'))
        plt.close()

def analisar_residuos(y_test, y_pred):
    """
    Realiza análise detalhada dos resíduos
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
    plt.savefig(os.path.join(results_dir, 'xgb_analise_residuos.png'))
    plt.close()
    
    # Estatísticas dos resíduos
    print("\nEstatísticas dos Resíduos:")
    print(f"Média: {np.mean(residuos):.4f}")
    print(f"Desvio Padrão: {np.std(residuos):.4f}")
    print(f"Skewness: {stats.skew(residuos):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuos):.4f}")

def plotar_importancia_features(modelo, X):
    """
    Plota a importância das features do XGBoost
    """
    importancias = pd.DataFrame({
        'Feature': X.columns,
        'Importância': modelo.feature_importances_
    }).sort_values('Importância', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(y=range(len(importancias)), width=importancias['Importância'])
    plt.yticks(range(len(importancias)), importancias['Feature'])
    plt.xlabel('Importância')
    plt.title('Importância das Features no XGBoost')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'xgb_importancia_features.png'))
    plt.close()
    
    return importancias

def treinar_modelo(X, y, n_repeticoes=10):
    """
    Treina o modelo n_repeticoes vezes e retorna as métricas
    """
    todas_metricas = []
    
    for i in range(n_repeticoes):
        print(f"\nExecutando repetição {i+1}/{n_repeticoes}")
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=i)
        
        # Treinamento
        modelo = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=i
        )
        modelo.fit(X_train, y_train)
        
        # Predições
        y_pred = modelo.predict(X_test)
        
        # Métricas
        metricas = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        todas_metricas.append(metricas)
        
        # Se for a última iteração, salvar as predições
        if i == n_repeticoes - 1:
            predicoes_finais = (y_test, y_pred)
            modelo_final = modelo
    
    # Calcular médias e desvios das métricas
    metricas_medias = {
        'mse_mean': np.mean([m['mse'] for m in todas_metricas]),
        'mse_std': np.std([m['mse'] for m in todas_metricas]),
        'rmse_mean': np.mean([m['rmse'] for m in todas_metricas]),
        'rmse_std': np.std([m['rmse'] for m in todas_metricas]),
        'mae_mean': np.mean([m['mae'] for m in todas_metricas]),
        'mae_std': np.std([m['mae'] for m in todas_metricas]),
        'r2_mean': np.mean([m['r2'] for m in todas_metricas]),
        'r2_std': np.std([m['r2'] for m in todas_metricas])
    }
    
    # Plot da distribuição das métricas
    plt.figure(figsize=(15, 5))
    
    # RMSE ao longo das repetições
    plt.subplot(1, 2, 1)
    rmse_values = [m['rmse'] for m in todas_metricas]
    plt.plot(range(1, n_repeticoes + 1), rmse_values, 'b-')
    plt.axhline(y=metricas_medias['rmse_mean'], color='r', linestyle='--', 
                label=f'Média: {metricas_medias["rmse_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('RMSE')
    plt.title('RMSE ao longo das repetições')
    plt.legend()
    
    # R² ao longo das repetições
    plt.subplot(1, 2, 2)
    r2_values = [m['r2'] for m in todas_metricas]
    plt.plot(range(1, n_repeticoes + 1), r2_values, 'g-')
    plt.axhline(y=metricas_medias['r2_mean'], color='r', linestyle='--', 
                label=f'Média: {metricas_medias["r2_mean"]:.2f}')
    plt.xlabel('Repetição')
    plt.ylabel('R²')
    plt.title('R² ao longo das repetições')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'xgb_metricas_repeticoes.png'))
    plt.close()
    
    # Boxplot das métricas
    plt.figure(figsize=(10, 6))
    plt.boxplot([rmse_values, r2_values], labels=['RMSE', 'R²'])
    plt.title('Distribuição das Métricas nas Repetições')
    plt.savefig(os.path.join(results_dir, 'xgb_boxplot_metricas.png'))
    plt.close()
    
    return modelo_final, metricas_medias, predicoes_finais, todas_metricas

def plotar_predicoes(y_test, y_pred):
    """
    Plota gráficos de predição
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'xgb_predicoes.png'))
    plt.close()

def main():
    """
    Função principal que executa todo o pipeline de análise
    """
    # Carregar dados
    print("Carregando dados...")
    df = carregar_dados()
    
    # Pré-processamento
    print("\nRealizando pré-processamento...")
    X, y, df = preprocessar_dados(df)
    
    # Análise exploratória
    print("\nRealizando análise exploratória...")
    analisar_dados(X, y, df)
    
    # Treinamento e avaliação
    print("\nIniciando treinamento do modelo...")
    modelo, metricas_medias, predicoes, todas_metricas = treinar_modelo(X, y)
    
    # Mostrar métricas médias
    print("\nMétricas médias das repetições:")
    print(f"MSE médio: {metricas_medias['mse_mean']:.4f} (±{metricas_medias['mse_std']:.4f})")
    print(f"RMSE médio: {metricas_medias['rmse_mean']:.4f} (±{metricas_medias['rmse_std']:.4f})")
    print(f"MAE médio: {metricas_medias['mae_mean']:.4f} (±{metricas_medias['mae_std']:.4f})")
    print(f"R² médio: {metricas_medias['r2_mean']:.4f} (±{metricas_medias['r2_std']:.4f})")
    
    # Análise de resíduos da última execução
    print("\nRealizando análise de resíduos...")
    analisar_residuos(predicoes[0], predicoes[1])
    
    # Plot de predições da última execução
    print("\nGerando gráficos de predições...")
    plotar_predicoes(predicoes[0], predicoes[1])
    
    # Plotar e mostrar importância das features
    print("\nAnalisando importância das features...")
    importancias = plotar_importancia_features(modelo, X)
    print("\nImportância das features:")
    print(importancias)
    
    print("\nAnálise completa! Os resultados foram salvos na pasta 'resultados'")

if __name__ == "__main__":
    main()