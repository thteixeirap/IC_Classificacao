import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Carregando o dataset Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target  # Adicionando a coluna de target (espécies)

# Verificando as primeiras linhas do dataset
print(df.head())

# Verificando informações básicas do dataset
print(df.info())

# Verificando se há valores nulos
print(df.isnull().sum())

# Identificando colunas numéricas
numeric_columns = iris.feature_names  # Todas as colunas são numéricas
categorical_columns = []  # Não há colunas categóricas neste dataset

# Convertendo colunas numéricas para float (caso necessário)
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Codificando a variável alvo (Species: 0 = Setosa, 1 = Versicolor, 2 = Virginica)
df['Species'] = df['Species'].astype(int)

# Visualização 1: Distribuição das espécies
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Species')
plt.title('Distribuição das Espécies de Iris')
plt.xlabel('Espécie (0: Setosa, 1: Versicolor, 2: Virginica)')
plt.ylabel('Quantidade')
plt.show()

# Visualização 2: Boxplot de Comprimento da Sépala por espécie
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Species', y='sepal length (cm)')
plt.title('Distribuição do Comprimento da Sépala por Espécie')
plt.xlabel('Espécie (0: Setosa, 1: Versicolor, 2: Virginica)')
plt.ylabel('Comprimento da Sépala (cm)')
plt.show()

# Separando features e target
X = df[numeric_columns]
y = df['Species']

# Verificando e removendo quaisquer linhas com valores nulos após as conversões
X = X.dropna()
y = y[X.index]

# Normalizando as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo Naive Bayes
nb = GaussianNB()

# Avaliando o modelo com validação cruzada (10 folds)
cv_scores = cross_val_score(nb, X_scaled, y, cv=10, scoring='accuracy')

# Exibindo os resultados da validação cruzada
print("\nResultados da Validação Cruzada (10 folds):")
print(f"Acurácia média: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")
print(f"Scores individuais: {cv_scores}")

# Treinando o modelo final com todos os dados de treino
nb.fit(X_train, y_train)

# Fazendo previsões
y_pred = nb.predict(X_test)

# Visualização 3: Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Função para fazer previsões com novos dados
def predict_iris_species(model, scaler, new_data):
    """
    Faz previsão de espécies de Iris para novos dados
    
    Args:
        model: Modelo Naive Bayes treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0: Setosa, 1: Versicolor, 2: Virginica)
    """
    # Garantindo que as colunas estejam na mesma ordem do treinamento
    new_data = new_data[numeric_columns]
    
    # Normalizando os novos dados
    scaled_data = scaler.transform(new_data)
    
    # Fazendo a previsão
    prediction = model.predict(scaled_data)
    
    return prediction

# Exemplo de como fazer uma previsão com novos dados
"""
novo_dado = pd.DataFrame({
    'sepal length (cm)': [5.1],
    'sepal width (cm)': [3.5],
    'petal length (cm)': [1.4],
    'petal width (cm)': [0.2]
})

prediction = predict_iris_species(nb, scaler, novo_dado)
print(f"Previsão para o novo dado: {iris.target_names[prediction[0]]}")
"""