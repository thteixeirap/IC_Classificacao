import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from itertools import cycle

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

# Visualização 2: Relação entre comprimento e largura das sépalas com cor por espécie
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='Species', alpha=0.6)
plt.title('Relação entre Comprimento e Largura das Sépalas')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Largura da Sépala (cm)')
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

# Criando e treinando o modelo SVM
print("Treinando o modelo SVM (isso pode levar alguns minutos)...")
svm = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)

# Avaliando o modelo com validação cruzada (10 folds)
cv_scores = cross_val_score(svm, X_scaled, y, cv=10, scoring='accuracy')

# Exibindo os resultados da validação cruzada
print("\nResultados da Validação Cruzada (10 folds):")
print(f"Acurácia média: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")
print(f"Scores individuais: {cv_scores}")

# Treinando o modelo final com todos os dados de treino
svm.fit(X_train, y_train)

# Fazendo previsões
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)

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

# Curva ROC para cada classe (multiclasse)
# Binarizando as classes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Calculando a curva ROC e a AUC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva ROC para cada classe
plt.figure(figsize=(10, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Curva ROC da classe {iris.target_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Multiclasse')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Função para fazer previsões com novos dados
def predict_iris_species(model, scaler, new_data):
    """
    Faz previsão de espécies de Iris para novos dados
    
    Args:
        model: Modelo SVM treinado
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

prediction = predict_iris_species(svm, scaler, novo_dado)
print(f"Previsão para o novo dado: {iris.target_names[prediction[0]]}")
"""