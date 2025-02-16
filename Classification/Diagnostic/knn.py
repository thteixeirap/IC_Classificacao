import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Carregando o dataset Breast Cancer Wisconsin (Diagnostic)
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['Diagnosis'] = cancer.target  # Adicionando a coluna de target (diagnóstico: 0 = maligno, 1 = benigno)

# Verificando as primeiras linhas do dataset
print(df.head())

# Verificando informações básicas do dataset
print(df.info())

# Verificando se há valores nulos
print(df.isnull().sum())

# Identificando colunas numéricas e categóricas
numeric_columns = cancer.feature_names  # Todas as colunas são numéricas
categorical_columns = []  # Não há colunas categóricas neste dataset

# Convertendo colunas numéricas para float (caso necessário)
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Codificando a variável alvo (Diagnosis: 0 = Maligno, 1 = Benigno)
df['Diagnosis'] = df['Diagnosis'].astype(int)

# Visualização 1: Distribuição dos diagnósticos
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Diagnosis')
plt.title('Distribuição dos Diagnósticos de Câncer de Mama')
plt.xlabel('Diagnóstico (0: Maligno, 1: Benigno)')
plt.ylabel('Quantidade')
plt.show()

# Visualização 2: Relação entre duas características com cor por diagnóstico
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='Diagnosis', alpha=0.6)
plt.title('Relação entre Raio Médio e Textura Média')
plt.xlabel('Raio Médio')
plt.ylabel('Textura Média')
plt.show()

# Separando features e target
X = df[numeric_columns]  # Usando apenas as colunas numéricas
y = df['Diagnosis']

# Verificando e removendo quaisquer linhas com valores nulos após as conversões
X = X.dropna()
y = y[X.index]

# Normalizando as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Testando diferentes valores de k
k_values = range(1, 21, 2)
train_scores = []
test_scores = []
cv_scores = []  # Lista para armazenar os scores da validação cruzada

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Treinando e avaliando no conjunto de treino e teste
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
    
    # Avaliando com validação cruzada (10 folds)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Visualização 3: Gráfico de desempenho para diferentes valores de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Treino')
plt.plot(k_values, test_scores, 'o-', label='Teste')
plt.plot(k_values, cv_scores, 'o-', label='Validação Cruzada (10 folds)')
plt.title('Acurácia do Modelo vs. Número de Vizinhos (k)')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

# Treinando o modelo final com o melhor k (baseado na validação cruzada)
best_k = k_values[np.argmax(cv_scores)]
print(f"\nMelhor valor de k encontrado: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Fazendo previsões
y_pred = knn.predict(X_test)

# Visualização 4: Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Visualização 5: Curva ROC
y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva (benigno)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Função para fazer previsões com novos dados
def predict_cancer_diagnosis(model, scaler, new_data):
    """
    Faz previsão de diagnóstico de câncer para novos dados
    
    Args:
        model: Modelo KNN treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0: Maligno, 1: Benigno)
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
    'mean radius': [17.99],
    'mean texture': [10.38],
    'mean perimeter': [122.8],
    'mean area': [1001.0],
    'mean smoothness': [0.1184],
    'mean compactness': [0.2776],
    'mean concavity': [0.3001],
    'mean concave points': [0.1471],
    'mean symmetry': [0.2419],
    'mean fractal dimension': [0.07871],
    'radius error': [1.095],
    'texture error': [0.9053],
    'perimeter error': [8.589],
    'area error': [153.4],
    'smoothness error': [0.006399],
    'compactness error': [0.04904],
    'concavity error': [0.05373],
    'concave points error': [0.01587],
    'symmetry error': [0.03003],
    'fractal dimension error': [0.006193],
    'worst radius': [25.38],
    'worst texture': [17.33],
    'worst perimeter': [184.6],
    'worst area': [2019.0],
    'worst smoothness': [0.1622],
    'worst compactness': [0.6656],
    'worst concavity': [0.7119],
    'worst concave points': [0.2654],
    'worst symmetry': [0.4601],
    'worst fractal dimension': [0.1189]
})

prediction = predict_cancer_diagnosis(knn, scaler, novo_dado)
print(f"Previsão para o novo dado: {'Benigno' if prediction[0] == 1 else 'Maligno'}")
"""