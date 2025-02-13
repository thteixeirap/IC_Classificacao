import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Carregando o dataset Pima Indians Diabetes
# O dataset pode ser baixado de: https://www.kaggle.com/uciml/pima-indians-diabetes-database
df = pd.read_csv('diabetes.csv')

# Verificando as primeiras linhas do dataset
print(df.head())

# Verificando informações básicas do dataset
print(df.info())

# Verificando se há valores nulos
print(df.isnull().sum())

# Identificando colunas numéricas e categóricas
numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_columns = []  # Não há colunas categóricas neste dataset

# Convertendo colunas numéricas para float (caso necessário)
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Codificando a variável alvo (Outcome: 0 = Não diabético, 1 = Diabético)
df['Outcome'] = df['Outcome'].astype(int)

# Visualização 1: Distribuição de diabetes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Outcome')
plt.title('Distribuição das Classes de Diabetes')
plt.xlabel('Outcome (0: Não diabético, 1: Diabético)')
plt.ylabel('Quantidade')
plt.show()

# Visualização 2: Glucose vs Idade com cor por Outcome
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Glucose', y='Age', hue='Outcome', alpha=0.6)
plt.title('Relação entre Glucose e Idade')
plt.xlabel('Glucose')
plt.ylabel('Idade')
plt.show()

# Separando features e target
X = df[numeric_columns]  # Usando apenas as colunas numéricas
y = df['Outcome']

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

# Curva ROC
# Calculando as probabilidades de previsão
y_prob = knn.predict_proba(X_test)[:, 1]  # Probabilidades para a classe positiva (Diabético)

# Calculando a curva ROC e a AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Visualizando a Curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Função para fazer previsões com novos dados
def predict_diabetes(model, scaler, new_data):
    """
    Faz previsão de diabetes para novos dados
    
    Args:
        model: Modelo KNN treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0: Não diabético, 1: Diabético)
    """
    # Garantindo que as colunas estejam na mesma ordem do treinamento
    new_data = new_data[numeric_columns]
    
    # Normalizando os novos dados
    scaled_data = scaler.transform(new_data)
    
    # Fazendo a previsão
    prediction = model.predict(scaled_data)
    
    return prediction