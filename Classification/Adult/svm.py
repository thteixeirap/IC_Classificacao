import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Carregando o dataset Adult
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

# Carregando os dados do arquivo CSV
df = pd.read_csv('adult.csv', names=columns)

# Pré-processamento dos dados
# Removendo espaços em branco
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Substituindo valores ausentes
df = df.replace('?', np.nan)
df = df.dropna()

# Identificando colunas numéricas e categóricas
numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'native-country']

# Convertendo colunas numéricas para float
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Codificando variáveis categóricas
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column].astype(str))

# Codificando a variável alvo
df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

# Separando features e target
X = df[numeric_columns + categorical_columns]  # Usando apenas as colunas processadas
y = df['income']

# Verificando e removendo quaisquer linhas com valores nulos após as conversões
X = X.dropna()
y = y[X.index]

# Normalizando as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usando validação cruzada com 10 folds
knn = KNeighborsClassifier(n_neighbors=5)  # Usando 5 como exemplo de valor de k

# Realizando a validação cruzada com 10 folds
cv_scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')

# Exibindo os resultados da validação cruzada
print(f'Acurácia média da validação cruzada com 10 folds: {cv_scores.mean():.4f}')
print(f'Desvio padrão da acurácia: {cv_scores.std():.4f}')

# Treinando o modelo final com os dados completos
knn.fit(X_scaled, y)

# Fazendo previsões
y_pred = knn.predict(X_scaled)

# Visualização 1: Matriz de Confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))

# Curva ROC
# Calculando as probabilidades de previsão
y_prob = knn.predict_proba(X_scaled)[:, 1]  # Probabilidades para a classe positiva (>50K)

# Calculando a curva ROC e a AUC
fpr, tpr, thresholds = roc_curve(y, y_prob)
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