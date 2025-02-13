import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
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
X = df[numeric_columns + categorical_columns]
y = df['income']

# Verificando e removendo quaisquer linhas com valores nulos após as conversões
X = X.dropna()
y = y[X.index]

# Normalizando as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usando validação cruzada com 10 folds
knn = GaussianNB()  # Usando Naive Bayes como modelo

# Realizando a validação cruzada com 10 folds
cv_scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')

# Exibindo os resultados da validação cruzada
print(f'Acurácia média da validação cruzada com 10 folds: {cv_scores.mean():.4f}')
print(f'Desvio padrão da acurácia: {cv_scores.std():.4f}')

# Treinando o modelo final com os dados completos
knn.fit(X_scaled, y)

# Fazendo previsões
y_pred = knn.predict(X_scaled)
y_pred_proba = knn.predict_proba(X_scaled)[:, 1]

# Visualização 1: Matriz de Confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Visualização 2: Curva ROC
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))

# Visualização 3: Distribuição das probabilidades previstas usando histograma
plt.figure(figsize=(10, 6))
for income in [0, 1]:
    mask = y == income
    plt.hist(y_pred_proba[mask], bins=30, alpha=0.5, 
             label=f'{"<=50K" if income == 0 else ">50K"}')
plt.title('Distribuição das Probabilidades Previstas por Classe Real')
plt.xlabel('Probabilidade de Renda >50K')
plt.ylabel('Contagem')
plt.legend()
plt.show()

# Análise das features mais importantes usando a variância das classes
feature_names = numeric_columns + categorical_columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Variance': np.mean(knn.var_, axis=0)
})
feature_importance = feature_importance.sort_values('Variance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='Variance', y='Feature')
plt.title('Top 10 Features Mais Importantes (Baseado na Variância)')
plt.xlabel('Variância Média')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Função para fazer previsões com novos dados
def predict_income(model, scaler, new_data):
    """
    Faz previsão de renda para novos dados
    
    Args:
        model: Modelo Naive Bayes treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0 para <=50K, 1 para >50K) e probabilidade
    """
    # Garantindo que as colunas estejam na mesma ordem do treinamento
    new_data = new_data[numeric_columns + categorical_columns]
    
    # Normalizando os novos dados
    scaled_data = scaler.transform(new_data)
    
    # Fazendo a previsão e obtendo probabilidades
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    
    return prediction, probability
