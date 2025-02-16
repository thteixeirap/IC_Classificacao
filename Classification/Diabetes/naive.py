import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
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

# Identificando colunas numéricas
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

# Visualização 2: Boxplot de Glucose por classe de diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Outcome', y='Glucose')
plt.title('Distribuição de Glucose por Classe de Diabetes')
plt.xlabel('Classe de Diabetes (0: Não diabético, 1: Diabético)')
plt.ylabel('Glucose')
plt.show()

# Separando features e target
X = df[numeric_columns]
y = df['Outcome']

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
y_pred_proba = nb.predict_proba(X_test)[:, 1]

# Visualização 3: Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Visualização 4: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
print(classification_report(y_test, y_pred))

# Visualização 5: Distribuição das probabilidades previstas usando histograma
plt.figure(figsize=(10, 6))
for outcome in [0, 1]:
    mask = y_test == outcome
    plt.hist(y_pred_proba[mask], bins=30, alpha=0.5, 
             label=f'{"Não diabético" if outcome == 0 else "Diabético"}')
plt.title('Distribuição das Probabilidades Previstas por Classe Real')
plt.xlabel('Probabilidade de Diabetes')
plt.ylabel('Contagem')
plt.legend()
plt.show()

# Análise das features mais importantes usando a variância das classes
feature_names = numeric_columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Variance': np.mean(nb.theta_, axis=0)  # Usando a média das médias das classes
})
feature_importance = feature_importance.sort_values('Variance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='Variance', y='Feature')
plt.title('Top 10 Features Mais Importantes (Baseado na Média das Médias das Classes)')
plt.xlabel('Média das Médias')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Função para fazer previsões com novos dados
def predict_diabetes(model, scaler, new_data):
    """
    Faz previsão de diabetes para novos dados
    
    Args:
        model: Modelo Naive Bayes treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0: Não diabético, 1: Diabético) e probabilidade
    """
    # Garantindo que as colunas estejam na mesma ordem do treinamento
    new_data = new_data[numeric_columns]
    
    # Normalizando os novos dados
    scaled_data = scaler.transform(new_data)
    
    # Fazendo a previsão e obtendo probabilidades
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    
    return prediction, probability

# Exemplo de como fazer uma previsão com novos dados
"""
novo_dado = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [120],
    'BloodPressure': [70],
    'SkinThickness': [30],
    'Insulin': [80],
    'BMI': [25.5],
    'DiabetesPedigreeFunction': [0.3],
    'Age': [35]
})

prediction, prob = predict_diabetes(nb, scaler, novo_dado)
print(f"Previsão para o novo dado: {'Diabético' if prediction[0] == 1 else 'Não diabético'}")
print(f"Probabilidade de diabetes: {prob[0]:.2f}")
"""