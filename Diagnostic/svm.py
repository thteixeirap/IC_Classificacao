import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# Carregando o dataset Breast Cancer Wisconsin
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Verificando as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Verificando informações básicas do dataset
print("\nInformações do dataset:")
print(df.info())

# Verificando se há valores nulos
print("\nVerificando valores nulos:")
print(df.isnull().sum())

# Todas as colunas são numéricas neste dataset
numeric_columns = df.columns[:-1]  # Todas exceto 'diagnosis'

# Visualização 1: Distribuição de diagnósticos
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='diagnosis')
plt.title('Distribuição das Classes de Diagnóstico')
plt.xlabel('Diagnóstico (0: Benigno, 1: Maligno)')
plt.ylabel('Quantidade')
plt.show()

# Visualização 2: Relação entre duas características importantes com cor por diagnóstico
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, 
                x='mean radius', 
                y='mean texture', 
                hue='diagnosis', 
                alpha=0.6)
plt.title('Relação entre Raio Médio e Textura Média')
plt.xlabel('Raio Médio')
plt.ylabel('Textura Média')
plt.show()

# Separando features e target
X = df[numeric_columns]
y = df['diagnosis']

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

print("\nResultados da Validação Cruzada (10 folds):")
print(f"Acurácia média: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")
print(f"Scores individuais: {cv_scores}")

# Treinando o modelo final
svm.fit(X_train, y_train)

# Fazendo previsões
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)[:, 1]

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
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Linha de referência')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - SVM')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Visualização 5: Distribuição das probabilidades previstas
plt.figure(figsize=(10, 6))
for outcome in [0, 1]:
    mask = y_test == outcome
    plt.hist(y_pred_proba[mask], bins=30, alpha=0.5, 
             label=f'{"Benigno" if outcome == 0 else "Maligno"}')
plt.title('Distribuição das Probabilidades Previstas por Classe Real')
plt.xlabel('Probabilidade de Malignidade')
plt.ylabel('Contagem')
plt.legend()
plt.show()

# Função para fazer previsões com novos dados
def predict_breast_cancer(model, scaler, new_data):
    """
    Faz previsão de câncer de mama para novos dados
    
    Args:
        model: Modelo SVM treinado
        scaler: Scaler ajustado aos dados de treino
        new_data: DataFrame com os mesmos campos do dataset original
    
    Returns:
        Previsão (0: Benigno, 1: Maligno) e probabilidade
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
    'mean radius': [15.0],
    'mean texture': [20.0],
    # ... adicionar todas as outras features ...
})

prediction, prob = predict_breast_cancer(svm, scaler, novo_dado)
print(f"Previsão: {'Maligno' if prediction[0] == 1 else 'Benigno'}")
print(f"Probabilidade de malignidade: {prob[0]:.2f}")
"""