import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data/wine_dataset.csv')

# Exibe os 5 primeros dados do dataset
dataset.head()

# Substitui os valores para facilitar a classificação
dataset['style'] = dataset['style'].replace('red', 0)
dataset['style'] = dataset['style'].replace('white', 1)

y = dataset['style']
x = dataset.drop('style', axis = 1)

# Prepara os conjuntos de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
print(dataset.shape, x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape)

# Modelo de classificação
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)

print(f'Amostra de dados real: {y_teste[400:405]}')

previsoes = modelo.predict(x_teste[400:405])
print(f'Previsões: {previsoes}')
