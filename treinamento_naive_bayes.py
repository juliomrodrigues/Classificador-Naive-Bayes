"""
Base de Dados: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.
"""
import pandas
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
 
base = pandas.read_csv('census.csv')
 
atributos = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Transformando variaveis categóricas(Strings) em números
label_encoder= LabelEncoder()
classe = label_encoder.fit_transform(classe)
atributos[:, 1] = label_encoder.fit_transform(atributos[:, 1])
atributos[:, 3] = label_encoder.fit_transform(atributos[:, 3])
atributos[:, 5] = label_encoder.fit_transform(atributos[:, 5])
atributos[:, 6] = label_encoder.fit_transform(atributos[:, 6])
atributos[:, 7] = label_encoder.fit_transform(atributos[:, 7])
atributos[:, 8] = label_encoder.fit_transform(atributos[:, 8])
atributos[:, 9] = label_encoder.fit_transform(atributos[:, 9])
atributos[:, 13] = label_encoder.fit_transform(atributos[:, 13])

# ONE_HOT_ENCODER
one_hot_encode = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
atributos = one_hot_encode.fit_transform(atributos).toarray()

# STANDARD_SCALER
scaler = StandardScaler()
atributos = scaler.fit_transform(atributos)

skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in skfold.split(atributos,
                                                    np.zeros(shape=(atributos.shape[0], 1))):
   classificador = GaussianNB()
   classificador.fit(atributos[indice_treinamento], classe[indice_treinamento])  
   previsoes = classificador.predict(atributos[indice_teste])
   precisao = accuracy_score(classe[indice_teste], previsoes)
   matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
   resultados.append(precisao)

matriz_final = np.mean(matrizes, axis = 0)
resultados = np.asarray(resultados)
   
print(f'Média = {resultados.mean()}')
print(f'Desvio Padrão = {resultados.std()}')
