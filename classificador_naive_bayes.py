"""
Created on Fri Jan 15 22:28:11 2021
@author: julio

Base de Dados: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.
"""

import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
 
base = pandas.read_csv('census.csv')
 
atributos = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Transformando variaveis categóricas(Strings) em números
label_encoder= LabelEncoder()
atributos[:, 1] = label_encoder.fit_transform(atributos[:, 1])
atributos[:, 3] = label_encoder.fit_transform(atributos[:, 3])
atributos[:, 5] = label_encoder.fit_transform(atributos[:, 5])
atributos[:, 6] = label_encoder.fit_transform(atributos[:, 6])
atributos[:, 7] = label_encoder.fit_transform(atributos[:, 7])
atributos[:, 8] = label_encoder.fit_transform(atributos[:, 8])
atributos[:, 9] = label_encoder.fit_transform(atributos[:, 9])
atributos[:, 13] = label_encoder.fit_transform(atributos[:, 13])

classe = label_encoder.fit_transform(classe)

#Aplicando escalonamento nos valores
scaler = StandardScaler()
atributos = scaler.fit_transform(atributos)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.15, random_state=0)

# Gerando Tabela de Classificação
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Calculando a precisão e gerando matriz de confusão
taxa_precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print(f'A taxa de precisão alcançada foi de {round(taxa_precisao*100)}%')