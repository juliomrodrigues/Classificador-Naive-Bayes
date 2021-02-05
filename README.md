# Classificador Naive Bayes
#### Utilizando aprendizagem bayesiana em uma base de dados real para classificar registros(Censo de 1994 - EUA).
#### Objetivo: Prever se um americano possui renda anual <= ou > 50 mil dólares por ano.

### Resultados - Validação Cruzada - StratifiedKFold
**Precisão** | **Pré-Processamentos** | **Desvio Padrão**
| :------: | :------: | :------: |
0.7950 | LabelEncoder | 0.0083
0.7952 | OneHotEncoder | 0.0083
**0.8039** | **LabelEncoder + StandardScaler** | **0.0083**
0.4778 | OneHotEncoder + StandardScaler | 0.0179
0.4778 | LabelEnconder + OneHotEncoder + StandardScaler | 0.0179

### Bibliotecas usadas:
- Pandas
- Sklearn
- Numpy

### Técnicas de Pré-Processamento e Tratamento dos dados usada:
- LabelEnconder(transformando atributos categóricos em números)
- StandardScaler(colocando os valores em escala)

### Ferramentas Usadas:
- Anaconda
- Spyder

### Linguagem:
- Python

#### Fonte da Base de Dados: 
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

#### Como usar:
Basta fazer o download do código fonte e da base de dados. Para executar o código por partes(células) e testar diferentes possibilidades de pré-processamento, recomendo uma IDE como Spyder ou o Jupyter. (Támbem é necessário ter o Python instalado no seu computador)
