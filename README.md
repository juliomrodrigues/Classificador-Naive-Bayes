# Classificador Naive Bayes
Treinando um modelo por aprendizagem bayesiana e aplicando em uma base de dados para classificar registros(Censo de 1994 - EUA).
Objetivo: Prever se um americano possui renda anual <= ou > 50 mil dólares por ano.

Base Line Classifier = 0.7559 (ZeroR)

### Resultados - Validação Cruzada - StratifiedKFold
**Precisão** | **Pré-Processamentos** | **Desvio Padrão**
| :------: | :------: | :------: |
0.7950 | LabelEncoder | 0.0083
0.7952 | OneHotEncoder | 0.0083
**0.8039** | **LabelEncoder + StandardScaler** | **0.0083**
0.4778 | OneHotEncoder + StandardScaler | 0.0179
0.4778 | LabelEnconder + OneHotEncoder + StandardScaler | 0.0179

### Matriz de Confusão (Média):
x | **0** | **1**
| :------: | :------: | :------: |
0 | **2352.7** | 119.3
1 | 519.2 | **264.9**

A Matriz na tabela acima é formada pela média de todas as matrizes geradas ao longo de 10 execuções usando pré-processamentos LabelEncoder + StandardScaler.

A diagonal principal (em negrito) destaca os registros classificados corretamente.

### Bibliotecas usadas:
- Pandas
- Sklearn
- Numpy

### Técnicas de Pré-Processamento e Tratamento dos dados usada:
- LabelEnconder;
- OneHotEncoder;
- StandardScaler;

### Ferramentas Usadas:
- Anaconda
- Spyder

### Linguagem:
- Python

### Fonte da Base de Dados: 
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

### Como usar:
1. Faça o download do classificador ja treinado dispoível neste mesmo repositório [aqui](https://github.com/juliomrodrigues/Classificador-Naive-Bayes/blob/main/classificador_naive_bayes.sav).
2. Abra o arquivo.py que deseja usar o classificador ou então criar um novo.
3. Execute o código abaixo para fazer a importação:
~~~~python
import pickle
naive_bayes = pickle.load(open('classificador_naive_bayes.sav', 'rb'))
~~~~~

#### Outros Classificadores:
- [Árvore de Decisão](https://github.com/juliomrodrigues/Arvore-de-Decisao)
- [Random Forest](https://github.com/juliomrodrigues/Random-Forest-Classificador)
- [Aprendizagem por Regras](https://github.com/juliomrodrigues/Classificador-Regras)
- [Aprendizagem por Instâncias(KNN)](https://github.com/juliomrodrigues/Classificador-KNN)
- [Regressão Logística](https://github.com/juliomrodrigues/Regressao-Logistica-Classificador)
- [SVM](https://github.com/juliomrodrigues/Classificador-SVM)
- [Rede Neural](https://github.com/juliomrodrigues/Classificador-Rede-Neural)
