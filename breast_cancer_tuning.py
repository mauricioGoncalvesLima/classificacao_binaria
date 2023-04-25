# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:06:17 2022

@author: Mauricio Gonçalves
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv(r'C:\Users\Mauricio Gonçalves\Documents\Cursos\Deep Learning\classificacao_binaria\entradas_breast.csv', sep=',')
classes = pd.read_csv(r'C:\Users\Mauricio Gonçalves\Documents\Cursos\Deep Learning\classificacao_binaria\saidas_breast.csv', sep=',')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loss,
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]
              }


grid_search = GridSearchCV(estimator = classificador, param_grid = parametros,
                           scoring = 'accuracy', cv = 5)

grid_search = grid_search.fit(previsores, classes)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_



