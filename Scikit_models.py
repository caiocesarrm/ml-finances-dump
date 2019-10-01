# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:59:26 2018

@author: caio/kelvin
"""

# -*- coding: utf-8 -*-
# In[1]:

import sys
import pandas
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import sklearn
import itertools
import talib as ta
import seaborn
import numpy as np
from keras import models
from keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(sys.version)
print(pandas.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

# In[Parametros]:

n_previsoes = 500


# In[2]:

def dataframe_ativos(simbolo_ativo):
    df = pandas.read_csv(simbolo_ativo + '.csv', 
                     parse_dates={'Date' : ['Data', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='Date')
    
    df = df.astype('float')
    df = df['2019-01-13' : ]
    
    df['ROC'] = ta.ROC(df.Close.values, timeperiod = 1)
    df[simbolo_ativo] = df['ROC'] 
    #df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 2)
    #df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 3)
    #df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 4)
    #df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 5)
    
    for i in range(len(df[simbolo_ativo])):
        if(df[simbolo_ativo].iloc[i] >= 0):
            df[simbolo_ativo].iloc[i] = 1
        else:
            df[simbolo_ativo].iloc[i] = 0

    '''
    for i in range(len(df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    
    for i in range(len(df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    '''
    '''
    for i in range(len(df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    
    for i in range(len(df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    '''       
    df = df.drop(['Close', 'Voltick', 'High', 'Open', 'Low', 'Volume', 'ROC'],axis=1)
    
    df.dropna(inplace = True)
    #periodo wing19
   
    #periodowinj19
    df = df['2019-02-13' : '2019-02-15']
    df.dropna(inplace = True)
    #df = df.between_time('09:15:00', '12:00:00')
    #df = df.between_time('12:00:00', '14:00:00')
    #df = df.between_time('14:00:00', '16:00:00')
    #df = df.between_time('16:15:00', '17:55:00')
    return df



#Carregar os dados
def dataframe_principal(simbolo_ativo):
    df = pandas.read_csv(simbolo_ativo + '.csv', 
                     parse_dates={'Date' : ['Data', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='Date')
    
    df = df.astype('float')
    df = df['2019-01-13' : ]

    df['Close_t1'] = df['Close'].shift(-1)
    df[simbolo_ativo] = ta.ROC(df.Close_t1.values, timeperiod = 1)
    #df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 2)
    #df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 3)
    #df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 4)
    #df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 5)
    df = df.drop(['Voltick', 'High', 'Open', 'Low', 'Volume'],axis=1)
    
    for i in range(len(df[simbolo_ativo])):
        if(df[simbolo_ativo].iloc[i] >= 0):
            df[simbolo_ativo].iloc[i] = 1
        else:
            df[simbolo_ativo].iloc[i] = 0
    '''
    for i in range(len(df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
            
    for i in range(len(df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    '''
    '''
    for i in range(len(df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    
    for i in range(len(df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)])):
        if(df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] >= 0):
            df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 1
        else:
            df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)].iloc[i] = 0
    
    '''
    
    #periodo wing19
    
    #periodowinj19
    df = df['2019-02-13' : '2019-02-15']
    df.dropna(inplace = True)
    #df = df.between_time('09:15:00', '12:00:00')
    #df = df.between_time('12:00:00', '14:00:00')
    #df = df.between_time('14:00:00', '16:00:00')
    #df = df.between_time('16:15:00', '17:55:00')
    return df



# In[8]:
def treinar(X, y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = len(X) - k, test_size = k, shuffle = False)
    
    return X_test, y_test, X_train, y_train

def teste_apenas_compra(close, closet1):
    lucro = 0
    prejuizo = 0
    falha = 0
    sucesso = 0
    
    dif = closet1 - close
    if(dif < 0):
        falha += 1
        if(closet1-close  < -50):
            prejuizo = -50
        else:
            prejuizo+= closet1-close
        #print("Prejuizo de compra foi de : ", prejuizo)
    elif(dif > 0):
        sucesso += 1
        lucro+= closet1-close
        #print("lucro de compra foi de : ", lucro)
    return lucro, falha, sucesso, prejuizo

def prever_resultado(close_real, fechamento_anterior, previsao, margem, close, closet1):
    lucro = 0
    prejuizo = 0
    falha = 0
    sucesso = 0
    sucesso_compra = 0
    falha_compra = 0
    sucesso_venda = 0
    falha_venda = 0
    '''
    if(close_real < 0 and previsao < 0):
        sucesso += 1
        sucesso_venda += 1
    elif(close_real < 0 and previsao > 0):
        falha += 1
        falha_compra+=1
    elif(close_real > 0 and previsao > 0):
        sucesso += 1
        sucesso_compra += 1
    elif(close_real > 0 and previsao < 0):
        falha += 1
        falha_venda+=1
    '''
    
    if(close_real == 0 and previsao == 0):
        sucesso += 1
        #lucro+= close-closet1
        #print("Lucro de venda foi de : ", lucro)
        sucesso_venda += 1
    elif(close_real == 0 and previsao == 1):
        falha += 1
        if(closet1-close  < -50):
            prejuizo = -50
        else:
            prejuizo+= closet1-close
        print("Prejuizo de compra foi de : ", prejuizo)
        falha_compra+=1
    elif(close_real == 1 and previsao == 1):
        sucesso += 1
        lucro+= closet1-close
        print("lucro de compra foi de : ", lucro)
        sucesso_compra += 1
    elif(close_real == 1 and previsao == 0):
        falha += 1
        #prejuizo += close-closet1
        #print("Prejuizo de venda foi de : ", prejuizo)
        falha_venda+=1
        
    '''
    #dif_lr = previsao - fechamento_anterior   
    if dif_lr > margem:
        lucro_op = close_real - fechamento_anterior
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
            sucesso_compra += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
            falha_compra += 1
            
    elif dif_lr < -margem: 
        lucro_op =  fechamento_anterior - close_real
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
            sucesso_venda += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
            falha_venda += 1
    '''
    return lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda


# In[Matriz de correlações entre ativos]:

#lista_simbolos = ['ABEV3M1', 'B3SA3M1', 'BBAS3M1', 'BBDC4M1', 'BRKM5M1', 'ELET3M1', 'GGBR4M1', 'ITUB4M1', 'LAME4M1', 'LREN3M1', 'MGLU3M1', 'NATU3M1', 'PCAR4M1', 'PETR4M1', 'SANB11M1', 'SUZB3M1', 'TIMP3M1', 'USIM5M1', 'VALE3M1']
lista_simbolos = ['ABEV3M1', 'B3SA3M1', 'BBAS3M1', 'BBDC4M1', 'ITUB4M1', 'MGLU3M1', 'PETR4M1', 'VALE3M1']
lista_dataframes = []

for simbolo in lista_simbolos:
    df_aux = dataframe_ativos(simbolo)
    lista_dataframes.append(df_aux)

df_ativos = pandas.concat(lista_dataframes, axis=1, sort=False)
df_ativos.dropna(inplace = True)

corr_df = df_ativos.corr(method='pearson')

corr_df.head().reset_index()
del corr_df.index.name
corr_df.head(10)

mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True

seaborn.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.show()

# In[Criação dataframe final]:
print("teste")

df_target = dataframe_principal('WINJ19M1')

df_final = pandas.concat([df_ativos, df_target], axis=1, sort=False)
df_final.dropna(inplace = True)

media = df_final['WINJ19M1'].mean()

closet1 = df_final['Close_t1']
close = df_final['Close']

X = df_final.drop(['WINJ19M1', 'Close', 'Close_t1'],axis=1)
y = df_final['WINJ19M1']

variacao = 0

# In[Regressão Linear]:
from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(X)
a = pca.singular_values_
b = pca.explained_variance_ratio_




falha_total = 0
sucesso_total = 0
lucro_total = 0
prejuizo_total = 0
falha_total_compra = 0
sucesso_total_compra = 0
falha_total_venda = 0
sucesso_total_venda = 0

lucro_teste_compra_total = 0
falha_teste_compra_total = 0
sucesso_teste_compra_total = 0
prejuizo_teste_compra_total = 0

X_test, y_test, X_train, y_train = treinar(X, y, n_previsoes)

#scaler = MinMaxScaler(feature_range = (0,1))

#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)



clfRF = RandomForestClassifier(n_estimators=600, max_depth=20, random_state=0)
clfRF.fit(X_train, y_train)
previsao = clfRF.predict(X_test)

'''
clfRF = LinearRegression()
clfRF.fit(X_train, y_train)
previsao = clfRF.predict(X_test)
'''

print("O MSE foi: ", mean_squared_error(previsao, y_test))



for i in range(1,n_previsoes):
    
    lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda = prever_resultado(y_test[i], y_test[i-1], previsao[i], variacao, close[len(close) - n_previsoes + i], closet1[len(close) - n_previsoes + i])
    lucro_teste_compra, falha_teste_compra, sucesso_teste_compra, prejuizo_teste_compra = teste_apenas_compra(close[len(close) - n_previsoes + i], closet1[len(close) - n_previsoes + i])
    
    prejuizo_total += prejuizo
    lucro_total += lucro
    falha_total += falha
    sucesso_total += sucesso
    falha_total_compra += falha_compra
    sucesso_total_compra +=sucesso_compra
    falha_total_venda += falha_venda
    sucesso_total_venda += sucesso_venda
    
    lucro_teste_compra_total += lucro_teste_compra
    falha_teste_compra_total += falha_teste_compra
    sucesso_teste_compra_total += sucesso_teste_compra
    prejuizo_teste_compra_total += prejuizo_teste_compra
 
        
  
print("\n\nSucesso Total é: ", sucesso_total)
print("Falha total é: ", falha_total)      
print("lucro Total é: ", lucro_total)
print("Prejuizo total é: ", prejuizo_total)   
print("Sucesso Total compra é: ", sucesso_total_compra)
print("Falha total compra é: ", falha_total_compra)      
print("Sucesso Total venda é: ", sucesso_total_venda)
print("falha total venda é: ", falha_total_venda)   

print("\nSucesso Total é: ", sucesso_teste_compra_total)
print("Falha total é: ", falha_teste_compra_total)      
print("lucro Total é: ", lucro_teste_compra_total)
print("Prejuizo total é: ", prejuizo_teste_compra_total)   


# In[Random Forest Regressor]:
'''
falha_total = 0
sucesso_total = 0
lucro_total = 0
prejuizo_total = 0
falha_total_compra = 0
sucesso_total_compra = 0
falha_total_venda = 0
sucesso_total_venda = 0

X_test, y_test, X_train, y_train = treinar(X, y, n_previsoes)

scalerRF = MinMaxScaler(feature_range = (0,1))

#X_train = scalerRF.fit_transform(X_train)
#X_test = scalerRF.transform(X_test)

RF = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=200)
RF.fit(X_train, y_train)
previsao = RF.predict(X_test)

print("O MSE foi: ", mean_squared_error(previsao, y_test))
print("\nTamanho de teste é: ", len(X_test))

for i in range(1,n_previsoes):
    
    
    lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda = prever_resultado(y_test[i], y_test[i-1], previsao[i], variacao)
    
    prejuizo_total += prejuizo
    lucro_total += lucro
    falha_total += falha
    sucesso_total += sucesso    
    falha_total_compra += falha_compra
    sucesso_total_compra +=sucesso_compra
    falha_total_venda += falha_venda
    sucesso_total_venda += sucesso_venda
  
print("Sucesso Total RF é: ", sucesso_total)
print("Falha total RF é: ", falha_total)      
print("lucro Total RF é: ", lucro_total)
print("Prejuizo total RF é: ", prejuizo_total)   
print("Sucesso Total compra RF é: ", sucesso_total_compra)
print("Falha total compra Rf é: ", falha_total_compra)      
print("Sucesso Total venda  RFé: ", sucesso_total_venda)
print("falha total venda RF é: ", falha_total_venda) 
'''

# In[Rede Neural]:
'''
falha_total = 0
sucesso_total = 0
lucro_total = 0
prejuizo_total = 0
falha_total_compra = 0
sucesso_total_compra = 0
falha_total_venda = 0
sucesso_total_venda = 0

X_test, y_test, X_train, y_train = treinar(X, y, n_previsoes)
'''
'''
scalerNN = MinMaxScaler(feature_range = (0,1))
scalerNNy = MinMaxScaler(feature_range = (0,1)) 

X_train = scalerNN.fit_transform(X_train)
X_test = scalerNN.transform(X_test)
#y_train = scalerNNy.fit_transform(y_train.values.reshape(-1, 1))
#y_test = scalerNNy.transform(y_test.values.reshape(-1,1))

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
'''
'''
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
'''
'''
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(X_train, y_train, epochs = 12, batch_size=32, verbose=0, validation_split = 0.1)

val_mse, val_mae = model.evaluate(X_test, y_test, verbose=1)
all_scores = []
all_scores.append(val_mae)
previsao = model.predict(X_test)

for i in range(0, len(previsao)):
    if previsao[i] > 0.5:
        previsao[i] = 1
    else:
        previsao[i] = 0

for i in range(1, n_previsoes):
    
    
    lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda = prever_resultado(y_test[i], y_test[i-1], previsao[i], variacao)
    
    prejuizo_total += prejuizo
    lucro_total += lucro
    falha_total += falha
    sucesso_total += sucesso
    falha_total_compra += falha_compra
    sucesso_total_compra +=sucesso_compra
    falha_total_venda += falha_venda
    sucesso_total_venda += sucesso_venda
        
  
print("Sucesso Total NN é: ", sucesso_total)
print("Falha total NN é: ", falha_total)      
print("lucro Total NN é: ", lucro_total)
print("Prejuizo total NN é: ", prejuizo_total)   
print("Sucesso Total compra NN é: ", sucesso_total_compra)
print("Falha total compra NN  é: ", falha_total_compra)      
print("Sucesso Total venda NN é: ", sucesso_total_venda)
print("falha total venda NN é: ", falha_total_venda)   

plt.plot(y_test, color = 'red', label = 'Fechamento Real')
plt.plot(previsao, color = 'blue', label = 'Fechamento Previsto')
plt.title('Predição')
plt.xlabel('Time')
plt.ylabel('Preços')
plt.legend()
plt.show()

plt.figure() 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='best') 

'''
