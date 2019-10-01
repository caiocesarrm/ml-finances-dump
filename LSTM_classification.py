# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:13:17 2018

@author: caioc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:15:25 2018

@author: caioc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import talib as ta
import sys
import matplotlib
import seaborn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


timesteps = 5
nb_epochs = 30
batch_size = 32
 

# In[1]:

def dataframe_ativos(simbolo_ativo):
    df = pandas.read_csv(simbolo_ativo + '.csv', 
                     parse_dates={'Date' : ['Data', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='Date')
    
    df = df.astype('float')
    df['ROC'] = ta.ROC(df.Close.values, timeperiod = 1)
    df[simbolo_ativo] = df['ROC'] 
    #df[simbolo_ativo] = df['Close'] 
    #df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 2)
    #df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 3)
    #df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 4)
    #df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 5)
    df = df.drop(['Close', 'Voltick', 'High', 'Open', 'Low', 'Volume', 'ROC'],axis=1)
    
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
   
    
    df.dropna(inplace = True)
    #periodo wing19
    df = df['2019-02-13' : '2019-02-15']
    #periodowinj19
    #df = df['2019-02-13' : '2019-02-15']
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
    

    df['Close_t1'] = df['Close'].shift(-1)
    #df[simbolo_ativo] = ta.ROC(df.Close_t1.values, timeperiod = 1)
    #df[simbolo_ativo] = df['Close'] 
    df[simbolo_ativo] = ta.ROC(df.Close.values, timeperiod = 1)
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
    
   
    df.dropna(inplace = True)
    #periodo wing19
    df = df['2019-02-13' : '2019-02-15']
    #periodowinj19
    #df = df['2019-02-13' : '2019-02-15']
    df.dropna(inplace = True)
    #df = df.between_time('09:15:00', '12:00:00')
    #df = df.between_time('12:00:00', '14:00:00')
    #df = df.between_time('14:00:00', '16:00:00')
    #df = df.between_time('16:15:00', '17:55:00')
    return df

# In[1]:

def treinar_lstm(X,y):
    #TimeSeries to Supervised
    X_train = []
    y_train = []
    
    for i in range(timesteps, len(X)):
            X_train.append(X[i-timesteps : i, :])
            y_train.append(y[i])
          
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #Dimensão, número de linhas, número de colunas(timesteps) e número de features
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
    regressor = Sequential()
    #Criar 3 layers LSTM
    regressor.add(LSTM(units = 50, input_shape = (X_train.shape[1], X_train.shape[2])))
    #Criar Output Layer
    regressor.add(Dense(units = 1))
    
    #compilar
    regressor.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics=['mae'])

    #treinar
    history = regressor.fit(X_train, y_train, epochs = nb_epochs, batch_size = batch_size,validation_split = 0.1)
    
    return regressor, history
  
    
lista_simbolos = ['ABEV3M1', 'B3SA3M1', 'BBAS3M1', 'BBDC4M1', 'ITUB4M1', 'MGLU3M1', 'PETR4M1', 'VALE3M1']
lista_dataframes = []

for simbolo in lista_simbolos:
    df_aux = dataframe_ativos(simbolo)
    lista_dataframes.append(df_aux)

df_ativos = pandas.concat(lista_dataframes, axis=1, sort=False)
df_ativos.dropna(inplace = True)

df_target = dataframe_principal('WINJ19M1')

df_final = pandas.concat([df_ativos, df_target], axis=1, sort=False)
df_final.dropna(inplace = True)

media = df_final['WINJ19M1'].mean()

close = df_final['Close']
closet1 = df_final['Close_t1']

df_final = df_final.drop(['Close', 'Close_t1'], axis = 1)

X = df_final.drop(['WINJ19M1'],axis=1)
y = df_final['WINJ19M1']

n_previsoes = 500



testsize = round(len(y) - n_previsoes)

close = close[testsize:]
closet1 = closet1[testsize:]
teste = X[testsize: ]
y_test = y[testsize:]

X = X[0:testsize]
y = y[0:testsize]

X = X.values
y = y.values

#scaler_x1 = StandardScaler()
#scaler_y1 = StandardScaler()

scaler_x1 = MinMaxScaler(feature_range=(0,1))
scaler_y1 = MinMaxScaler(feature_range=(0,1))

X = scaler_x1.fit_transform(X)
#y = scaler_y1.fit_transform(y.reshape(-1,1))

regressor, history = treinar_lstm(X,y)


X = teste
X_train = []
y_test_real = []
X = scaler_x1.transform(X)
for i in range(timesteps, len(X)):
    X_train.append(X[i-timesteps : i, :])
    y_test_real.append(y_test[i])
      

X_train = np.array(X_train)
#Dimensão, número de linhas, número de colunas(timesteps) e número de features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
previsao = regressor.predict(X_train)
#predictions = scaler_y1.inverse_transform(predictions)

falha_total = 0
sucesso_total = 0
lucro_total = 0
prejuizo_total = 0
falha_total_compra = 0
sucesso_total_compra = 0
falha_total_venda = 0
sucesso_total_venda = 0
variacao =0

def prever_resultado(close_real, fechamento_anterior, previsao, margem, close, closet1):
    lucro = 0
    prejuizo = 0
    falha = 0
    sucesso = 0
    sucesso_compra = 0
    falha_compra = 0
    sucesso_venda = 0
    falha_venda = 0
    print(close_real)
    print(previsao)
    print(close)
    print(closet1)
    if previsao > 0.5:
        previsao = 1
    else:
        previsao = 0
    
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
        
    return lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda

for i in range(len(y_test_real)):
    
    lucro, falha, sucesso, prejuizo, sucesso_compra, falha_compra, sucesso_venda, falha_venda = prever_resultado(y_test[i+timesteps], y_test[i-1], previsao[i], variacao, close[i+timesteps], closet1[i+timesteps])
    
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
# In[1]:
#Prever
'''
df_test = ler_dataframe('WINZ18M1-prever2.csv')

df_total = pandas.concat((df, df_test), axis = 0)

X2 = df_total.drop(['Close', 'Voltick', 'Close_t_1', 'diff'],axis=1)
y2 = df_test['Close']

inputs = X2[len(df_total) - len(df_test) - timesteps : ].values
inputs = scaler_x1.transform(inputs)

X_test = []

for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps : i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler_y1.inverse_transform(predicted_stock_price)

real_price = y2.values
fechamento_anterior = df_test['Close'].values

def prever_resultado_lstm(real_price, previsao_preco, fechamento_anterior):
    lucro = 0
    prejuizo = 0
    falha = 0
    sucesso = 0
    compra = 0
    venda = 0
    sucesso_compra = 0
    falha_compra = 0
    sucesso_venda = 0
    falha_venda = 0
    
    #dif_lr = previsao_preco
    dif_lr = previsao_preco - fechamento_anterior
    print("\nvalor de i é : ", i)
    print("Predição LSTM = ",previsao_preco, "   Close real = " ,real_price, "Fechamento anterior: ", fechamento_anterior)
    print("A diferença da previsão foi de ", dif_lr, " pontos")
    
    if dif_lr > 0:
        compra = 1
        lucro_op = real_price - fechamento_anterior
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
            sucesso_compra += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
            falha_compra += 1
        print("Resultado da operacao foi de compra: ", lucro_op)
    elif dif_lr < 0:
        venda = 1
        lucro_op =  fechamento_anterior - real_price
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
            sucesso_venda += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
            falha_venda += 1
        print("Resultado da operacao foi de venda: ", lucro_op)
        
    print("lucro atual: ", lucro)
    print("Prejuizo atual: ", prejuizo)
    return lucro, falha, sucesso, prejuizo, compra, venda, sucesso_compra, sucesso_venda, falha_compra, falha_venda
    
        
combinado_compra1 = []
combinado_venda1 = []

falha_compra_lstm = 0
falha_venda_lstm = 0
sucesso_compra_lstm = 0
sucesso_venda_lstm =0
falha_total_lstm = 0
sucesso_total_lstm = 0
lucro_total_lstm = 0
prejuizo_total_lstm = 0

for i in range(1, len(predicted_stock_price)):
    lucro, falha, sucesso, prejuizo, compra, venda, sucesso_compra, sucesso_venda, \
        falha_compra, falha_venda = prever_resultado_lstm(real_price[i], predicted_stock_price[i],fechamento_anterior[i-1])
    
    prejuizo_total_lstm += prejuizo
    lucro_total_lstm += lucro
    falha_total_lstm += falha
    sucesso_total_lstm += sucesso
    falha_compra_lstm += falha_compra
    falha_venda_lstm += falha_venda
    sucesso_compra_lstm += sucesso_compra
    sucesso_venda_lstm += sucesso_venda
    
    if compra == 1:
        combinado_compra1.append(i)
    elif venda == 1:
        combinado_venda1.append(i)


# In[lr]:

#Carregar os dados
df_lr = ler_dataframe('WINZ18M1LR.csv')

X_lr = df_lr.drop(['Close', 'Voltick', 'Close_t_1', 'diff'],axis=1)
y_lr = df_lr['Close_t_1']


def treinar(X_lr, y_lr, k):
    X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, train_size = len(X_lr) - k, test_size = k, shuffle = False)
    
    return X_test, y_test, X_train, y_train



def prever_resultado(X_prever, y_prever, y_train):
    lucro = 0
    prejuizo = 0
    falha = 0
    sucesso = 0
    compra = 0
    venda = 0
    i = 0
    
    previsao_preco = LR.predict(X_prever[i].reshape(1,-1))

    dif_lr = previsao_preco - y_train[len(y_train) - 1]
    print("\nvalor de i é : ", i)
    print("Predição REAL LR = ",previsao_preco, "   Close real = " ,y_prever[i], "Fechamento anterior: ", y_train[len(y_train) - 1])
    print("A diferença da previsão foi de ", dif_lr, " pontos")
    
    
    if dif_lr > 0:
        compra = 1
        lucro_op = y_prever[i] - y_train[len(y_train)- 1]
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
        print("Resultado da operacao foi de compra: ", lucro_op)
    elif dif_lr < 0: 
        venda = 1
        lucro_op =  y_train[len(y_train) - 1] - y_prever[i]
        if lucro_op > 0:
            lucro += lucro_op
            sucesso += 1
        elif lucro_op < 0:
            prejuizo += lucro_op
            falha += 1
        print("Resultado da operacao foi de venda: ", lucro_op)
        
    print("lucro atual: ", lucro)
    print("Prejuizo atual: ", prejuizo)
    return lucro, falha, sucesso, prejuizo, compra, venda

falha_total_lr = 0
sucesso_total_lr = 0
lucro_total_lr = 0
prejuizo_total_lr = 0

combinado_compra2 = []
combinado_venda2 = []

for i in range(len(real_price), 0, -1):
    X_test_lr, y_test_lr, X_train_lr, y_train_lr = treinar(X_lr, y_lr, i)

    fechamento_atual = y_train_lr

    scaler = StandardScaler()

    X_train_lr = scaler.fit_transform(X_train_lr)
    X_test_lr = scaler.transform(X_test_lr)

    LR = LinearRegression()
    LR.fit(X_train_lr, y_train_lr)
    predictionsLR = LR.predict(X_test_lr)
    print("O MSE foi: ", mean_squared_error(predictionsLR, y_test_lr))
    print("\nTamanho de teste é: ", len(X_test_lr))

    lucro, falha, sucesso, prejuizo, compra, venda = prever_resultado(X_test_lr, y_test_lr,fechamento_atual)
    
    prejuizo_total_lr += prejuizo
    lucro_total_lr += lucro
    falha_total_lr += falha
    sucesso_total_lr += sucesso
    if compra == 1:
        combinado_compra2.append(i)
    elif venda == 1:
        combinado_venda2.append(i)

if len(combinado_compra1) <= len(combinado_compra2):
    c = len(combinado_compra1)
else:
    c = len(combinado_compra2)
 
combinado_compra3 = []    


for a in range(0, c):
    for b in range(0, c):
        if combinado_compra1[a] == combinado_compra2[b]:
            combinado_compra3.append(combinado_compra2[b])
     
falha_total_compra = 0
sucesso_total_compra = 0
lucro_total_compra = 0
prejuizo_total_compra = 0


combinado_compra3 = combinado_compra3[::-1]    

for i in combinado_compra3:
    print(i)
    X_test_lr, y_test_lr, X_train_lr, y_train_lr = treinar(X_lr, y_lr, len(real_price) - i)
 
    fechamento_atual = y_train_lr

    scaler = StandardScaler()

    X_train_lr = scaler.fit_transform(X_train_lr)
    X_test_lr = scaler.transform(X_test_lr)
    
    LR = LinearRegression()
    LR.fit(X_train_lr, y_train_lr)
    predictionsLR = LR.predict(X_test_lr)
    print("O MSE foi: ", mean_squared_error(predictionsLR, y_test_lr))
    print("\nTamanho de teste é: ", len(X_test_lr))

    lucro, falha, sucesso, prejuizo, compra, venda = prever_resultado(X_test_lr, y_test_lr,fechamento_atual)
    
    #lucro, falha, sucesso, prejuizo, compra, venda = prever_resultado_lstm(real_price[i], predicted_stock_price[i],fechamento_anterior[i-1])

    prejuizo_total_compra += prejuizo
    lucro_total_compra += lucro
    falha_total_compra += falha
    sucesso_total_compra += sucesso



if len(combinado_venda1) <= len(combinado_venda2):
    c = len(combinado_venda1)
else:
    c = len(combinado_venda2)
 
combinado_venda3 = []    

for a in range(0, c):
    for b in range(0, c):
        if combinado_venda1[a] == combinado_venda2[b]:
            combinado_venda3.append(combinado_venda2[b])
            
falha_total_venda = 0
sucesso_total_venda = 0
lucro_total_venda = 0
prejuizo_total_venda = 0

combinado_venda3 = combinado_venda3[::-1]

for i in combinado_venda3:
    print(i)
    
    X_test_lr, y_test_lr, X_train_lr, y_train_lr = treinar(X_lr, y_lr, len(real_price) - i)

    fechamento_atual = y_train_lr
    
    scaler = StandardScaler()
    
    X_train_lr = scaler.fit_transform(X_train_lr)
    X_test_lr = scaler.transform(X_test_lr)

    LR = LinearRegression()
    LR.fit(X_train_lr, y_train_lr)
    predictionsLR = LR.predict(X_test_lr)
    print("O MSE foi: ", mean_squared_error(predictionsLR, y_test_lr))
    print("\nTamanho de teste é: ", len(X_test_lr))
    

    lucro, falha, sucesso, prejuizo, compra, venda = prever_resultado(X_test_lr, y_test_lr,fechamento_atual)
    
    #lucro, falha, sucesso, prejuizo, compra, venda = prever_resultado_lstm(real_price[i], predicted_stock_price[i],fechamento_anterior[i-1])

    prejuizo_total_venda += prejuizo
    lucro_total_venda += lucro
    falha_total_venda += falha
    sucesso_total_venda += sucesso
    

print("Resultados Venda combinada")
print("Sucesso Total de Venda é: ", sucesso_total_venda)
print("Falha total de Venda é: ", falha_total_venda)      
print("lucro Total de Venda é: ", lucro_total_venda)
print("Prejuizo total de Venda é: ",  prejuizo_total_venda)  
print("\n\n") 

print("Resultados Compra Combinada")
print("Sucesso Total de compra é: ", sucesso_total_compra)
print("Falha total de compra é: ", falha_total_compra)      
print("lucro Total de compra é: ", lucro_total_compra)
print("Prejuizo total de compra é: ", prejuizo_total_compra)  
print("\n\n") 

print("Resultados LSTM")
print("Sucesso Total  é: ", sucesso_total_lstm)
print("Falha total é: ", falha_total_lstm)      
print("lucro Total é: ", lucro_total_lstm)
print("Prejuizo total é: ", prejuizo_total_lstm)
print("Sucesso compra Total  é: ", sucesso_compra_lstm)
print("Falha compra é: ", falha_compra_lstm)
print("Sucesso venda total é: ", sucesso_venda_lstm)      
print("Falha venda é: ", falha_venda_lstm)
print("\n\n") 

print("Resultados LR")
print("Sucesso Total é: ", sucesso_total_lr)
print("Falha total é: ", falha_total_lr)      
print("lucro Total é: ", lucro_total_lr)
print("Prejuizo total é: ", prejuizo_total_lr) 
print("\n\n") 

#plot
plt.plot(real_price, color = 'red', label = 'Real Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
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