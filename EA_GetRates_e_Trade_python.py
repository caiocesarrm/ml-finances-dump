# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:17:43 2018

@author: caioc
"""

import sys
import pandas
import matplotlib.pyplot as plt
import seaborn
import time
import talib as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


print(sys.version)

# In[Parametros]:

n_previsoes = 50
lista_de_simbolos = ['ABEV3', 'B3SA3', 'BBAS3', 'BBDC4', 'ITUB4', 'MGLU3', 'PETR4', 'VALE3']

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
    df = df['2019-01-13' : ]

    df['Close_t1'] = df['Close'].shift(-1)
    df[simbolo_ativo] = ta.ROC(df.Close_t1.values, timeperiod = 1)
    #df["{simbolo_ativo}_t1".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 2)
    #df["{simbolo_ativo}_t2".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 3)
    #df["{simbolo_ativo}_t3".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 4)
    #df["{simbolo_ativo}_t4".format(simbolo_ativo = simbolo_ativo)] = ta.ROC(df.Close.values, timeperiod = 5)
    df = df.drop(['Voltick', 'High', 'Open', 'Low', 'Volume', 'Close', 'Close_t1'],axis=1)
    
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
    #df = df['2019-02-13' : '2019-02-15']
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

df_target = dataframe_principal('WING19M1')

df_final = pandas.concat([df_ativos, df_target], axis=1, sort=False)
df_final.dropna(inplace = True)

media = df_final['WING19M1'].mean()



X = df_final.drop(['WING19M1'],axis=1)
y = df_final['WING19M1']

variacao = 0

# In[Regressão Linear]:

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



clfRF = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
clfRF.fit(X_train, y_train)
previsao = clfRF.predict(X_test)
# In[11]:

def ler_prever():
    df = pandas.read_csv('dados_metatrader.csv', 
                     parse_dates={'Date' : ['Data', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='Date')    
    
    df = df.astype('float')
    
    df['ABEV3M1'] = ta.ROC(df.ABEV3M1.values, timeperiod = 1)
    for i in range(len(df['ABEV3M1'])):
        if(df['ABEV3M1'].iloc[i] >= 0):
            df['ABEV3M1'].iloc[i] = 1
        else:
     
            df['ABEV3M1'].iloc[i] = 0
    
    df['B3SA3M1'] = ta.ROC(df.B3SA3M1.values, timeperiod = 1)
    for i in range(len(df['B3SA3M1'])):
        if(df['B3SA3M1'].iloc[i] >= 0):
            df['B3SA3M1'].iloc[i] = 1
        else:
     
            df['B3SA3M1'].iloc[i] = 0
    
    df['BBAS3M1'] = ta.ROC(df.BBAS3M1.values, timeperiod = 1)
    for i in range(len(df['BBAS3M1'])):
        if(df['BBAS3M1'].iloc[i] >= 0):
            df['BBAS3M1'].iloc[i] = 1
        else:
     
            df['BBAS3M1'].iloc[i] = 0
            
    df['BBDC4M1'] = ta.ROC(df.BBDC4M1.values, timeperiod = 1)
    for i in range(len(df['BBDC4M1'])):
        if(df['BBDC4M1'].iloc[i] >= 0):
            df['BBDC4M1'].iloc[i] = 1
        else:
     
            df['BBDC4M1'].iloc[i] = 0
    
    df['ITUB4M1'] = ta.ROC(df.ITUB4M1.values, timeperiod = 1)
    for i in range(len(df['ITUB4M1'])):
        if(df['ITUB4M1'].iloc[i] >= 0):
            df['ITUB4M1'].iloc[i] = 1
        else:
     
            df['ITUB4M1'].iloc[i] = 0
    
    df['MGLU3M1'] = ta.ROC(df.MGLU3M1.values, timeperiod = 1)
    for i in range(len(df['MGLU3M1'])):
        if(df['MGLU3M1'].iloc[i] >= 0):
            df['MGLU3M1'].iloc[i] = 1
        else:
     
            df['MGLU3M1'].iloc[i] = 0
            
    df['PETR4M1'] = ta.ROC(df.PETR4M1.values, timeperiod = 1)
    for i in range(len(df['PETR4M1'])):
        if(df['PETR4M1'].iloc[i] >= 0):
            df['PETR4M1'].iloc[i] = 1
        else:
     
            df['PETR4M1'].iloc[i] = 0
            
    df['VALE3M1'] = ta.ROC(df.VALE3M1.values, timeperiod = 1)
    for i in range(len(df['VALE3M1'])):
        if(df['VALE3M1'].iloc[i] >= 0):
            df['VALE3M1'].iloc[i] = 1
        else:
     
            df['VALE3M1'].iloc[i] = 0
            
    df.dropna(inplace = True)
    
    return df

#executar para decidir se vai comprar ou vender
def prever_atual():
    df = ler_prever()
    df.dropna(inplace = True)
    previsao = clfRF.predict(df)
    resultado = previsao[-1]
    return resultado


# In[12]:   

#lista_de_simbolos = ['ABEV3', 'B3SA3', 'BBAS3', 'BBDC4', 'BRKM5', 'ELET3', 'GGBR4', 'ITUB4', 'LAME4', 'LREN3', 'MGLU3', 'NATU3', 'PCAR4', 'PETR4', 'SANB11', 'SUZB3', 'TIMP3', 'USIM5', 'VALE3']

# IMPORT zmq library
import zmq

#trade-action-type-symbol-open/close price-StopLoss-TakeProfit-Tradecomment
# Sample Commands for ZeroMQ MT4 EA
eurusd_buy_order = b"TRADE|OPEN|0|EURUSD|0|50|50|Python-to-MT4"
winJ19_buy_order = b"TRADE|OPEN|0|WINJ19|0|50|50|Python-to-MT4"
winJ19_encerrar_order = b"TRADE|ENCERRAR|0|WINJ19|0|50|50|Python-to-MT4"
winJ19_sell_order = b"TRADE|OPEN|1|WINJ19|0|50|50|Python-to-MT4"
usdjpy_buy_order = b"TRADE|OPEN|0|USDJPY|0|50|50|Python-to-MT4"
eurusd_sell_order = b"TRADE|OPEN|1|EURUSD|0|50|50|Python-to-MT4"
eurusd_closebuy_order = "TRADE|CLOSE|0|EURUSD|0|50|50|Python-to-MT4"
get_rates = b"DATA|WINJ19"

# Create ZMQ Context
context_script = zmq.Context()

# Create REQ Socket
reqSocket_script = context_script.socket(zmq.REQ)
reqSocket_script.connect(b"tcp://localhost:5557")

# Create PULL Socket
pullSocket_script = context_script.socket(zmq.PULL)
pullSocket_script.connect(b"tcp://localhost:5558")

# Create ZMQ Context
context = zmq.Context()

# Create REQ Socket
reqSocket = context.socket(zmq.REQ)
reqSocket.connect(b"tcp://localhost:5555")

# Create PULL Socket
pullSocket = context.socket(zmq.PULL)
pullSocket.connect(b"tcp://localhost:5556")
    

def zeromq_mt4_script_client():
        
    # Send RATES command to ZeroMQ MT4 EA
    for symbol in lista_de_simbolos:
        byte_array_conversion = symbol.encode()
        receber_close_simbolo = b"DATA|" + byte_array_conversion
        print(receber_close_simbolo)
        remote_send_rates(reqSocket_script, receber_close_simbolo)
    
    print("testeeee")
    # Send BUY EURUSD command to ZeroMQ MT4 EA
    # remote_send(reqSocket, eurusd_buy_order)
    
    # Send CLOSE EURUSD command to ZeroMQ MT4 EA. You'll need to append the 
    # trade's ORDER ID to the end, as below for example:
    # remote_send(reqSocket, eurusd_closebuy_order + "|" + "12345678")
    
    # PULL from pullSocket
    remote_pull(pullSocket_script)
    
def remote_send_rates(socket, data):
    
    try:
        socket.send(data)
        msg = socket.recv_string()

        msg = msg.replace(" ", ",")
        verificar_primeiro = data.decode('utf-8')
        verificar_primeiro = verificar_primeiro[5:]
        if(msg != ""):
            if(verificar_primeiro == lista_de_simbolos[0]):
                with open('dados_metatrader.csv', "a") as f:
                    f.write("\n"+msg)
                    f.close()
            else:
                msg = msg[17: ]
                with open('dados_metatrader.csv', "a") as f:
                    f.write(","+msg)
                    f.close()
    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")


# Sample Function for Client
def zeromq_mt4_ea_client(ordem):

    # Send RATES command to ZeroMQ MT4 EA
    
    for i in range(1):
        remote_send(reqSocket, ordem)
    
        
    # Send BUY EURUSD command to ZeroMQ MT4 EA
    # remote_send(reqSocket, eurusd_buy_order)
    
    # Send CLOSE EURUSD command to ZeroMQ MT4 EA. You'll need to append the 
    # trade's ORDER ID to the end, as below for example:
    # remote_send(reqSocket, eurusd_closebuy_order + "|" + "12345678")
    
    # PULL from pullSocket
    remote_pull(pullSocket)
    
# Function to send commands to ZeroMQ MT4 EA
def remote_send(socket, data):
    
    try:
        socket.send(data)
        msg = socket.recv_string()
        print(msg)
    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")
    
# Function to retrieve data from ZeroMQ MT4 EA
def remote_pull(socket):
    
    try:
        msg = socket.recv(flags=zmq.NOBLOCK)
        print(msg)
        return msg
         
    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")
        
#zeromq_mt4_ea_client()
#zeromq_mt4_script_client()



# Run Tests
#usar 1 socket para receber os dados case 4 no EA
#usar o outro socket para enviar ordens case 1 no EA
while True:
    print("entrei1")
    zeromq_mt4_script_client()
    resultado = prever_atual()
    print("Resultado foi:", resultado)
    if resultado == 1:
        zeromq_mt4_ea_client(winJ19_buy_order)
    else:
        zeromq_mt4_ea_client(winJ19_encerrar_order)
        print("ordem de venda, compra não executada")
    
    time.sleep(1)


#usar um socket para receber dados e enviar ordens, case 4 no EA
'''
while True:
    print("entrei1")
    zeromq_mt4_script_client()
    resultado = prever_atual()
    if resultado == 1:
        zeromq_mt4_ea_client()
       
        print("Aguardando proximo minuto")   
    time.sleep(1)
'''