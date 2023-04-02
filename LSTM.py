import numpy as np
import pandas as pd
import tensorflow as tf

# Carregando dados de entrada em CSV
data = pd.read_csv('dados_exer_aula_8.csv')

# Separando dados de treinamento e teste
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Normalizando os dados de entrada
data_mean = train_data.mean()
data_std = train_data.std()
train_data = (train_data - data_mean) / data_std
test_data = (test_data - data_mean) / data_std

# Convertendo dados de entrada em arrays numpy
x_train = np.array(train_data.drop(columns=['target']))
y_train = np.array(train_data['target'])
x_test = np.array(test_data.drop(columns=['target']))
y_test = np.array(test_data['target'])

# Criando modelo LSTM
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dense(1))

# Compilando modelo
model.compile(loss='mse', optimizer='adam')

# Treinando modelo
model.fit(x_train.reshape(x_train.shape[0], x_train.shape[1], 1), y_train, epochs=100, batch_size=16, validation_split=0.2)

# Avaliando modelo
test_loss = model.evaluate(x_test.reshape(x_test.shape[0], x_test.shape[1], 1), y_test)
print(f'Test loss: {test_loss}')