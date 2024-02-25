import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simular datos de ventas (como se generó anteriormente)
# Consideraremos solo la columna de ventas diarias para simplificar
ventas_df = pd.DataFrame({
    'fecha': pd.date_range(start='2019-01-01', end='2023-12-31', freq='D'),
    'ventas_diarias': np.random.randint(50, 200, size=1826)
})

# Normalizar los datos
scaler = MinMaxScaler()
ventas_diarias = scaler.fit_transform(ventas_df['ventas_diarias'].values.reshape(-1, 1))

# Preparar datos para LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Definir el número de pasos de tiempo
n_steps = 7

# Preparar los datos en secuencias para el modelo LSTM
X, y = prepare_data(ventas_diarias, n_steps)

# Reformatear los datos para que sean compatibles con LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# Ajustar el modelo
model.fit(X, y, epochs=12, verbose=1)

# Hacer una predicción para los próximos 7 días
ultimos_7_dias = ventas_diarias[-n_steps:]
X_input = ultimos_7_dias.reshape((1, n_steps, 1))
prediccion = model.predict(X_input, verbose=0)

# Desescalar la predicción
prediccion_desescalada = scaler.inverse_transform(prediccion)

print("Predicción de ventas para los próximos 7 días:", prediccion_desescalada)

