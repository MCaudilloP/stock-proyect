import pandas as pd
import numpy as np

# Definir el rango de fechas de los últimos 5 años
fecha_inicio = '2019-01-01'
fecha_fin = '2023-12-31'

# Crear un rango de fechas diarias
rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')

# Crear un DataFrame vacío para almacenar las ventas simuladas
ventas_df = pd.DataFrame(index=rango_fechas)

# Generar datos simulados de ventas diarias
ventas_df['ventas_diarias'] = np.random.randint(50, 200, size=len(rango_fechas))

# Agregar una columna para simular el tipo de producto vendido (por ejemplo)
productos = ['Vodka', 'Whisky', 'Ron', 'Tequila', 'Ginebra']
ventas_df['tipo_producto'] = np.random.choice(productos, size=len(rango_fechas))

# Agregar una columna para simular el precio unitario de venta
# Supongamos precios promedio de cada tipo de licor
precios_promedio = {'Vodka': 20, 'Whisky': 30, 'Ron': 25, 'Tequila': 35, 'Ginebra': 25}
ventas_df['precio_unitario'] = ventas_df['tipo_producto'].map(precios_promedio)

# Calcular el total de ventas diarias
ventas_df['total_ventas_diarias'] = ventas_df['ventas_diarias'] * ventas_df['precio_unitario']

# Visualizar el DataFrame simulado de ventas
print(ventas_df.head())
