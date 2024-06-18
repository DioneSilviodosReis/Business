import pandas as pd
import matplotlib.pyplot as plt

clientes = pd.read_csv('clientes.csv')

top_setores = clientes['setor'].value_counts().nlargest(10)

# Plotar o gráfico de barras horizontal
plt.figure(figsize=(10, 6))
top_setores.plot(kind='barh', color='skyblue')
plt.title('Top 10 Setores com Maior Número de Clientes')
plt.xlabel('Número de Clientes')
plt.ylabel('Setor')
plt.gca().invert_yaxis()  # Inverter a ordem dos setores no eixo y para maior para menor
plt.show()