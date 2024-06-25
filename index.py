import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
clientes_df = pd.read_csv('Clientes.csv', sep=';', encoding='latin1')
vendas_df = pd.read_csv('Vendas.csv', sep=';', encoding='latin1')
produtos_vendidos_df = pd.read_csv('Produtos_Vendidos.csv', sep=';', encoding='latin1')

# Ajustar a codificação e converter as datas
clientes_df['cidade'] = clientes_df['cidade'].str.encode('latin1').str.decode('utf-8')
clientes_df['data_nascimento'] = pd.to_datetime(clientes_df['data_nascimento'], errors='coerce')
vendas_df['DATA'] = pd.to_datetime(vendas_df['DATA'], errors='coerce')

# Resumo das vendas por data
vendas_por_data = vendas_df.resample('M', on='DATA').sum()

# Plotar o gráfico de linhas do tempo das vendas
plt.figure(figsize=(12, 6))
plt.plot(vendas_por_data.index, vendas_por_data['valor'], marker='o', linestyle='-')
plt.title('Vendas ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Valor Total das Vendas')
plt.grid(True)
plt.show()


# Gráfico de dispersão do valor das vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(x='DATA', y='valor', data=vendas_df)
plt.title('Dispersão do Valor das Vendas')

plt.xlabel('Data')
plt.ylabel('Valor das Vendas')
plt.show()

# Gráfico de dispersão do limite de crédito por idade dos clientes
clientes_df['idade'] = (pd.to_datetime('today') - clientes_df['data_nascimento']).dt.days // 365
plt.figure(figsize=(10, 6))
sns.scatterplot(x='idade', y='limite', data=clientes_df)
plt.title('Dispersão do Limite de Crédito por Idade dos Clientes')
plt.xlabel('Idade')
plt.ylabel('Limite de Crédito')
plt.show()

# Calcular a idade dos clientes
clientes_df['idade'] = (pd.to_datetime('today') - clientes_df['data_nascimento']).dt.days // 365

# Medidas de dispersão e centralidade para vendas
vendas_estatisticas = vendas_df['valor'].describe()
vendas_media = vendas_df['valor'].mean()
vendas_mediana = vendas_df['valor'].median()
vendas_desvio_padrao = vendas_df['valor'].std()

# Medidas de dispersão e centralidade para limites de crédito
limites_estatisticas = clientes_df['limite'].describe()
limites_media = clientes_df['limite'].mean()
limites_mediana = clientes_df['limite'].median()
limites_desvio_padrao = clientes_df['limite'].std()

# Exibir as estatísticas
print("Estatísticas das Vendas:")
print(vendas_estatisticas)
print(f"Média: {vendas_media}")
print(f"Mediana: {vendas_mediana}")
print(f"Desvio Padrão: {vendas_desvio_padrao}")

print("\nEstatísticas dos Limites de Crédito:")
print(limites_estatisticas)
print(f"Média: {limites_media}")
print(f"Mediana: {limites_mediana}")
print(f"Desvio Padrão: {limites_desvio_padrao}")
