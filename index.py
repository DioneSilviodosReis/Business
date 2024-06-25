import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


clientes = pd.read_csv('clientes.csv')
vendas = pd.read_csv('venda.csv')
produtos = pd.read_csv('produtos.csv')
produtos_vendidos = pd.read_csv('produtos_vendidos.csv')
formas_pagamento = pd.read_csv('formas_pagamento.csv')

top_setores = clientes['setor'].value_counts().nlargest(10)

data_nascimento = pd.to_datetime(clientes['data_nascimento'], errors='coerce')
data_nascimento  = data_nascimento.dropna()
ano = data_nascimento.dt.year
ano_atual = pd.Timestamp.now().year
idade = ano_atual - ano
clientes['idade'] = idade

faixas_etarias = [0, 18, 30, 40, 50, 60, 100]
labels_faixas = ['0-17', '18-29', '30-39', '40-49', '50-59', '60+'] 


clientes['faixa_etaria'] = pd.cut(idade, bins=faixas_etarias, labels=labels_faixas)

contagem_faixas = clientes['faixa_etaria'].value_counts().sort_index()


plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(contagem_faixas)))
contagem_faixas.plot(kind='bar', color=colors)
plt.title('Distribuição de Clientes por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(contagem_faixas)))
top_setores.plot(kind='barh', color=colors)
plt.title('Top 10 Setores com Maior Número de Clientes')
plt.xlabel('Número de Clientes')
plt.ylabel('Setor')
plt.gca().invert_yaxis() 
plt.show()

clientes_vendas = pd.merge(clientes, vendas, left_on='Codigo', right_on='cliente', how='inner')

vendas_por_faixa = clientes_vendas['faixa_etaria'].value_counts()


porcentagens = vendas_por_faixa / vendas_por_faixa.sum() * 100


plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(porcentagens)))
plt.pie(porcentagens, labels=porcentagens.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Proporção de Vendas por Faixa Etária')
plt.axis('equal')  
plt.tight_layout()
plt.show()

clientes_vendas = pd.merge(clientes, vendas, left_on='Codigo', right_on='cliente', how='inner')

contagem_vendas_genero = clientes_vendas['genereo'].value_counts()


porcentagens = contagem_vendas_genero / contagem_vendas_genero.sum() * 100


plt.figure(figsize=(8, 6))
colors = ['lightcoral', 'lightblue']
plt.pie(porcentagens, labels=porcentagens.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Proporção de Vendas por Gênero')
plt.axis('equal')  
plt.tight_layout()
plt.show()


vendas_formas_pagamento = pd.merge(vendas, formas_pagamento, left_on='codigo_forma_pagamento', right_on='codigo_forma_pagamento', how='inner')


contagem_formas_pagamento = vendas_formas_pagamento['forma_pagamento'].value_counts()


plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(contagem_formas_pagamento)))
contagem_formas_pagamento.plot(kind='barh', color=colors)
plt.title('Formas de Pagamento Mais Utilizadas pelos Clientes')
plt.xlabel('Número de Transações')
plt.ylabel('Forma de Pagamento')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Previsões
clientes['genereo'].fillna('Outro', inplace=True)  
clientes['setor'].fillna('Outro', inplace=True)   
clientes['idade'].fillna(clientes['idade'].mean(), inplace=True) 

# Convertendo variáveis categóricas para numéricas usando LabelEncoder
le = LabelEncoder()
clientes['genereo'] = le.fit_transform(clientes['genereo'])
clientes['setor'] = le.fit_transform(clientes['setor'])

# Selecionar features para o modelo
X = clientes[['genereo', 'setor']]
y = clientes['idade']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir o modelo de Random Forest para regressão
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Médio Absoluto (MAE): {mae:.2f}')
print(f'Erro Médio Quadrático (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação (R²): {r2:.2f}')