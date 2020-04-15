
# Ajustes linear da dados experimentais de Arrhenius

# importando os pacotes do python
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Lendo o arquivo csv na pasta
dados = pd.read_csv('Arrhenius.csv')

#transformando data frame em vetor
dados_np = dados.values

#vetor do eixo x
x = 1/dados_np[:,0]
#vetor do eixo y
y = np.log(dados_np[:,1])

#Ajustando a correlação linear
linear_p = np.polyfit(x, y, 1)

Ea = linear_p[0] *(-8.314) # Estimativa da energia de ativação em J /mol
A = np.exp(linear_p[1]) # Fator de frequencia

#impressão dos dados experimentais e do modelo linear
plt.plot(x, y, "o", label = "pontos experimentais")
plt.plot(x, np.polyval(linear_p, x), "-r", label = "modelo")
plt.xlabel("1 / Temperatura")
plt.ylabel("Log de k")
plt.rcParams['figure.figsize'] = (11,7)
plt.title("Ajuste de uma reta")
plt.legend()
plt.show()

yfit = linear_p[0] * x + linear_p[1] # calcula os valores preditos
yresid = y - yfit # resíduo = valor real - valor ajustado (valor predito)
SQresid = sum(pow(yresid,2)) # soma dos quadrados dos resíduos 
SQtotal = len(y) * np.var(y) # número de elementos do vetor y vezes a variância de y
R2 = 1 - SQresid/SQtotal # coeficiente de determinação


#dados do modelo ajustado
dados_v = np.transpose(np.exp(yfit))
dados_df = pd.DataFrame({"Temperatura":dados_np[:,0], "velocidade_mod":dados_v, "velocidade_exp":dados_np[:,1]})
print(dados_df)

print(f'A energia de ativação é: {Ea:9.7} Joules/mol')
print(f'O fator de frequencia é: {A:9.5} 1/s')
print(f'O Coeficiente de ajuste é: {R2:5.5}')



# Ajustes não linear da dados experimentais de Arrhenius

# importando os pacotes do python
# Lendo o arquivo csv na pasta
dados = pd.read_csv('Arrhenius.csv')

#transformando data frame em vetor
dados_np = dados.values

#vetor do eixo x
x = dados_np[:, 0]
print(x)

#vetor do eixo y
y = dados_np[:, 1]
print(y)

# Definição do modelo
def Arr(x, A, E):
  R = 8.314  # constante dos gases em Joule/ (mol * K)
  return A * np.exp(-(E/(R*x)))

# Chamada de curve_fit
popt, pcov = curve_fit(Arr, x, y, p0=(10000, 10), maxfev=2000)
p1, p2 = popt

# Impressão dos parâmetros ótimos
print("Os parâmetros ajustados são p1=%g, p2=%g" % (p1, p2))

y_ajustado = Arr(x, p1, p2)
#impressão dos dados experimentais e do modelo
plt.plot(x, y, "o", label="pontos experimentais")
plt.plot(x, y_ajustado, "-r", label="modelo")
plt.xlabel("Temperatura")
plt.ylabel("Velocidade de reação")
plt.rcParams['figure.figsize'] = (11, 7)
plt.title("Ajuste de uma função Arrhenius")
plt.legend()
plt.show()

r2 = 1. - sum((Arr(x, p1, p2) - y) ** 2) / sum((y - np.mean(y)) ** 2)

#dados do modelo ajustado
dados_df = pd.DataFrame(
    {"Temperatura": x, "velocidade_mod": y_ajustado, "velocidade_exp": y})
print(dados_df)

print(f'A energia de ativação é: {p2:9.7} Joules/mol')
print(f'O fator de frequencia é: {p1:9.5} 1/s')
print(f'O Coeficiente de ajuste é: {r2:5.4}')





