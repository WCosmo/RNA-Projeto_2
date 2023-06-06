"""
RNA - Projeto 2
Projeto: Algoritmo Backpropagation
Estudantes: Wilson Cosmo, Artur Machado
Data: 29/05/2023

"""

import matplotlib.pyplot as plt
from datetime import datetime  #to create timestamps
import pandas as pd
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=5, help='Número de neurônios da camada de entrada')
parser.add_argument('--h', type=int, default=1, help='Número de neurônios da camada escondida')
parser.add_argument('--o', type=int, default=1, help='Número de neurônios da camada de saída')
parser.add_argument('--ep', type=int, default=100, help='Número de épocas')
parser.add_argument('--sd', type=int, default=7, help='Seed para gerar os números aleatórios')
parser.add_argument('--pa', type=int, default=20, help='Paciência do critério de parada')
parser.add_argument('--lr', type=float, default=0.0001, help='Taxa de aprendizado')
parser.add_argument('--h_func', type=str, default='sigmoid', help='Função de ativação da camada oculta')
parser.add_argument('--o_func', type=str, default='linear', help='Função de ativação da camada de saída')
parser.add_argument('--pf', type=str, default='./dataset/dadosmamografia.xlsx', help='Local do arquivo .xlsx')
opt = parser.parse_args()

random.seed(opt.sd) #define a seed to reproduce the results

raw_data = pd.read_excel(opt.pf)

all_c = []
input_c = []
output_c = []

for i in range(opt.i):
  all_c.append('input'+str(i))
  input_c.append('input'+str(i))

for j in range(opt.o):
  all_c.append('output'+str(j))
  output_c.append('output'+str(j))

raw_data.columns = [all_c]

normalized_data = (raw_data-raw_data.min())/(raw_data.max()-raw_data.min())

X = normalized_data[input_c].to_numpy()
Y = normalized_data[output_c].to_numpy()

#Erro médio quadrático
def mse(a, b):
  return np.sqrt(np.mean((a - b) ** 2))

#Função de ativação sigmoid
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#Derivada da função de ativação sigmoid
def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))

#Função de ativação tangente hiperbólica
def tanh(x):
    return np.tanh(x)

#Derivada da função de ativação tangente hiperbólica (tanh)
def d_tanh(x):
    return 1 - np.tanh(x)**2

#Função de ativação linear
def linear(x):
  return x #f(x) = x

#Derivada da função de ativação linear
def d_linear(x):
    return 1 * pow(x, 0)

#Classe da Rede Neural
class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, h_function='sigmoid', o_function='linear'):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.h_function = h_function
    self.o_function = o_function

    #Inicialização dos pesos aleatórios
    self.weights_hidden = np.zeros((self.input_size, self.hidden_size))
    self.weights_output = np.zeros((self.hidden_size, self.output_size))

    for i in range(self.input_size):
      for j in range(self.hidden_size):
        self.weights_hidden[i,j] = random.random()

    for k in range(self.hidden_size):
      for l in range(self.output_size):
        self.weights_output[k,l] = random.random()


  def feedforward(self, X):
    #Cálculo da saída da camada oculta
    if(self.h_function == 'sigmoid'):
      hidden_layer_output = sigmoid(np.dot(X, self.weights_hidden))
    elif(self.h_function == 'tanh'):
      hidden_layer_output = tanh(np.dot(X, self.weights_hidden))
    else:
      hidden_layer_output = linear(np.dot(X, self.weights_hidden))

    #Cálculo da saída final
    if(self.o_function == 'sigmoid'):
      output_layer_output = sigmoid(np.dot(hidden_layer_output, self.weights_output))
    elif(self.o_function == 'tanh'):
      output_layer_output = tanh(np.dot(hidden_layer_output, self.weights_output))
    else:
      output_layer_output = linear(np.dot(hidden_layer_output, self.weights_output))

    return output_layer_output

  def backpropagation(self, X, y, learning_rate):
    #Feedforward
    if(self.h_function == 'sigmoid'):
      hidden_layer_output = sigmoid(np.dot(X, self.weights_hidden))
    elif(self.h_function == 'tanh'):
      hidden_layer_output = tanh(np.dot(X, self.weights_hidden))
    else:
      hidden_layer_output = linear(np.dot(X, self.weights_hidden))

    if(self.o_function == 'sigmoid'):
      output_layer_output = sigmoid(np.dot(hidden_layer_output, self.weights_output))
    elif(self.o_function == 'tanh'):
      output_layer_output = tanh(np.dot(hidden_layer_output, self.weights_output))
    else:
      output_layer_output = linear(np.dot(hidden_layer_output, self.weights_output))


    #Cálculo do erro
    output_error = mse(y, output_layer_output)

    #Cálculo dos gradientes
    if(self.o_function == 'sigmoid'):
      output_gradient = output_error * d_sigmoid(output_layer_output)
    elif(self.o_function == 'tanh'):
      output_gradient = output_error * d_tanh(output_layer_output)
    else:
      output_gradient = output_error * d_linear(output_layer_output)

    if(self.h_function == 'sigmoid'):
      hidden_gradient = np.dot(output_gradient, self.weights_output.T) * d_sigmoid(hidden_layer_output)
    elif(self.h_function == 'tanh'):
      hidden_gradient = np.dot(output_gradient, self.weights_output.T) * d_tanh(hidden_layer_output)
    else:
      hidden_gradient = np.dot(output_gradient, self.weights_output.T) * d_linear(hidden_layer_output)


    #Atualização dos pesos
    self.weights_output += learning_rate * np.dot(hidden_layer_output.T, output_gradient)
    self.weights_hidden += learning_rate * np.dot(X.T, hidden_gradient)

  def predict(self, X): #realiza uma predição
    return self.feedforward(X)

  def b_accuracy(self, X, y): #avalia a acurácia do modelo considerando uma saída binária
    predictions = self.predict(X)
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == y)
    return accuracy

  def b_mse(self, X, y): #avalia a acurácia do modelo considerando uma saída binária
    predictions = self.predict(X)
    mm = mse(predictions, y)
    return mm

  def train(self, X, y, epochs, learning_rate, validation_split=0.2, patience=20):
    #Dividir os dados em conjuntos de treinamento e validação
    num_samples = X.shape[0]
    num_validation_samples = int(num_samples * validation_split)
    num_training_samples = num_samples - num_validation_samples

    training_X = X[:num_training_samples]
    training_y = y[:num_training_samples]
    validation_X = X[num_training_samples:]
    validation_y = y[num_training_samples:]

    #VAriáveis para o plot:
    t_acc = []
    v_acc = []

    t_mse = []
    v_mse = []

    buff_acc = 0 #buffer da acurácia
    buff_epoch = 0
    pc = 0 #contador da paciencia

    print('\nTreinamento iniciado:\n- Numero de épocas: ', epochs, '\n- Taxa de aprendizado: ', learning_rate, '\n- Paciência: ', patience, '\n')
    for epoch in range(epochs):
      #Treinar a rede com os dados de treinamento
      self.backpropagation(training_X, training_y, learning_rate)

      #Calcular a acurácia no conjunto de treinamento
      training_accuracy = self.b_accuracy(training_X, training_y)
      training_mse = self.b_mse(training_X, training_y)

      t_acc.append(training_accuracy)
      t_mse.append(training_mse)

      #Calcular a acurácia no conjunto de validação
      validation_accuracy = self.b_accuracy(validation_X, validation_y)
      validation_mse = self.b_mse(validation_X, validation_y)

      v_acc.append(validation_accuracy)
      v_mse.append(validation_mse)

      #Exibir métricas de desempenho
      print(f"Época: {epoch} | Acurácia de treino: {training_accuracy:.2f} | Acurácia de validação: {validation_accuracy:.2f}")

      if validation_accuracy < buff_acc:
        pc = pc + 1
      else:
        h_buff = self.weights_hidden
        o_buff = self.weights_output
        buff_epoch = epoch
        buff_acc = validation_accuracy

      if pc > patience:
        print('Critério de parada por validação cruzada atingido!')
        print('Melhor resultado na época: ', buff_epoch)
        self.weights_hidden = h_buff
        self.weights_output = o_buff
        break

    t_acc = np.array(t_acc)
    v_acc = np.array(v_acc)

    t_mse = np.array(t_mse)
    v_mse = np.array(v_mse)

    plt.plot(t_acc, '-b', label='Treino')
    plt.plot(v_acc, '-r', label='Validação')
    plt.legend(loc='upper right')
    plt.title('Acurácia de Treinamento e Validação')
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid()
    plt.show()

    plt.plot(t_mse, '-b', label='Treino')
    plt.plot(v_mse, '-r', label='Validação')
    plt.legend(loc='upper right')
    plt.title('Erro médio quadrático de Treinamento e Validação')
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.grid()
    plt.show()


current_time = datetime.now()
timestamp = current_time.timestamp()
date_time = datetime.fromtimestamp(timestamp)
str_date_time = date_time.strftime('%d-%m-%Y_%H-%M-%S')
print('\nTimestamp de inicialização: ', str_date_time)

network3 = NeuralNetwork(opt.i, opt.h, opt.o, h_function=opt.h_func, o_function=opt.o_func) #declare the network

print('\nRede inicializada: \n- N entrada: ', network3.input_size, '\n- N oculto: ', network3.hidden_size, '\n- N saida: ', network3.output_size, '\n\n- F Ativação da camada oculta: ', network3.h_function, '\n- F Ativação da camada de saída: ', network3.o_function)

print('\nPesos da camada escondida: (pré treino)')
print(network3.weights_hidden)
print('\nPesos da camada de saída: (pré treino)')
print(network3.weights_output)

network3.train(X, Y, epochs=opt.ep, learning_rate=opt.lr, patience=opt.pa)

print('\nPesos da camada escondida: (pós treino)')
print(network3.weights_hidden)
print('\nPesos da camada de saída: (pós treino)')
print(network3.weights_output)
