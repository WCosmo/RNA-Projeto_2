# RNA - Projeto 2

Uma implementação simples do algoritmo backpropagation para a disciplina de RNA do PPGEE/UFPA.

# Como executar

`simple_backpropagation.py --i (int) --h (int) --o (int) --ep (int) --sd (int) --pa (int) --lr (float) --h_func (str) --o_func (str) --pf (str)`

# Parâmetros

- --i, Número de neurônios da camada de entrada, default = 5
- --h, Número de neurônios da camada escondida, default = 1
- --o, Número de neurônios da camada de saída, default = 1
- --ep, Número de épocas, default = 100
- --sd, Seed para gerar os números aleatórios, default = 7
- --pa, Paciência do critério de parada, default = 20
- --lr, Taxa de aprendizado, default = 0.0001
- --h_func, Função de ativação da camada oculta podendo ser sigmoid, tanh ou linear, default = 'sigmoid'
- --o_func, Função de ativação da camada de saída podendo ser sigmoid, tanh ou linear, default = 'linear'
- --pf, Local do arquivo do dataset na extensão .xlsx, default = './dataset/dadosmamografia.xlsx'

