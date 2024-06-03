import random
import numpy as np
import scipy.io

# Carregar coordenadas das cidades de um arquivo .mat
def carregar_coordenadas(arquivo):
       data = np.loadtxt(arquivo)
       return data

# Função de aptidão: calcula a distância total do percurso
def calcular_distancia(percurso, coordenadas):
    distancia_total = 0
    for i in range(len(percurso) - 1):
        cidade_atual = coordenadas[percurso[i]]
        proxima_cidade = coordenadas[percurso[i + 1]]
        distancia_total += np.sqrt((proxima_cidade[0] - cidade_atual[0])**2 + (proxima_cidade[1] - cidade_atual[1])**2)
    # Adiciona a distância de volta à cidade inicial
    cidade_atual = coordenadas[percurso[-1]]
    primeira_cidade = coordenadas[percurso[0]]
    distancia_total += np.sqrt((primeira_cidade[0] - cidade_atual[0])**2 + (primeira_cidade[1] - cidade_atual[1])**2)
    return distancia_total

# Geração inicial: criar uma população de percursos aleatórios
def gerar_populacao_inicial(tamanho_populacao, num_cidades):
    populacao = []
    for _ in range(tamanho_populacao):
        percurso = list(np.random.permutation(num_cidades))
        populacao.append(percurso)
    return populacao

# Seleção por roleta inversa
def selecao_roleta(populacao, coordenadas):
    distancias = np.array([calcular_distancia(individuo, coordenadas) for individuo in populacao])
    aptidoes = 1 / distancias
    soma_aptidoes = np.sum(aptidoes)
    probabilidades = aptidoes / soma_aptidoes
    escolhido = np.random.choice(len(populacao), p=probabilidades)
    return populacao[escolhido]

# Crossover: PMX (Partially Mapped Crossover)
def crossover_pmx(parent1, parent2):
    size = len(parent1)
    p1, p2 = [None]*size, [None]*size

    # Escolher dois pontos de crossover aleatórios
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    # Copiar segmento do primeiro pai para o filho
    p1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    p2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]

    # Mapeamento do PMX
    for i in range(cxpoint1, cxpoint2):
        if parent2[i] not in p1:
            while p1[i] is not None:
                i = parent2.index(parent1[i])
            p1[i] = parent2[i]
        if parent1[i] not in p2:
            while p2[i] is not None:
                i = parent1.index(parent2[i])
            p2[i] = parent1[i]

    # Preencher os restantes
    for i in range(size):
        if p1[i] is None:
            p1[i] = parent2[i]
        if p2[i] is None:
            p2[i] = parent1[i]

    return p1, p2

# Mutação: Swap de duas cidades
def mutacao(percurso, taxa_mutacao=0.01):
    for i in range(len(percurso)):
        if random.random() < taxa_mutacao:
            j = random.randint(0, len(percurso) - 1)
            percurso[i], percurso[j] = percurso[j], percurso[i]
    return percurso

# Algoritmo genético principal
def algoritmo_genetico(tamanho_populacao, num_geracoes, coordenadas):
    num_cidades = coordenadas.shape[0]
    populacao = gerar_populacao_inicial(tamanho_populacao, num_cidades)
    
    for _ in range(num_geracoes):
        # Ordenar a população pela aptidão (distância)
        populacao = sorted(populacao, key=lambda p: calcular_distancia(p, coordenadas))
        
        # Manter a metade da população atual
        nova_populacao = populacao[:tamanho_populacao // 2]
        
        # Gerar a outra metade por seleção e crossover
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_roleta(populacao, coordenadas)
            pai2 = selecao_roleta(populacao, coordenadas)
            filho1, filho2 = crossover_pmx(pai1, pai2)
            nova_populacao.append(mutacao(filho1))
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(mutacao(filho2))

        populacao = nova_populacao

    melhor_percurso = populacao[0]
    melhor_distancia = calcular_distancia(melhor_percurso, coordenadas)
    return melhor_percurso, melhor_distancia

# Parâmetros do algoritmo genético
tamanho_populacao = 100
num_geracoes = 500

# Carregar as coordenadas das cidades do arquivo .mat
coordenadas = carregar_coordenadas('cidades.mat')

# Executar o algoritmo genético
melhor_percurso, melhor_distancia = algoritmo_genetico(tamanho_populacao, num_geracoes, coordenadas)

print("Melhor percurso encontrado:", melhor_percurso)
print("Distância do melhor percurso:", melhor_distancia)