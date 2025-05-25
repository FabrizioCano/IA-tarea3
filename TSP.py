import numpy as np
import random as rand
from nsga import nsga1
import csv
class TSP:
    def __init__(self,filename):
        self.n_cities=0
        self.n_objectives=0
        self.adx_matrix=[]
        self.filename=filename
        self._load_file()
        
    def _load_file(self):
        with open(self.filename,'r') as file:
            lines=[line.strip() for line in file.readlines() if line.strip() != '']
        
        """ establecer el numero de ciudades y objetivos """
        self.n_cities=int(lines[0])
        self.n_objectives=int(lines[1])
        
        """ leer desde la linea 2 """
        line=2
        
        """ leer por cada objetivo una matriz, la matriz de adjayencia sera una matriz de matrices donde cada una representa a un objetivo """
        for i in range(self.n_objectives):
            mat=[]
            
            for j in range(self.n_cities):
                row=list(map(float,lines[line].split()))
                mat.append(row)
                line+=1
            self.adx_matrix.append(np.array(mat))
            
    def evaluar(self,tour):
        """ evalua un camino y retorna una tupla con los costos asociados a este camino """
        costos=[]
        for mat in self.adx_matrix:
            cost=0
            
            for j in range(len(tour)):
                """ nos aseguramos de formar un ciclo hamiltoniano con el camnio, empieza en la primera ciudad y termina en esa misma """
                cost+=mat[tour[j]] [tour[(j+1)%len(tour)]]
            costos.append(cost)
        return tuple(costos)

    def print_summary(self):
        print(f"Number of cities: {self.n_cities}")
        print(f"Number of objectives: {self.n_objectives}")
        for i, matrix in enumerate(self.adx_matrix):
            print(f"Objective {i+1} distance matrix (shape={matrix.shape}) loaded.")




def solve_with_nsga():
    avg_fronts = []

    with open("frentes_pareto.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corrida", "Objetivo1", "Objetivo2"]) 

        for problem in range(1,5): 
            print(f'Episodio: {problem}')
            Ytrue = nsga1(tsp1, 100, 200, 0.2, 0.3)
            avg_fronts.append(Ytrue)

            
            for obj1, obj2 in Ytrue:
                writer.writerow([f"corrida_{problem}", float(obj1), float(obj2)])

           
            Ytrue_float = [(float(a), float(b)) for (a, b) in Ytrue]
            print(f'Ytrue de la corrida {problem}: {Ytrue_float}')

        """ Ordenar todos los frentes por el primer objetivo"""
        frentes_ordenados = []
        for frente in avg_fronts:
            frente_ordenado = sorted(frente, key=lambda x: x[0])
            frentes_ordenados.append(frente_ordenado)

        """ Promediar punto a punto """
        
        min_len = min(len(frente) for frente in frentes_ordenados)

        Ytrue_avg = []
        for i in range(min_len):
            avg_sol = np.mean([frente[i] for frente in frentes_ordenados], axis=0)
            Ytrue_avg.append(tuple(avg_sol))

        
        for obj1, obj2 in Ytrue_avg:
            writer.writerow(["promedio", float(obj1), float(obj2)])

    return Ytrue_avg

        
tsp1=TSP("tsp_KROAB100.TSP.TXT")
tsp1.print_summary()

ytrue=solve_with_nsga()
print(f'Ytrue resultante de las 5 corridas: {ytrue}')
""" 

for i in range(100):
    tour = list(range(tsp1.n_cities))
    rand.shuffle(tour)
    cost = tsp1.evaluar(tour)
    print(f'Tour {i+1} cost (ambos objetivos): ({cost[0]:.2f}, {cost[1]:.2f})') """