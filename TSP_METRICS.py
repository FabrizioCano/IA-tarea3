import numpy as np
import random as rand
from nsga import nsga1 # Asegúrate de que nsga.py esté en el mismo directorio
from spea import spea1_algorithm # Asegúrate de que spea.py esté en el mismo directorio
import csv
import math

# --- Funciones de Métricas ---
def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos (soluciones en el espacio de objetivos)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def generational_distance(obtained_front, reference_front):
    """
    Calcula la Distancia Generacional (GD) o una métrica de distancia similar.
    Mide la cercanía del frente obtenido al frente de referencia.
    Un valor más bajo es mejor.
    """
    if not obtained_front or not reference_front:
        # Si uno de los frentes está vacío, la distancia es infinita o 0 si ambos están vacíos
        if not obtained_front and not reference_front:
            return 0.0
        return float('inf')

    distances = []
    for p_obtained in obtained_front:
        min_dist = float('inf')
        for p_ref in reference_front:
            dist = euclidean_distance(p_obtained, p_ref)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)

    if not distances: # Esto podría pasar si obtained_front estaba vacío, pero ya lo manejamos arriba.
        return 0.0

    # Fórmula original de GD: sqrt(sum(d^2))/N
    gd = math.sqrt(sum(d**2 for d in distances)) / len(distances)
    return gd

def spacing(front):
    """
    Calcula la métrica de 'Spacing' (Espaciado) para evaluar la distribución de un frente.
    Un valor más bajo indica una mejor distribución (más uniforme).
    Asume que el frente está ordenado por al menos un objetivo.
    """
    if len(front) < 2:
        return 0.0 # No hay distancias para calcular si hay menos de 2 puntos

    # Convertir a numpy array para facilitar las operaciones y asegurar tipos flotantes
    # Ordenar por el primer objetivo para calcular distancias entre vecinos
    front_np = np.array(sorted(front, key=lambda x: x[0]))

    distances = []
    for i in range(len(front_np) - 1):
        distances.append(euclidean_distance(front_np[i], front_np[i+1]))

    if not distances: # En caso de que el frente tenga 1 solo elemento
        return 0.0

    mean_dist = np.mean(distances)
    # Suma de los cuadrados de las diferencias entre cada distancia y la distancia media
    sum_diff_sq = sum([(d - mean_dist)**2 for d in distances])
    
    # Calcular la desviación estándar muestral
    if len(distances) > 1:
        s = math.sqrt(sum_diff_sq / (len(distances) - 1))
    else:
        s = 0.0 # Si solo hay una distancia, la desviación es 0

    return s

def calculate_spread(front, reference_front_min_obj, reference_front_max_obj):
    """
    Calcula una métrica de 'Spread' (Extensión) para evaluar el rango cubierto por el frente.
    Un valor más alto es generalmente mejor, indica que el frente cubre un rango más amplio.
    Se necesita el min y max de los objetivos del frente de referencia para la normalización.
    """
    if len(front) == 0:
        return 0.0

    front_np = np.array(front) # Convertir a numpy array para facilitar min/max
    
    # Extraer los valores mínimos y máximos para cada objetivo en el frente obtenido
    min_obj_obtained = np.min(front_np, axis=0)
    max_obj_obtained = np.max(front_np, axis=0)

    # Calcular la extensión en cada objetivo para el frente obtenido y el de referencia
    extent_obj_obtained = max_obj_obtained - min_obj_obtained
    extent_ref_front = reference_front_max_obj - reference_front_min_obj

    spread_value = 0.0
    for i in range(len(extent_obj_obtained)):
        if extent_ref_front[i] > 0:
            # Proporción de la extensión obtenida respecto a la de referencia
            spread_value += (extent_obj_obtained[i] / extent_ref_front[i])
        else:
            # Si el frente de referencia tiene extensión cero para un objetivo,
            # significa que todos sus puntos tienen el mismo valor para ese objetivo.
            # Aquí, si el frente obtenido tiene alguna extensión para ese objetivo,
            # sumamos 1.0 (o un valor fijo) para indicar que cubre algo.
            spread_value += (1.0 if extent_obj_obtained[i] > 0 else 0.0)
            
    # Podrías normalizar este valor sumado dividiendo por el número de objetivos
    # spread_value /= len(extent_obj_obtained) # Descomentar si quieres el promedio de proporciones
    return spread_value

def calculate_extent(front):
    """
    Calcula la extensión (rango) cubierto por el frente en cada objetivo.
    Retorna una lista donde cada elemento es el rango para un objetivo.
    """
    if len(front) == 0:
        return [0.0] * 2 # Asumiendo 2 objetivos (o ajusta a tsp_instance.n_objectives si está disponible)

    front_np = np.array(front)
    min_obj_values = np.min(front_np, axis=0)
    max_obj_values = np.max(front_np, axis=0)

    extent_per_objective = max_obj_values - min_obj_values
    return extent_per_objective.tolist()

def dominates(obj1, obj2):
    """Verifica si obj1 domina a obj2 (para minimización)."""
    better_or_equal = all(f1 <= f2 for f1, f2 in zip(obj1, obj2))
    strictly_better = any(f1 < f2 for f1, f2 in zip(obj1, obj2))
    return better_or_equal and strictly_better

def check_for_dominated_solutions(front):
    """
    Verifica cuántos elementos en el 'frente' dado son dominados por otros elementos
    dentro del MISMO frente. Esto indica la 'pureza' del frente devuelto.
    Un valor de 0 es ideal.
    """
    if not front:
        return 0

    dominated_count = 0
    # Convertir a lista de tuplas para asegurar inmutabilidad y comparación adecuada
    front_list = [tuple(obj) for obj in front]

    for i in range(len(front_list)):
        is_current_dominated = False
        for j in range(len(front_list)):
            if i != j and dominates(front_list[j], front_list[i]):
                is_current_dominated = True
                dominated_count += 1
                break # Una vez que es dominado, no necesitamos verificar con otros
        # No sumamos 1 si la solución es dominada por ella misma (i==j), lo cual ya se evita.
    return dominated_count

# --- Clase TSP (sin cambios) ---
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
        
        self.n_cities=int(lines[0])
        self.n_objectives=int(lines[1])
        
        line=2
        
        for i in range(self.n_objectives):
            mat=[]
            
            for j in range(self.n_cities):
                row=list(map(float,lines[line].split()))
                mat.append(row)
                line+=1
            self.adx_matrix.append(np.array(mat))
            
    def evaluar(self,tour):
        costos=[]
        for mat in self.adx_matrix:
            cost=0
            
            for j in range(len(tour)):
                cost+=mat[tour[j]] [tour[(j+1)%len(tour)]]
            costos.append(cost)
        return tuple(costos)

    def print_summary(self):
        print(f"Número de ciudades: {self.n_cities}")
        print(f"Número de objetivos: {self.n_objectives}")
        for i, matrix in enumerate(self.adx_matrix):
            print(f"Matriz de distancias para Objetivo {i+1} (forma={matrix.shape}) cargada.")

# --- Funciones para ejecutar algoritmos y recopilar frentes ---

def solve_with_nsga(tsp_instance, num_runs=5): # Agregado num_runs como parámetro
    all_nsga_fronts_per_run = [] # Para almacenar el frente de cada corrida
    all_nsga_solutions_combined = [] # Para el frente de referencia global

    with open("frentes_pareto_nsga.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corrida", "Objetivo1", "Objetivo2"]) 

        for problem_run in range(1, num_runs + 1): 
            print(f'\n--- Corriendo NSGA-I, Episodio: {problem_run} ---')
            Ytrue = nsga1(tsp_instance, 100, 200, 0.2, 0.3) # Asegúrate de que los parámetros sean consistentes
            
            # Convertir a flotantes y asegurar que los puntos sean tuplas para consistencia
            Ytrue_float = [tuple(map(float, obj)) for obj in Ytrue]
            
            all_nsga_fronts_per_run.append(Ytrue_float)
            all_nsga_solutions_combined.extend(Ytrue_float) 

            for obj1, obj2 in Ytrue_float:
                writer.writerow([f"corrida_{problem_run}", obj1, obj2])

            print(f'Frente de Pareto de la corrida {problem_run} (NSGA-I): {Ytrue_float}')

    # Retorna tanto los frentes individuales por corrida como la combinación de todos
    return all_nsga_fronts_per_run, all_nsga_solutions_combined


def solve_with_spea(tsp_instance, num_runs=5): # Agregado num_runs como parámetro
    all_spea_fronts_per_run = [] # Para almacenar el frente de cada corrida
    all_spea_solutions_combined = [] # Para el frente de referencia global

    # Parameters for SPEA1 (can be adjusted)
    pop_size = 100
    generations = 200
    crossover_rate = 0.8
    mutation_rate = 0.1
    archive_size = int(pop_size * 0.5)

    with open("frentes_pareto_spea.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corrida", "Objetivo1", "Objetivo2"])

        for problem_run in range(1, num_runs + 1):
            print(f'\n--- Corriendo SPEA1, Episodio: {problem_run} ---')
            Ytrue_spea = spea1_algorithm(tsp_instance, pop_size, archive_size, generations, crossover_rate, mutation_rate)
            
            # Convertir a flotantes y asegurar que los puntos sean tuplas para consistencia
            Ytrue_spea_float = [tuple(map(float, obj)) for obj in Ytrue_spea]

            all_spea_fronts_per_run.append(Ytrue_spea_float)
            all_spea_solutions_combined.extend(Ytrue_spea_float) 

            for obj1, obj2 in Ytrue_spea_float:
                writer.writerow([f"corrida_{problem_run}", obj1, obj2])

            print(f'Frente de Pareto de la corrida {problem_run} (SPEA1): {Ytrue_spea_float}')

    # Retorna tanto los frentes individuales por corrida como la combinación de todos
    return all_spea_fronts_per_run, all_spea_solutions_combined


# --- Función para obtener el frente de Pareto de referencia global ---
def get_reference_pareto_front(all_solutions):
    """
    Dada una lista de todas las soluciones (objetivos) de múltiples corridas/algoritmos,
    identifica el frente de Pareto de referencia (no dominado) global.
    """
    if not all_solutions:
        return []

    # Convertir a set para eliminar duplicados y luego a lista para iterar
    unique_solutions = list(set(all_solutions))
    
    pareto_front = []
    for i in range(len(unique_solutions)):
        is_dominated = False
        for j in range(len(unique_solutions)):
            if i != j and dominates(unique_solutions[j], unique_solutions[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(unique_solutions[i])
            
    # Opcional: ordenar el frente de referencia para mejor visualización
    return sorted(pareto_front, key=lambda x: x[0])


# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    tsp1 = TSP("tsp_KROAB100.TSP.TXT")
    tsp1.print_summary()

    num_runs_for_metrics = 5 # Define el número de corridas para calcular el promedio y las métricas

    # --- Ejecutar algoritmos y recopilar todos los frentes ---
    # `nsga_fronts_per_run` contiene una lista de frentes, uno por cada corrida
    # `all_nsga_solutions_combined` contiene todos los puntos de todas las corridas de NSGA
    nsga_fronts_per_run, all_nsga_solutions_combined = solve_with_nsga(tsp1, num_runs=num_runs_for_metrics)
    
    # Lo mismo para SPEA1
    spea_fronts_per_run, all_spea_solutions_combined = solve_with_spea(tsp1, num_runs=num_runs_for_metrics)

    # --- Combinar TODOS los puntos de TODOS los algoritmos y obtener el frente de Pareto de referencia global ---
    all_combined_solutions_for_reference = all_nsga_solutions_combined + all_spea_solutions_combined
    reference_front = get_reference_pareto_front(all_combined_solutions_for_reference)
    
    print("\n--- Frente de Pareto de Referencia GLOBAL (Aproximado) ---")
    print(reference_front)

    if reference_front:
        ref_front_np = np.array(reference_front)
        reference_front_min_obj = np.min(ref_front_np, axis=0)
        reference_front_max_obj = np.max(ref_front_np, axis=0)
    else:
        # En caso de que el frente de referencia esté vacío (poco probable si los algoritmos generan soluciones)
        reference_front_min_obj = np.array([0.0, 0.0]) # Valores por defecto para evitar errores
        reference_front_max_obj = np.array([1.0, 1.0]) # Valores por defecto
        print("Advertencia: No se pudo generar un frente de Pareto de referencia. Las métricas de distancia/extensión podrían ser inválidas.")


    # --- Calcular y promediar métricas para NSGA-I a través de las corridas ---
    print("\n--- Métricas PROMEDIO para NSGA-I ---")
    gd_nsga_list = []
    spacing_nsga_list = []
    spread_nsga_list = []
    dominated_nsga_list = []
    extent_nsga_list = []

    for run_front in nsga_fronts_per_run:
        # M1: Distancia al frente óptimo
        gd_nsga_list.append(generational_distance(run_front, reference_front))
        # M2: Distribución del frente (Spacing)
        spacing_nsga_list.append(spacing(run_front))
        # M3: Extensión del frente (Spread)
        spread_nsga_list.append(calculate_spread(run_front, reference_front_min_obj, reference_front_max_obj))
        # M3: Extensión del frente (Extent - rango puro)
        extent_nsga_list.append(calculate_extent(run_front))
        # Error: Elementos que no pertenecen al frente óptimo (dominados internamente)
        dominated_nsga_list.append(check_for_dominated_solutions(run_front))
    
    print(f"M1 (Distancia al frente optimo) NSGA-I (Promedio): {np.mean(gd_nsga_list):.4f} (Desv.Est: {np.std(gd_nsga_list):.4f})")
    print(f"M2 (Distribucion - Spacing) NSGA-I (Promedio): {np.mean(spacing_nsga_list):.4f} (Desv.Est: {np.std(spacing_nsga_list):.4f})")
    print(f"M3 (Extension - Spread) NSGA-I (Promedio): {np.mean(spread_nsga_list):.4f} (Desv.Est: {np.std(spread_nsga_list):.4f})")
    # Para Extent, podríamos promediar cada componente o solo mostrar el promedio de las listas
    print(f"M3 (Extension del frente - Rango) NSGA-I (Promedio por objetivo): {np.mean(extent_nsga_list, axis=0)}")
    print(f"Error (Elementos dominados en frente NSGA-I) (Promedio): {np.mean(dominated_nsga_list):.2f}")


    # --- Calcular y promediar métricas para SPEA1 a través de las corridas ---
    print("\n--- Métricas PROMEDIO para SPEA1 ---")
    gd_spea_list = []
    spacing_spea_list = []
    spread_spea_list = []
    dominated_spea_list = []
    extent_spea_list = []

    for run_front in spea_fronts_per_run:
        # M1: Distancia al frente óptimo
        gd_spea_list.append(generational_distance(run_front, reference_front))
        # M2: Distribución del frente (Spacing)
        spacing_spea_list.append(spacing(run_front))
        # M3: Extensión del frente (Spread)
        spread_spea_list.append(calculate_spread(run_front, reference_front_min_obj, reference_front_max_obj))
        # M3: Extensión del frente (Extent - rango puro)
        extent_spea_list.append(calculate_extent(run_front))
        # Error: Elementos que no pertenecen al frente óptimo (dominados internamente)
        dominated_spea_list.append(check_for_dominated_solutions(run_front))
    
    print(f"M1 (Distancia al frente optimo) SPEA1 (Promedio): {np.mean(gd_spea_list):.4f} (Desv.Est: {np.std(gd_spea_list):.4f})")
    print(f"M2 (Distribucion - Spacing) SPEA1 (Promedio): {np.mean(spacing_spea_list):.4f} (Desv.Est: {np.std(spacing_spea_list):.4f})")
    print(f"M3 (Extension - Spread) SPEA1 (Promedio): {np.mean(spread_spea_list):.4f} (Desv.Est: {np.std(spread_spea_list):.4f})")
    print(f"M3 (Extension del frente - Rango) SPEA1 (Promedio por objetivo): {np.mean(extent_spea_list, axis=0)}")
    print(f"Error (Elementos dominados en frente SPEA1) (Promedio): {np.mean(dominated_spea_list):.2f}")
