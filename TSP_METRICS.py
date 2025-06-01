import numpy as np
import random as rand
from TSP import TSP
from nsga import nsga1 
from spea import spea1_algorithm 
import csv
import math
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
# --- Funciones de Métricas ---
def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos (soluciones en el espacio de objetivos)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def generational_distance_m1(obtained_front, reference_front):
    # Calcula la métrica M1 (Generational Distance) para evaluar la distancia promedio de un frente obtenido al frente de referencia.
    if not obtained_front or not reference_front:
        if not obtained_front and not reference_front:
            return 0.0
        return float('inf')

    #obtiene las distancias mínimas de cada punto del frente obtenido al frente de referencia
    distances = []
    for p_obtained in obtained_front:
        min_dist = min(
            euclidean_distance(p_obtained, p_ref)
            for p_ref in reference_front
        )
        distances.append(min_dist)

    return sum(distances) / len(distances)

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




""" Función para calcular la distribución del frente (Spacing) con un parámetro sigma """
def distancia(reference_front, sigma=0.1):
    ref_np = np.array(reference_front)
    min_point = np.min(ref_np, axis=0)
    max_point = np.max(ref_np, axis=0)
    distancia_extremos = np.linalg.norm(max_point - min_point)
    return sigma * distancia_extremos


def calcular_m2_interno(frente, sigma=0.1):
    """
    Calcula la métrica M2 de distribución del frente de Pareto.
    
    frente: lista de puntos (lista de listas o array de forma [n_puntos, n_objetivos])
    sigma: umbral de distancia
    
    Retorna: valor de la métrica M2
    """
    F = np.array(frente)
    n = len(F)
    total = 0
    
    for i in range(n):
        p = F[i]
        # Distancias a todos los demás puntos del frente
        distancias = np.linalg.norm(F - p, axis=1)
        # Excluye la distancia a sí mismo (normalmente 0)
        distancias[i] = 0.0  
        # Cuenta cuántos puntos están a una distancia mayor que sigma
        conteo = np.sum(distancias > sigma)
        total += conteo

    m2 = total / (n - 1)
    return m2

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
            
  
    # spread_value /= len(extent_obj_obtained)
    return spread_value

def calculate_extent(front):
    if len(front) == 0:
        return 0.0

    front_np = np.array(front)
    min_point = np.min(front_np, axis=0)
    max_point = np.max(front_np, axis=0)

    # Distancia euclidiana entre extremos
    extent = np.linalg.norm(max_point - min_point)
    return extent

def dominates(obj1, obj2):
    """Verifica si obj1 domina a obj2 (para minimización)."""
    better_or_equal = all(f1 <= f2 for f1, f2 in zip(obj1, obj2))
    strictly_better = any(f1 < f2 for f1, f2 in zip(obj1, obj2))
    return better_or_equal and strictly_better

""" def check_for_dominated_solutions(front):
    
    if not front:
        return 0

    dominated_count = 0

    front_list = [tuple(obj) for obj in front]

    for i in range(len(front_list)):
        is_current_dominated = False
        for j in range(len(front_list)):
            if i != j and dominates(front_list[j], front_list[i]):
                is_current_dominated = True
                dominated_count += 1
                break # Si se encuentra una solución que domina a la actual, no es necesario seguir comparando
       
    return dominated_count """

def error_M4(obtained_front, reference_front, tolerance=1e-3):
    """
    Calcula el error M4 según la fórmula:
    M4(F) = 1 - |F ∩ F*| / |F|
    Donde F es el frente obtenido y F* es el frente optimo de referencia.
    Donde la intersección se define por cercanía (norma Euclidiana <= tolerancia).
    """
    intersection_count = 0
    for sol in obtained_front:
        for ref_sol in reference_front:
            if np.linalg.norm(np.array(sol) - np.array(ref_sol)) <= tolerance:
                intersection_count += 1
                break  # Evita contar duplicados
    if len(obtained_front) == 0:
        return 1.0  # Máximo error si no hay soluciones
    return 1 - (intersection_count / len(obtained_front))


# --- Funciones para ejecutar algoritmos y recopilar frentes ---

def solve_with_nsga(tsp_instance,name_instance, num_runs=5):
    all_nsga_fronts_per_run = [] # Para almacenar el frente de cada corrida
    all_nsga_solutions_combined = [] # Para el frente de referencia global

    with open(f'frentes_pareto_nsga.{name_instance}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corrida", "Objetivo1", "Objetivo2"]) 

        for problem_run in range(1, num_runs + 1): 
            print(f'\n--- Corriendo NSGA-I, Episodio: {problem_run} ---')
            Ytrue = nsga1(tsp_instance, 100, 200, 0.2, 0.3) 
            
            # Convertir a flotantes y asegurar que los puntos sean tuplas para consistencia
            Ytrue_float = [tuple(map(float, obj)) for obj in Ytrue]
            
            all_nsga_fronts_per_run.append(Ytrue_float)
            all_nsga_solutions_combined.extend(Ytrue_float) 

            for obj1, obj2 in Ytrue_float:
                writer.writerow([f"corrida_{problem_run}", obj1, obj2])

            print(f'Frente de Pareto de la corrida {problem_run} (NSGA-I): {Ytrue_float}')

    
        # --- Cálculo del frente promedio punto a punto ---
        min_len = min(len(frente) for frente in all_nsga_fronts_per_run)

        Ytrue_avg = []
        for i in range(min_len):
            avg_sol = np.mean([frente[i] for frente in all_nsga_fronts_per_run], axis=0)
            Ytrue_avg.append(tuple(avg_sol))

       
        for obj1, obj2 in Ytrue_avg:
            writer.writerow(["promedio", obj1, obj2])
            
    # Retorna tanto los frentes individuales por corrida como la combinación de todos
    return all_nsga_fronts_per_run, all_nsga_solutions_combined


def solve_with_spea(tsp_instance, name_instance,num_runs=5): 
    all_spea_fronts_per_run = [] # frente de cada corrida
    all_spea_solutions_combined = [] # frente de referencia global

    # Parameters for SPEA1
    pop_size = 100
    generations = 200
    crossover_rate = 0.8
    mutation_rate = 0.1
    archive_size = 100

    with open(f'frentes_pareto_spea.{name_instance}.csv', mode='w', newline='') as file:
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
            
        # --- Cálculo del frente promedio punto a punto ---
        frentes_ordenados = []
        for frente in all_spea_fronts_per_run:
            frente_ordenado = sorted(frente, key=lambda x: x[0])
            frentes_ordenados.append(frente_ordenado)

        min_len = min(len(frente) for frente in frentes_ordenados)

        Ytrue_avg = []
        if min_len > 0:
            for i in range(min_len):
                avg_sol = np.mean([frente[i] for frente in frentes_ordenados], axis=0)
                Ytrue_avg.append(tuple(avg_sol))

            for obj1, obj2 in Ytrue_avg:
                writer.writerow(["promedio_spea", obj1, obj2])

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
            

    return sorted(pareto_front, key=lambda x: x[0])


def find_best_front(fronts, reference_front, reference_front_min_obj, reference_front_max_obj):
    best_index = -1
    best_metrics = None  # Será una tupla: (M1, M2, -M3) para comparar fácilmente (todos minimización)
    
    for idx, run_front in enumerate(fronts):
        m1 = generational_distance_m1(run_front, reference_front)
        m2 = calcular_m2_interno(run_front)
        m3 = calculate_extent(run_front)
        error= error_M4(run_front, reference_front)
        metrics_tuple = (m1, m2, -m3,error)

        if best_metrics is None or metrics_tuple < best_metrics:
            best_metrics = metrics_tuple
            best_index = idx

    return best_index, fronts[best_index], best_metrics
instancia = 0
if __name__ == '__main__':
    
    if instancia == 0:
        tsp1 = TSP("tsp_KROAB100.TSP.TXT")
        tsp1.print_summary()
        name_instance = "KROAB100"
        
    else:
        tsp1 = TSP("tsp_kroac100.tsp.txt")
        tsp1.print_summary()
        name_instance = "kroac100"
    # --- Parámetros de la corrida ---
    num_runs_for_metrics = 5 # Define el número de corridas para calcular el promedio y las métricas

    # --- Ejecutar algoritmos y recopilar todos los frentes ---
    # `nsga_fronts_per_run` contiene una lista de frentes, uno por cada corrida
    # `all_nsga_solutions_combined` contiene todos los puntos de todas las corridas de NSGA
    nsga_fronts_per_run, all_nsga_solutions_combined = solve_with_nsga(tsp1,name_instance, num_runs=num_runs_for_metrics)
    
    
    spea_fronts_per_run, all_spea_solutions_combined = solve_with_spea(tsp1,name_instance, num_runs=num_runs_for_metrics)

    # --- Combinar todos los puntos de todos los algoritmos y obtener el frente de Pareto de referencia global ---
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

    """  print("\n--- Mejor frente NSGA-I ---")
    best_idx_nsga, best_front_nsga, best_metrics_nsga = find_best_front(
        nsga_fronts_per_run,
        reference_front,
        reference_front_min_obj,
        reference_front_max_obj
    )
    print(f"Índice del mejor frente: {best_idx_nsga}")
    print(f"M1 (GD): {best_metrics_nsga[0]:.4f}")
    print(f"M2 (Distribución σ): {best_metrics_nsga[1]:.4f}")
    print(f"M3 (Spread): {-best_metrics_nsga[2]:.4f}")   """

    for run_front in nsga_fronts_per_run:
        # M1: Distancia al frente óptimo
        gd_nsga_list.append(generational_distance_m1(run_front, reference_front))
        # M2: Distribución del frente (Spacing)
        spacing_nsga_list.append(calcular_m2_interno(run_front))
        # M3: Extensión del frente (Spread)
        spread_nsga_list.append(calculate_spread(run_front, reference_front_min_obj, reference_front_max_obj))
        # M3: Extensión del frente (Extent - rango puro)
        extent_nsga_list.append(calculate_extent(run_front))
        # Error: Elementos que no pertenecen al frente óptimo (dominados internamente)
        dominated_nsga_list.append(error_M4(run_front,reference_front))
    
    print(f"M1 (Distancia al frente optimo) NSGA-I (Promedio): {np.mean(gd_nsga_list):.4f} (Desv.Est: {np.std(gd_nsga_list):.4f})")
    print(f"M2 (Distribucion - Spacing) NSGA-I (Promedio): {np.mean(spacing_nsga_list):.4f} (Desv.Est: {np.std(spacing_nsga_list):.4f})")
    print(f"M3 (Extension - Spread) NSGA-I (Promedio): {np.mean(spread_nsga_list):.4f} (Desv.Est: {np.std(spread_nsga_list):.4f})")
    # Para Extent, podríamos promediar cada componente o solo mostrar el promedio de las listas
    print(f"M3 (Extension del frente - Rango) NSGA-I (Promedio por objetivo): {np.mean(extent_nsga_list, axis=0)}")
    print(f"Error (Elementos dominados en frente NSGA-I) (Promedio): {np.mean(dominated_nsga_list):.2f}, Desviación: {np.std(dominated_nsga_list):.2f}")


    # --- Calcular y promediar métricas para SPEA1 a través de las corridas ---
    print("\n--- Métricas PROMEDIO para SPEA1 ---")
    gd_spea_list = []
    spacing_spea_list = []
    spread_spea_list = []
    dominated_spea_list = []
    extent_spea_list = []
    
    """ 
    print("\n--- Mejor frente SPEA1 ---")
    best_idx_spea, best_front_spea, best_metrics_spea = find_best_front(
        spea_fronts_per_run,
        reference_front,
        reference_front_min_obj,
        reference_front_max_obj
    )
    print(f"Índice del mejor frente: {best_idx_spea}")
    print(f"M1 (GD): {best_metrics_spea[0]:.4f}")
    print(f"M2 (Distribución σ): {best_metrics_spea[1]:.4f}")
    print(f"M3 (Spread): {-best_metrics_spea[2]:.4f}") """


    for run_front in spea_fronts_per_run:
        # M1: Distancia al frente óptimo
        gd_spea_list.append(generational_distance_m1(run_front, reference_front))
        # M2: Distribución del frente (Spacing)
        spacing_spea_list.append(calcular_m2_interno(run_front))
        # M3: Extensión del frente (Spread)
        spread_spea_list.append(calculate_spread(run_front, reference_front_min_obj, reference_front_max_obj))
        # M3: Extensión del frente (Extent - rango puro)
        extent_spea_list.append(calculate_extent(run_front))
        # Error: Elementos que no pertenecen al frente óptimo (dominados internamente)
        dominated_spea_list.append(error_M4(run_front,reference_front))
    
    print(f"M1 (Distancia al frente optimo) SPEA1 (Promedio): {np.mean(gd_spea_list):.4f} (Desv.Est: {np.std(gd_spea_list):.4f})")
    print(f"M2 (Distribucion - Spacing) SPEA1 (Promedio): {np.mean(spacing_spea_list):.4f} (Desv.Est: {np.std(spacing_spea_list):.4f})")
    print(f"M3 (Extension - Spread) SPEA1 (Promedio): {np.mean(spread_spea_list):.4f} (Desv.Est: {np.std(spread_spea_list):.4f})")
    print(f"M3 (Extension del frente - Rango) SPEA1 (Promedio por objetivo): {np.mean(extent_spea_list, axis=0)}")
    print(f"Error (Elementos dominados en frente SPEA1) (Promedio): {np.mean(dominated_spea_list):.2f}, Desviación: {np.std(dominated_spea_list):.2f}")
    
    """ comparar los mejores frentes de ambos algoritmos """
    print("\n--- Comparación de los mejores frentes de NSGA-I y SPEA1 ---")
    best_idx_nsga, best_front_nsga, best_metrics_nsga = find_best_front(
        nsga_fronts_per_run,
        reference_front,
        reference_front_min_obj,
        reference_front_max_obj
    )
    best_idx_spea, best_front_spea, best_metrics_spea = find_best_front(
        spea_fronts_per_run,
        reference_front,
        reference_front_min_obj,
        reference_front_max_obj
    )
    print(f"Índice del mejor frente NSGA-I: {best_idx_nsga}")
    print(f"M1 (GD) NSGA-I: {best_metrics_nsga[0]:.4f}, M2 (Distribución σ): {best_metrics_nsga[1]:.4f}, M3 (Spread): {-best_metrics_nsga[2]:.4f}, Error: {best_metrics_nsga[3]:.2f}")
    print(f"Índice del mejor frente SPEA1: {best_idx_spea}")
    print(f"M1 (GD) SPEA1: {best_metrics_spea[0]:.4f}, M2 (Distribución σ): {best_metrics_spea[1]:.4f}, M3 (Spread): {-best_metrics_spea[2]:.4f}, Error: {best_metrics_spea[3]:.2f}")
    print("\n--- Frentes obtenidos ---")
    print(f"Mejor frente NSGA-I: {best_front_nsga}")
    print(f"Mejor frente SPEA1: {best_front_spea}")
    
    