import random as rand
import math

""" funcion que genera la poblacion incial aleatoriamente
N: tamaño de la poblacion
n_cities: cantidad de ciudades
"""


def generate_population(n_cities, N):
    population = []
    initial_tour = list(range(n_cities))

    for i in range(N):
        t = initial_tour[:]
        rand.shuffle(t)
        population.append(t)
    return population


""" 2. Evaluacion de fitness """


def fitness_evaluation(problem, population):
    return [problem.evaluar(tour) for tour in population]


""" 3. Seleccion mediante torneo binario """


def binary_tournament(population, fitness, k=2):
    selected = []
    for i in range(len(population)):
        cand = rand.sample(list(zip(population, fitness)), k)
        """ aqui se debe usar pareto """
        winner = min(cand, key=lambda x: (x[1]))
        selected.append(winner[0])
    return selected


""" 4. Operadores geneticos
"""

""" crossover ordenado: Para el cruce basado en orden, primero se selecciona un subconjunto de ciudades del primer progenitor. En los descendientes, estas ciudades aparecen en el mismo orden que en el primer progenitor, pero en las posiciones tomadas del segundo progenitor. Luego, las posiciones restantes se completan con las ciudades del segundo progenitor.
"""


def crossover_ordenado(parent1, parent2):
    size = len(parent1)

    """ Se elige una primera parte del primer padre:parent1, desde la posicion "a" a la "b"  y se copia al hijo en las mismas posiciones"""
    a, b = sorted(rand.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    """ Luego se construye un cromosoma con los elementos del otro padre:parent2 que no esten en el hijo:child, evitando asi repetidos (ciudades repetidas) """
    fill = [x for x in parent2 if x not in child]

    """ se procede a recorrer el cromosoma hijo y se llena cada posicion vacia con un elemento ciudad de fill en orden, asi retornando un camino o permutacion valido """
    c = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[c]
            c += 1
    return child


""" operador mutacion que recibe un tour o camino como parametro y una tasa de mutacion, si la tasa generada aleatoriamente es menor a la tasa de mutacion, se procede a mutar ciudades aleatorias del camimo "tour" """


def mutation(tour, mutation_rate):
    tour = tour[:]

    if rand.random() < mutation_rate:
        i, j = rand.sample(range((len(tour))), 2)
        tour[i], tour[j] = tour[j],tour[i]

    return tour


""" funcion que observa si un camino solucion domina a otro
all: si un camino es igual o mejor que otro en todos los objetivos
any: si un camino es estrictamente mejor que otro en al menos un objetivo
"""


def dominates_solution(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


""" aplicar ordenamiento no dominado 
Recibe un conjunto de soluciones evaluadas en varios objeticos y organiza las mismas en niveles por frente segun la dominancia de Pareto
Retorna una lista de listas donde cada lista contiene los indices de las soluciones de un frente en particular
"""


def nondominated_sorting(objectives):
    """inicializar la lista de soluciones que representara una lista de soluciones dominadas por una solucion i en particular:
    Sea la solucion 0: Q[0]=[1,2], la solucion 0 domina a la solucion 1 y 2
    """
    Q = [[] for i in range(len(objectives))]
    """ contador de soluciones dominadas por una solucion i en particular """
    count_domination = [0] * len(objectives)
    """ ranking de frentes a clasificar """
    ranking = [0] * len(objectives)
    """ lista de frentes rankeados, es un mapa directo por índice, para cualquier solución se puede conocer su nivel """
    fronts = [[]]

    """ ciclo que evalua todas las soluciones entre si """
    for a in range(len(objectives)):
        for b in range(len(objectives)):
            """si la solucion a domina a la solucion b, se agrega su indice a la lista de soluciones dominadas de a"""
            if dominates_solution(objectives[a], objectives[b]):
                Q[a].append(b)
                """ en cambio si b domina a la solucion a , se incrementa el contador de dominacion de a """
            elif dominates_solution(objectives[b], objectives[a]):
                count_domination[a] += 1
        """ si a la solucion a no le domina ninguna otra, significa que es parte del primer frente pareto (nivel 0) y se lo agrega al primer frente """
        if count_domination[a] == 0:
            ranking[a] = 0
            fronts[0].append(a)

    """ aqui se realiza la clasificacion de frentes a partiendo del frente 0 """
    i = 0
    """ mientras existan soluciones en el frente actual """
    while fronts[i]:
        """lista de soluciones que van a pertenecer al siguiente prente"""
        next_front = []
        """ por cada solucion p en el frente actual se recorre cada solucion q que es dominada por p Q[p] """
        for p in fronts[i]:
            for q in Q[p]:
                """se le resta una dominacion a q debido a que ya se proceso p (q ya no esta dominada por p)"""
                count_domination[q] -= 1
                """ si q ya no es dominada por ninguna otra solucion se lo asigna al siguiente frente y se lo asigna al siguiente nivel de frente """
                if count_domination[q] == 0:
                    ranking[q] = i + 1
                    next_front.append(q)
        """ se aumenta de frente y se agrega el frente obtenido a la lista de frentes """
        i += 1
        fronts.append(next_front)
    """ elimina el ultimo frente (lista vacia) """
    fronts.pop()
    return fronts


""" calcula la distancia euclidiana ente dos puntos en el frente de manera a conocer la distancia para aplicar fitness sharing por nicho """


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


""" funcion que comprueba si una solucion esta en el nicho de la otra, si la distancia dist es menor a sigma, comparten nicho y se comparte el fitness de acuerdo a la funcion 1-(distancia/sigma)^2 """


def sharing_function(dist, sigma_share):
    if dist < sigma_share:
        return 1 - (dist / sigma_share) ** 2
    else:
        return 0


""" hace fitness sharing por nichos """


def fitness_sharing(objectives, sigma_share):
    """lista de fitness sharing donde cada posicion correspondera a una solucion"""
    shared_fitness = [0.0] * len(objectives)
    """ por cada solucion i se calcula su distancia a todas las demas soluciones y se evalua cuanto """
    for i in range(len(objectives)):
        """contador de nicho que acumulara cuanto comparte su nicho la solucion i con las demas soluciones"""
        niche_count = 0.0
        for j in range(len(objectives)):
            """calcula la distancia euclidiana entre la solucion i y el punto j"""
            dist = euclidean_distance(objectives[i], objectives[j])
            """ incrementa en ese valor el cointador de nicho """
            niche_count += sharing_function(dist, sigma_share)
        """ calcula el fitness compartido de la solucion i y le asigna el inverso del valor de su nicho a la solucion y si no significa que no posee ningun vecino cerca y se le asigna un fitness alto para darle prioridad """
        shared_fitness[i] = 1 / niche_count if niche_count != 0 else float("inf")
    return shared_fitness


""" funcion de seleccion de N soluciones para la siguiente generacion, se priorizaa aquellas no dominadas y se aplica fitess sharing para mantener la diversidad """


def select_parents_nsga1(population, objectives, N, sigma_share=0.2):
    """hacer la coleccion de soluciones no dominadas y ordenarlas mediante rankings por frentes pareto"""
    fronts = nondominated_sorting(objectives)
    """ listas donde se guardaran los padres seleccionados """
    new_population = []
    new_objectives = []

    """ recorre cada frente pareto """
    for front in fronts:
        """si al agregar un frente se supera el tamaño de la poblacion se debe seleccionar una parte del frente"""
        if len(new_population) + len(front) > N:
            """aqui se extrae los valores de los objetivos para cada solucion en el frente"""
            front_objs = [objectives[i] for i in front]
            """ se aplica fitness sharing para compartir el fitness con el frente """
            fitnesses = fitness_sharing(front_objs, sigma_share)
            """ ordena las soluciones del frente segun su fitness, de mejor a peor, para darle diversidad a la poblacion """
            sorted_front = sorted(zip(front, fitnesses), key=lambda x: -x[1])

            """ por cada solucion del frente va agregando a la nueva poblacion las soluciones hasta llegar a N """
            for idx, _ in sorted_front:
                if len(new_population) < N:
                    new_population.append(population[idx])
                    new_objectives.append(objectives[idx])
                else:
                    break
        else:
            for idx in front:
                new_population.append(population[idx])
                new_objectives.append(objectives[idx])

    """ se retorna la nueva poblacion y sus valores de objetivos correspondientes """
    return new_population, new_objectives


""" problem:instancia del problema tsp
N: tamaño de la poblacion
generations: numero de generaciones
mutation_rate: tasa de mutacion
"""


def nsga1(problem, N=100, generations=250, mutation_rate=0.2, sigma=0.3):
    """Inicializar población aleatoria y evaluarla en cada uno de los objetivos"""
    population = generate_population(problem.n_cities, N)
    objective_values = [problem.evaluar(ind) for ind in population]

    """ ciclo de evolucion por generaciones """
    for gen in range(generations):
        print(f"Generacion: {gen+1}")
        """ lista de la siguiente poblacion """

        """ calcualr el fitness para el toreno binario """
        fitnesses = fitness_sharing(objective_values, sigma)
        """ se seleccionan los padres para el crossover """
        mating_pool = binary_tournament(population, fitnesses, k=2)
        """ aplicar operadores geneticos """
        descendents = []
        while len(descendents) < N:
            p1, p2 = rand.sample(mating_pool, 2)
            child = crossover_ordenado(p1, p2)
            child = mutation(child, mutation_rate)
            descendents.append(child)

        """ se evalua la poblacion hija o descendente """
        descendents_objectives = [problem.evaluar(ind) for ind in descendents]

        """ se aplica elitismo """
        combined_population = population + descendents
        combined_objectives = objective_values + descendents_objectives

        """ se selecciona la proxima generacion aplicando NSGA con fitness sharing"""
        population, objective_values = select_parents_nsga1(
            combined_population, combined_objectives, N, sigma
        )

        print(f"Generación {gen+1} completada.")

    """ 
        Se extrae el primer frente no dominado (frente de Pareto), 
        que contiene las mejores soluciones en términos de dominancia. 
        Estas soluciones no son dominadas por ninguna otra en la población, 
        por lo tanto, representan el compromiso óptimo entre los objetivos (óptimos de Pareto).
    """
    pareto_front_indices = nondominated_sorting(objective_values)[0]
    pareto_front = [objective_values[i] for i in pareto_front_indices]

    return pareto_front
