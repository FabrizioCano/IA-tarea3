import random
import math
from collections import namedtuple

Individual = namedtuple("Individual", ["permutation", "objectives", "strength", "raw_fitness"])

def dominates(obj1, obj2):
    """Verifica si obj1 domina a obj2 (para minimización)."""
    better_or_equal = all(f1 <= f2 for f1, f2 in zip(obj1, obj2))
    strictly_better = any(f1 < f2 for f1, f2 in zip(obj1, obj2))
    return better_or_equal and strictly_better

def calculate_strength(population):
    # Calcula la fuerza de cada individuo (cuántos individuos domina)
    for i in range(len(population)):
        strength = 0
        for j in range(len(population)):
            if i != j and dominates(population[i].objectives, population[j].objectives):
                strength += 1
        population[i] = population[i]._replace(strength=strength)
    return population

def calculate_raw_fitness(population, archive):
    # Calcula el fitness bruto como la suma de las fuerzas de los que lo dominan
    combined = population + archive
    raw_fitness_values = []

    for i in range(len(combined)):
        raw_fitness = 0
        for j in range(len(combined)):
            if i != j and dominates(combined[j].objectives, combined[i].objectives):
                raw_fitness += combined[j].strength
        raw_fitness_values.append(raw_fitness)

    for i in range(len(population)):
        population[i] = population[i]._replace(raw_fitness=raw_fitness_values[i])

    for i in range(len(archive)):
        archive[i] = archive[i]._replace(raw_fitness=raw_fitness_values[len(population) + i])

    return population, archive

def environmental_selection_spea1(population, archive, archive_size):
    # Selección ambiental: selecciona individuos para el nuevo archivo
    combined = population + archive
    next_archive = [ind for ind in combined if ind.strength == 0]

    if len(next_archive) > archive_size:
        # Si hay demasiados, recorta aleatoriamente
        random.shuffle(next_archive)
        next_archive = next_archive[:archive_size]
    elif len(next_archive) < archive_size:
        # Si hay pocos, rellena con individuos dominados ordenados por fitness bruto
        dominated = sorted([ind for ind in combined if ind.strength > 0], key=lambda ind: ind.raw_fitness)
        next_archive.extend(dominated[:archive_size - len(next_archive)])
        random.shuffle(next_archive)
        next_archive = next_archive[:archive_size]

    return next_archive

def selection(population, num_parents):
    # Selección por torneo binario con base en el fitness bruto (menor es mejor)
    parents = []
    for _ in range(num_parents):
        p1, p2 = random.sample(population, 2)
        if p1.raw_fitness <= p2.raw_fitness:
            parents.append(p1)
        else:
            parents.append(p2)
    return parents

def crossover(parent1, parent2, crossover_rate):
    # Cruce PMX para TSP
    if random.random() < crossover_rate:
        permutation1 = list(parent1.permutation)
        permutation2 = list(parent2.permutation)
        n = len(permutation1)
        p1, p2 = sorted(random.sample(range(n), 2))
        offspring1 = [None] * n
        offspring2 = [None] * n

        for i in range(p1, p2 + 1):
            offspring1[i] = permutation1[i]
            offspring2[i] = permutation2[i]

        for i in range(n):
            if offspring1[i] is None:
                gene = permutation2[i]
                while gene in offspring1:
                    gene = permutation1[permutation2.index(gene)]
                offspring1[i] = gene

            if offspring2[i] is None:
                gene = permutation1[i]
                while gene in offspring2:
                    gene = permutation2[permutation1.index(gene)]
                offspring2[i] = gene

        return (
            Individual(tuple(offspring1), None, None, None),
            Individual(tuple(offspring2), None, None, None)
        )
    else:
        # Si no hay cruce, devuelve los padres originales
        return parent1, parent2

def mutate(individual, mutation_rate):
    # Mutación por intercambio (swap) para TSP
    if random.random() < mutation_rate:
        permutation = list(individual.permutation)
        i, j = random.sample(range(len(permutation)), 2)
        permutation[i], permutation[j] = permutation[j], permutation[i]
        return Individual(tuple(permutation), None, None, None)
    else:
        return individual

def spea1_algorithm(tsp_instance, population_size, archive_size, generations, crossover_rate, mutation_rate):
    # Algoritmo principal SPEA1
    num_cities = tsp_instance.n_cities
    population = []

    # Inicializa la población con permutaciones aleatorias
    for _ in range(population_size):
        permutation = tuple(random.sample(range(num_cities), num_cities))
        objectives = tsp_instance.evaluar(permutation)
        population.append(Individual(permutation, objectives, None, None))

    archive = []

    for generation in range(generations):
        # Calcula fuerza y fitness bruto
        population = calculate_strength(population)
        archive = calculate_strength(archive)
        population, archive = calculate_raw_fitness(population, archive)

        # Selección ambiental para actualizar el archivo
        archive = environmental_selection_spea1(population, archive, archive_size)

        if generation == generations - 1:
            break

        # Selección y reproducción
        mating_pool = selection(population + archive, population_size)
        next_population = []

        for i in range(0, population_size, 2):
            parent1 = mating_pool[i % len(mating_pool)]
            parent2 = mating_pool[(i + 1) % len(mating_pool)]

            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)

            obj1 = tsp_instance.evaluar(offspring1.permutation)
            obj2 = tsp_instance.evaluar(offspring2.permutation)

            next_population.append(Individual(offspring1.permutation, obj1, None, None))
            next_population.append(Individual(offspring2.permutation, obj2, None, None))

        population = next_population

    # Devuelve los objetivos de los individuos no dominados del archivo final
    pareto_individuals = [ind for ind in archive if ind.strength == 0]
    pareto_objectives = [ind.objectives for ind in pareto_individuals]

    return pareto_objectives
