
import random
import numpy
from deap import creator, tools, base, algorithms

NBR_ITEMS = 20
IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50


# Create a dictionary of items that can be stored in the knapsack
possible_items = {}
for i in range(NBR_ITEMS):
    possible_items[i] = (random.randint(1,10), random.uniform(0,100)) # weight, value


def buildToolbox():
    # Weights represent weight and value, the goal is to minimize
    # weight and maximize value
    creator.create(name="Fitness", base=base.Fitness, weights=(-1.0, 1.0))

    # Individual solutions contains a set of items
    creator.create(name="Individual", base=set, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", random.randrange, NBR_ITEMS)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def buildStats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register(name="avg", function=numpy.mean, axis=0)
    stats.register(name="std", function=numpy.std, axis=0)
    stats.register(name="min", function=numpy.min, axis=0)
    stats.register(name="max", function=numpy.max, axis=0)
    return stats

def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += possible_items[item][0]
        value += possible_items[item][1]

    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 10000, 0 # Bad solution, overweight or too many items
    
    return weight, value

def cxSet(ind1, ind2):
    temp = set(ind1)
    ind1 &= ind2 # Intersection of two sets
    ind2 ^= temp # Symmetric difference
    return ind1, ind2

def mutSet(individual):
    if random.random() < 0.5:
        if len(individual) > 0: # Cannot remove from empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))

    return individual,

def main():
    NGEN = 100 # Number of generations
    MU = 50 # Population size
    LAMBDA = 100 # Children to product
    CXPB = 0.7 # Crossover probability
    MUTPB = 0.2 # Mutation probability

    # Create the toolbox, stats collector, and hall-of-fame recorder
    toolbox = buildToolbox()
    stats = buildStats()
    hof = tools.ParetoFront()

    # Create the population
    pop = toolbox.population(n=MU)
    
    # Run the algorithm
    algorithms.eaMuPlusLambda(population=pop, 
                              toolbox=toolbox, 
                              mu=MU, 
                              lambda_=LAMBDA, 
                              cxpb=CXPB, 
                              mutpb=MUTPB, 
                              ngen=NGEN, 
                              stats=stats, 
                              halloffame=hof)
    
    return pop, stats, hof

if __name__ == '__main__':
    pop, stats, hof = main()
    print(pop)
    print(stats)
    print(hof)