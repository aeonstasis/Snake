__author__ = 'Aaron'

from constants import *
from NeuralNet import *
import Snake


# Default tuning parameters
POP_SIZE = 20
NUM_GENS = 100
CROSS_RATE = 0.70
MUTATE_RATE = 0.15


class Genome:  # TODO figure out whether to integrate ann here
    """
    Represents an individual in a population.
    In this case, the weights representing an individual neural net.
    """

    def __init__(self, weights=None, fitness=None):
        self.weights = weights
        self.fitness = fitness

    def copy(self):
        """
        Return deep copy of an individual.
        :return: a deep copy
        """
        copy = Genome(self.weights, self.fitness)
        return copy

    def get_fitness(self):
        """
        Returns the fitness of an individual genome. Calculates it once, then returns it when prompted again.
        :return: the genome's fitness
        """
        if self.fitness is not None:
            return self.fitness
        else:
            fitness = Snake.fitness(self.weights)
            return fitness

    def __str__(self):
        """
        Pretty print the weights of the neural network.
        """


class GA:
    """
    Encapsulates the methods needed to solve an optimization problem using a genetic
    algorithm.
    """

    def __init__(self, num_weights, pop_size=POP_SIZE, num_gens=NUM_GENS, mut_rate=MUTATE_RATE, cross_rate=CROSS_RATE):
        """
        Initialize the genetic algorithm to interface to the ANN.
        :param pop_size: number of genomes per generation
        :param mut_rate: probability of mutation
        :param num_gens: number of generations to run
        :param cross_rate: crossover rate
        :param num_weights: the total number of weights in our neural net
        """
        # Problem parameters
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.genome_length = num_weights
        self.genomes = None

        # Current population descriptors
        self.total_fitness = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.worst_fitness = 0
        self.best_genome = None

    @staticmethod
    def select(pop_subset, t):
        """
        Implements tournament-style selection of the population.
        :param pop_subset: some subset of the population
        :param t: size of the tournament
        :return: fittest individual from some subset of the population.
        """
        assert t >= 1, "Need at least two individuals"
        best = pop_subset[random.randrange(0, len(pop_subset))]
        for i in range(1, t):
            next_ind = pop_subset[random.randrange(0, len(pop_subset))]
            if next_ind.fitness > best.fitness:
                best = next_ind
        return best

    def mutate(self, genome):
        """
        Implements mutation, where weights may be changed.
        :param genome: genome in question
        """
        weights = genome.weights
        for i in range(len(weights)):
            if self.mut_rate >= random.random():
                weights[i] += (random.random() * 2) - 1  # add delta noise in [-1, 1]

    def crossover(self, g1, g2):
        """
        Implement uniform crossover, given two parent individuals.
        Return two children constructed from the weights.
        :type g1: Genome
        :type g2: Genome
        :param g1: first parent
        :param g2: second parent
        :return: tuple containing two children genomes
        """
        assert (len(g1.weights) == len(g2.weights))

        for i in range(len(g1.weights)):
            if self.cross_rate >= random.random():
                temp = g1.weights[i]
                g1.weights[i] = g2.weights[i]
                g2.weights[i] = temp
        return Genome(g1.weights), Genome(g2.weights)

    def epoch(self, old_population):
        """
        Runs the GA for one generation, updating the population.
        :return: the new population
        """
        self.total_fitness = 0  # Reset old total fitness

        # Identify current most fit individual
        for genome in old_population:
            self.total_fitness += genome.fitness
            if self.best_genome is None or genome.get_fitness() > self.best_genome.get_fitness():
                self.best_genome = genome

        # Record fitness value parameters
        self.avg_fitness = self.total_fitness / len(old_population)
        self.best_fitness = self.best_genome.get_fitness()

        # Generate next generation of individuals
        population_next = 0 * [None]
        for j in range(self.pop_size // 2):
            # Tournament select two parents
            p_a = self.select(old_population, 2)
            p_b = self.select(old_population, 2)

            # Create two children through uniform crossover
            (c_a, c_b) = self.crossover(p_a.copy(), p_b.copy())

            # Mutate children using a form of bitflip mutation
            self.mutate(c_a)
            self.mutate(c_b)

            # Add children to population pool for next gen
            population_next.append(c_a)
            population_next.append(c_b)
        return population_next


# Runs the GA for the ANN Snake problem
def main():
    ga = GA(1)  # TODO resolve calculation of num_weights

    # Initialize a random population of neural nets
    population = ga.pop_size * [None]
    for i in range(0, ga.pop_size):
        ind = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN)
        population[i] = ind

    # Run for num_gens generations
    for i in range(ga.num_gens):
        # Call epoch at each step
        population = ga.epoch(population)

    # Output structure of fittest individual
    print ga.best_genome

