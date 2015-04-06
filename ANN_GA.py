__author__ = 'Aaron'

from constants import *
from NeuralNet import *
import Snake


# Default tuning parameters
POP_SIZE = 20
NUM_GENS = 50
CROSS_RATE = 0.70
MUTATE_RATE = 0.15

HEADLESS = 1
NUM_WEIGHTS = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN).num_weights


class Genome:
    """
    Represents an individual in a population.
    In this case, the weights representing an individual neural net.
    """

    def __init__(self, weights=None, fitness=None):
        # Set weights
        if weights is None:
            self.weights = [(random.uniform(0, 1) * 2) - 1 for _ in range(NUM_WEIGHTS)]
        else:
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
        if self.fitness is None:
            self.fitness = self.calc_fitness()
        return self.fitness

    def calc_fitness(self):
        """
        Calculates the fitness of an individual genome.
        """
        return Snake.fitness(self.weights, HEADLESS)

    def __str__(self):
        """
        Pretty print the weights of the neural network.

        output = ""
        for i in range(len(self.ann.layers)):
            output += "Layer " + i + ": "
            for neuron in self.ann.layers[i].neurons:
                weights += neuron.weights
        """
        return self.weights


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
        :return: a reference to a new mutated Genome
        """
        new_genome = genome.copy()
        weights = new_genome.weights
        for i in range(NUM_WEIGHTS):
            if self.mut_rate >= random.random():
                weights[i] += (random.random() * 2) - 1  # add delta noise in [-1, 1]
        new_genome.fitness = new_genome.calc_fitness()
        return new_genome

    def crossover(self, g1, g2):
        """
        Implement uniform crossover, given two parent individuals.
        Return two children constructed from the weights.
        :param g1: first parent
        :param g2: second parent
        :return: tuple containing two children sets of weights
        """
        c1 = g1.copy()
        c2 = g2.copy()

        assert (len(c1.weights) == len(c2.weights))

        for i in range(len(c1.weights)):
            if self.cross_rate >= random.random():
                temp = c1.weights[i]
                c1.weights[i] = c2.weights[i]
                c2.weights[i] = temp

        c1.fitness = c1.calc_fitness()
        c2.fitness = c2.calc_fitness()

        return c1, c2

    def epoch(self, old_population):
        """
        Runs the GA for one generation, updating the population.
        :return: the new population
        """
        self.total_fitness = 0  # Reset old total fitness

        # Identify current most fit individual and perform fitness calculations
        for genome in old_population:
            self.total_fitness += genome.get_fitness()
            if self.best_genome is None or genome.get_fitness() > self.best_genome.get_fitness():
                self.best_genome = genome.copy()

        # Record fitness value parameters
        self.avg_fitness = self.total_fitness / len(old_population)
        self.best_fitness = self.best_genome.get_fitness()

        # Generate next generation of individuals
        population_next = 0 * [None]
        for i in range(self.pop_size // 2):
            # Tournament select two parents
            p_a = self.select(old_population, 2)
            p_b = self.select(old_population, 2)

            # Create two children through uniform crossover
            (c_a, c_b) = self.crossover(p_a, p_b)

            # Add children to population pool for next gen
            population_next.append(c_a)
            population_next.append(c_b)

        # Override two weights with the current best (elitism)
        """
        old_population[0].ann.set_weights(self.best_genome.ann.get_weights())
        old_population[1].ann.set_weights(self.best_genome.ann.get_weights())
        """

        return old_population


# Runs the GA for the ANN Snake problem
def main():
    # Calculate number of weights
    ga = GA(NUM_WEIGHTS)

    # Initialize a random population of neural nets
    population = ga.pop_size * [None]
    for i in range(0, ga.pop_size):
        population[i] = Genome()

    # Store best genome for each generation
    best_genomes = []

    # Run for num_gens generations
    for i in range(ga.num_gens):
        # Call epoch at each step
        population = ga.epoch(population)
        best_genomes.append(ga.best_genome.weights)

        # Print population characteristics
        print "Gen " + str(i) + ": " + "best - " + str(ga.best_fitness) + \
              ", avg - " + str(ga.avg_fitness)

    # Output structure of fittest individual
    print best_genomes
    print ga.best_genome.__str__

# Run the GA
if __name__ == '__main__':
    main()

