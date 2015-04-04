__author__ = 'Aaron'

import random

from math import *


# Default tuning parameters
POP_SIZE = 20
NUM_GENS = 100
CROSS_RATE = 0.70
MUTATE_RATE = 0.15


class Genome:
    """
    Represents an individual in a population.
    In this case, the weights representing an individual neural net.
    """

    def __init__(self, weights=None):
        self.weights = weights
        self.fitness = None

    def copy(self):
        """
        Return deep copy of an individual.
        :return: a deep copy
        """
        copy = Genome(self.weights, self.fitness)
        return copy

    def fitness(self):
        """
        Returns the fitness of an individual genome. Calculates it once, then returns it when prompted again.
        :return: the genome's fitness
        """
        if self.fitness is not None:
            return self.fitness
        else:
            fitness = Snake.fitness(self.weights)  # TODO fix the Snake class import
            return fitness


class GA:
    """
    Encapsulates the methods needed to solve an optimization problem using a genetic
    algorithm.
    """

    def __init__(self, num_weights, pop_size=POP_SIZE, mut_rate=MUTATE_RATE, cross_rate=CROSS_RATE):
        """
        Initialize the genetic algorithm to interface to the ANN.
        :param pop_size: number of genomes per generation
        :param mut_rate: probability of mutation
        :param cross_rate: crossover rate
        :param num_weights: the total number of weights in our neural net
        """
        # Problem parameters
        self.pop_size = pop_size
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
                bit = 1 if weights[i] == 0 else 1
                weights = weights[0:i] + str(bit) + weights[i + 1:]

    def crossover(self, e1, e2):
        """
        Implement uniform crossover, given two parent individuals.
        Return two children constructed from the weightss.
        :type e1: Genome
        :type e2: Genome
        :param e1: first parent
        :param e2: second parent
        :return: tuple containing two weights bitstrings
        """
        assert (isinstance(e1, Genome) and isinstance(e2, Genome))
        assert (len(e1.weights) == len(e2.weights))

        for i in range(len(e1.weights)):
            if self.cross_rate >= random.random():
                temp = e1.weights[i]
                e1.set_weights(e1.weights[0:i] + e2.weights[i] + e1.weights[i + 1:])
                e2.set_weights(e2.weights[0:i] + temp + e2.weights[i + 1:])
        return Genome(e1.weights), Genome(e2.weights)

    def optimize(self):
        """
        Implements a genetic algorithm in order to solve an
        optimization problem.
        :return: the solution to the optimization problem
        """
        population = self.n * [None]  # Initialize population of random individuals
        for i in range(0, self.n):
            bit_string = random.randint(0, pow(2, numBits))  # Bitstring treated as unsigned integer
            format_string = '0' + str(numBits) + 'b'
            bit_string = format(bit_string, format_string)
            ind = Genome(bit_string)
            population[i] = ind

        best = None

        # Run for num_gens generations
        for i in range(NUM_GENS):  # Control number of generations
            total_fitness = 0
            for elem in population:  # Identify current most fit individual
                elem_fitness = elem.fitness
                total_fitness += elem_fitness
                if self.problem == "max":
                    if best is None or elem_fitness > best.fitness:
                        best = elem
                else:
                    if best is None or elem_fitness < best.fitness:
                        best = elem

            avg_fitness = total_fitness / len(population)  # Record fitness value of population for run
            best_fitness = best.fitness
            self.update_data(avg_fitness, best_fitness, i)

            population_next = 0 * [None]
            for j in range(self.n // 2):  # Generate next generation of individuals
                p_a = GA.select(self, population, 2)  # Tournament select two parents
                p_b = GA.select(self, population, 2)
                (c_a, c_b) = GA.crossover(p_a.copy(), p_b.copy())  # Create two children through uniform crossover
                GA.mutate(c_a)  # Mutate both children using uniform bit-flip
                GA.mutate(c_b)
                population_next.append(c_a)  # Add children to population pool for next gen
                population_next.append(c_b)
            population = population_next  # Advance one generation
        assert (isinstance(best, Genome))

        GA.cnt += 1  # Advance run count

        return best.decode(best.weights)  # Return decoded value of best individual


# Runs the GA for the ANN Snake problem
def main():

# Initialize a random population of weights

#

