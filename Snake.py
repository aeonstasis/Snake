__author__ = 'Aaron'

# IMPORTS
import sys

from States import *
from constants import *
from NeuralNet import *


# GLOBAL VARIABLES
clock = pygame.time.Clock()
ann = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN)


# STATE MANAGER
class StateManager(object):
    def __init__(self, ann=None):
        """
        Initializes the state manager.
        Contains "global" variables to hold neural network and score.
        """
        self.ann = ann
        self.fitness = 0

        self.state = None
        self.go_to(MenuState())

    def go_to(self, state):
        self.state = state
        self.state.manager = self


# GAME ENGINE
def main():
    pygame.init()
    pygame.display.set_caption("Snake")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((0, 0, 0))

    running = True
    manager = StateManager()

    while running:
        clock.tick(FRAMES_PER_SEC)

        if pygame.event.get(QUIT):
            sys.exit(0)

        manager.state.handle_events(pygame.event.get())
        manager.state.update()
        manager.state.render(screen)
        pygame.display.flip()


# FITNESS FUNCTION
def fitness(weights, headless=1):
    """
    Calculate the fitness function.
    :param weights: weights representing the ANN.
    :return: score of the ANN represented
    """
    ann.set_weights(weights)

    pygame.init()
    pygame.display.set_caption("Snake")

    screen = None

    if headless == 0:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.fill((0, 0, 0))

    manager = StateManager(ann)
    manager.go_to(PlayState())

    while not isinstance(manager.state, GameOverState):
        if headless == 0:
            clock.tick(FRAMES_PER_SEC)

        manager.state.update()

        if headless == 0:
            manager.state.render(screen)
            pygame.display.flip()

    return manager.fitness


# PROGRAM EXECUTION
if __name__ == '__main__':
    # main()

    # Test ANN produced from GA
    best_weights = (
    [-12.040412221822939, -7.7276866105106174, -4.178103442726051, 38.04481707409031, -3.084177518587347,
     -3.587111310576874, 4.830187603871782, -10.13398314456584, 6.137672889803868, 12.605095802020552,
     -3.707881512272568, -8.276612273245078, 23.767903015552406, 5.777817774669535, 10.53141269039588,
     -11.381030605881719, -2.4723596705146895, 9.556512456719728, -12.860618054266808, 29.318853252612527,
     -11.919035262640463, -3.4221716863628213, -21.84272721071376, 12.886933114416156])
    fitness(best_weights, 0)

