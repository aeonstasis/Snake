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
    best_weights = ([-11.821244852818976, -2.20180348221581, -3.1432738436821492, -1.846636210039, 16.088523601553067,
                     -5.078935578783578, -3.472459234268853, 18.849961330749537, 5.494860806469014, -8.688621882024375,
                     -5.856720526530262, 1.2341642990084858, 6.232410148602231, 5.448576709585462, 3.481965245895725,
                     -4.0643458346427535, -0.34901987435450654, -0.8664047204723968, 10.546458017161697,
                     2.312698770853382, -2.652342554395508, -11.135443022979118, 2.5112338176940145,
                     -2.797735846264893])
    fitness(best_weights, 0)

