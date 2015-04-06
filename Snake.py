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
    [-13.539177782779761, 10.044627569130881, 1.589651570663407, -24.349519296801994, -14.370223951975165,
     -7.172281127131621, 3.4175062022306246, -1.3134393478957211, 10.236309431125237, -7.162719390613946,
     2.9881775471342076, 7.307938923187242, -6.339337092226408, -3.6601892771693336, 17.208349102589644,
     -3.017681499511462, 2.9700315108808577, 14.528190748878364, -14.188022987802368, -7.127745575378374,
     1.1333060305379374, -0.5912357842477052, -4.865173815688814, 5.143910903386375])
    fitness(best_weights, 0)

