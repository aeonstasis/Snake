__author__ = 'Aaron'

# IMPORTS
import sys

from States import *
from constants import *
from NeuralNet import *


# GLOBAL VARIABLES
clock = pygame.time.Clock()
HEADLESS = 0


# STATE MANAGER
class StateManager(object):
    def __init__(self, ann=None):
        """
        Initializes the state manager.
        Contains "global" variables to hold neural network and score.
        """
        self.ann = ann
        self.score = 0  # used to determine fitness
        self.moves = []  # used to track the moves made in the game for debugging purposes

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
def fitness(ann):
    """
    Calculate the fitness function.
    :param ann: the ANN being represented.
    :return: score of the ANN represented
    """
    pygame.init()
    pygame.display.set_caption("Snake")

    screen = None
    if HEADLESS == 0:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.fill((0, 0, 0))

    manager = StateManager(ann)
    manager.go_to(MenuState())
    while not isinstance(manager.state, GameOverState):
        if HEADLESS == 0:
            clock.tick(FRAMES_PER_SEC)

        manager.state.update()
        manager.state.render(screen)

    return manager.score


# PROGRAM EXECUTION
if __name__ == '__main__':
    """
    main()
    """
    ann = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN)
    score = fitness(ann)
    print(score)

