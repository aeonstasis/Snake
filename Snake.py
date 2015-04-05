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
    def __init__(self, default_state=None):
        """
        Initializes the state manager.
        :param default_state: state to begin in
        """
        self.state = None
        if default_state is None:
            self.go_to(MenuState())
        else:
            self.go_to(default_state())

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
    manager = StateManager()  # Set to 1 to have ANN play

    while running:
        clock.tick(FRAMES_PER_SEC)

        if pygame.event.get(QUIT):
            sys.exit(0)

        manager.state.handle_events(pygame.event.get())
        manager.state.update()
        manager.state.render(screen)
        pygame.display.flip()


# FITNESS FUNCTION
def fitness(weights):
    """
    Calculate the fitness function.
    :param weights: weights of the ANN being represented.
    :return: score of the ANN represented by the weights
    """
    pygame.init()
    pygame.display.set_caption("Snake")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((0, 0, 0))

    manager = StateManager(PlayState)
    while isinstance(manager.state, PlayState):
        clock.tick(FRAMES_PER_SEC)

        manager.state.update()
        if HEADLESS == 0:
            manager.state.render(screen)

    return manager.score


# PROGRAM EXECUTION
if __name__ == '__main__':
    # main()
    ann = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN)
    score = fitness(ann.get_weights())
    print(score)