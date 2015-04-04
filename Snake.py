__author__ = 'Aaron'

# IMPORTS
import sys

from States import *
from constants import *


# GLOBAL VARIABLES
clock = pygame.time.Clock()
HEADLESS = 0


# STATE MANAGER
class StateManager(object):
    def __init__(self, is_ann=0):
        """
        Initializes the state manager.
        :param is_ann: If set to 1, will use the neural network to play the game.
        """
        self.state = None
        if is_ann:
            self.go_to(PlayState())
            self.score = 0
        else:
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

    manager = StateManager(0)  # Set to 1 to have ANN play
    while manager.state != GameOverState:
        clock.tick(FRAMES_PER_SEC)

        manager.state.update()
        if HEADLESS == 0:
            manager.state.render(screen)

    return manager.score


# PROGRAM EXECUTION
if __name__ == '__main__':
    main()
