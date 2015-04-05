__author__ = 'Aaron'

# IMPORTS
import sys

from States import *
from constants import *


# GLOBAL VARIABLES
clock = pygame.time.Clock()


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
def fitness(ann, headless=1):
    """
    Calculate the fitness function.
    :param ann: the ANN being represented.
    :return: score of the ANN represented
    """
    pygame.init()
    pygame.display.set_caption("Snake")

    screen = None
    ai_fps = 120 if headless else FRAMES_PER_SEC

    if headless == 0:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.fill((0, 0, 0))

    manager = StateManager(ann)
    manager.go_to(PlayState())

    while not isinstance(manager.state, GameOverState):
        clock.tick(ai_fps)

        manager.state.update()

        if headless == 0:
            manager.state.render(screen)
            pygame.display.flip()

    return manager.fitness


# PROGRAM EXECUTION
if __name__ == '__main__':
    main()

    # Test ANN produced from GA
    """
    ann = NeuralNet(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN, NUM_PER_HIDDEN)
    score = fitness(ann, 0)
    print(score)
    """
