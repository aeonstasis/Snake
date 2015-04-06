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
    # main()

    # Test ANN produced from GA
    best_weights = ([0.7986857139873349, 0.17580650976454404, 0.40308316205974803, -0.05788492025228664,
                     -1.7217180817441924, 0.06459429037369824, -0.7281467389549743, 1.6084745036809662,
                     0.04988494877801131, -0.045405613571901204, -1.4008008275559443, -0.25691494229359346,
                     3.533971541240774, 1.214817359571597, -0.2644190297848732, 3.095122855541602,
                     -1.2854250764912705, -0.2475636733260691, 2.523655582279237, 1.2254136885302618,
                     5.514600694104605, -2.4350228323947434, -1.2428349618364274, -0.028879797534371865])
    fitness(best_weights, 0)

