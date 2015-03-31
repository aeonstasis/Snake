__author__ = 'Aaron'

# IMPORTS
from States import *
from constants import *
from pygame.locals import *
import sys

# GLOBAL VARIABLES
clock = pygame.time.Clock()


# STATE MANAGER
class StateManager(object):
    def __init__(self):
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

# PROGRAM EXECUTION
if __name__ == '__main__':
    main()
