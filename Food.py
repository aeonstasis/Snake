__author__ = 'Aaron'

from constants import *
import random
import Board

# FOOD DEFINITION
class Food:
    def __init__(self):
        self.position = Food.next_indices()
        self.old_center = None
        self.center = Board.get_center(self.position[0], self.position[1])

        self.radius = CELL_SIDE // 2 - 1
        self.score = 10
        self.color = YELLOW

    @staticmethod
    def next_indices():
        """
        Calculate next index randomly for a food item to appear.
        :return: row and column position
        """
        row = random.randint(0, NUM_ROWS - 1)
        column = random.randint(0, NUM_COLS - 1)
        return row, column

    def move(self):
        """
        Moves the food to a new location and saves its old location.
        Also, updates its center.
        """
        self.old_center = self.center
        self.position = Food.next_indices()
        self.center = Board.get_center(self.position[0], self.position[1])
