__author__ = 'Aaron'

from Board import *


# PLAYER DEFINITION
class Player():
    def __init__(self, start_position, direction, player_color, player_number):
        self.positions = [start_position]
        self.color = player_color
        self.direction = direction
        self.number = player_number

        self.position_set = set(self.positions)
        self.transitions = []
        self.delete = None, None
        self.length = 1
        self.score = 0

    def update(self):
        """
        Calculate the next space to occupy.
        """
        row, column = self.get_position()
        if self.direction == 'LEFT':
            column += -1
        elif self.direction == 'RIGHT':
            column += 1
        elif self.direction == 'UP':
            row += -1
        elif self.direction == 'DOWN':
            row += 1
        return row, column

    def set_position(self, row, column):
        """
        Cement the calculated position once validated.
        Make sure not to add more positions than current snake length.
        :param row: row to move to
        :param column: column to move to
        :return -1 if doubling back on itself, 0 otherwise
        """
        if (row, column) in self.position_set:
            return -1
        else:
            self.transitions.append(self.direction)
            self.positions.append((row, column))
            self.position_set.add((row, column))

        # Restrict length of snake
        if len(self.positions) > self.length:
            self.delete = self.positions.pop(0), self.transitions.pop(0)

        return 0

    def get_position(self):
        """
        Returns current position of player.
        :return: (row, column)
        """
        return self.positions[-1]
