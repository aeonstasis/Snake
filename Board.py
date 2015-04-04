__author__ = 'Aaron'

from constants import *


# BOARD DEFINITION
class Board:
    def __init__(self):
        """
        Initializes board matrix to zeros and draws it to the screen.
        """
        self.board = [[0 * i * j for i in range(NUM_COLS)] for j in range(NUM_ROWS)]

    def reset(self):
        """
        Reset all cells in the board to their default unfilled state.
        """
        for row in range(NUM_ROWS):
            for column in range(NUM_COLS):
                self.board[row][column] = EMPTY

    def set_cell(self, row, column, number):
        """
        Set the matrix cell with the corresponding row and column to the player number.
        Used to determine what state the cell is in.
        :param row: row to set
        :param column: column to set
        :param number: player id
        """
        self.board[row][column] = number

    def get_cell(self, row, column):
        """
        Get the player number from the matrix cell with the corresponding row and column.
        Used to determine what state the cell is in.
        :param row: row to set
        :param column: column to set
        :return number: state of cell
        """
        if row < 0 or row >= NUM_ROWS or column < 0 or column >= NUM_COLS:
            return OUT_BOUNDS
        return self.board[row][column]

    @staticmethod
    def distance_to(position0, position):
        """
        Calculate the direction from one cell to another. Returns position - position0.
        :return: (delta_x, delta_y)
        """
        (row0, column0) = position0
        (row1, column1) = position
        return row1 - row0, column1 - column0

    @staticmethod
    def cell_equals(position0, position1):
        """
        Returns whether the two positions are equal
        :param position0: first position
        :param position1: second position
        :return: True if the positions are equal, False otherwise
        """
        return (position0[0] == position1[0]) and (position0[1] == position1[1])

    @staticmethod
    def get_center(row, column):
        """
        Given a cell, return the x, y pixel location of the center of the cell.
        :param row: row of cell
        :param column: column of cell
        :return: (center_x, center_y)
        """
        center_x = (CELL_MARGIN + CELL_SIDE) * column + HORIZONTAL_OFFSET + (CELL_SIDE // 2)
        center_y = (CELL_MARGIN + CELL_SIDE) * row + VERTICAL_OFFSET + (CELL_SIDE // 2)
        return int(center_x), int(center_y)

    def check_collision(self, position):
        """
        When a player attempts to move to a new cell, check that is it empty or food.
        Also performs a bounds check.
        :param position: row and column of the cell
        :return: -1 if the cell has a non-zero value or is out of bounds
                  0 if the cell is valid and empty
                  1 if the cell contains food
        """
        row, column = position
        out_bounds = row < 0 or row >= NUM_ROWS or column < 0 or column >= NUM_COLS
        if out_bounds:
            return -1
        elif self.board[row][column] == EMPTY:
            return 0
        elif self.board[row][column] == FOOD:
            return 1
