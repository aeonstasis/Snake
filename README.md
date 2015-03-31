# Snake
This is an implementation of 2D Snake using pygame. The game is being used to to implement an artificial neural network for AI learning.

# Gameplay
The player controls a "snake", indicated by a colored square, with the arrow keys, moving around a grid.
Colliding with itself or a wall results in a game over, while colliding with "food", yellow circles, picks it up.
The objective of the game is to collect "food" items that respawn randomly in the grid when the previous one is collected.
When the snake collects food, its length will increase by one, making movement progressively more difficult.

# Implementation Details
The game is laid out in a very OOP fashion. Constants and values derived from them were stored in an effective configuration file. A main class, Snake.py, implemented the animation cycle and defined classes for the board, player, and food. A separate class, States.py, defined the behavior of different states that would be processed by the main class.

# constants.py 
This file holds game constants and values derived from them that determine layout and behavior details.
This allows for easy tweaking of parameters and avoids using "magic numbers". This also allows for potential implementation of dynamic resizing.

# Snake.py 
This file contains the runner main() method, as well as class definitions for the Player, Board, and Food.
The Food, Player, and Board classes define their respective instance variables and methods that update them. Board contains static methods for collision checking and cell operations. 
main() follows the animation cycle and delegates behavior to one of several States. 

# States.py 

# Artificial Neural Network

# In Progress
