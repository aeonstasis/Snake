# Snake
This is an implementation of 2D Snake using pygame. The game is being used to to implement an artificial neural network for AI learning.

## Gameplay
The player controls a "snake", indicated by a colored square, with the arrow keys, moving around a grid.
Colliding with itself or a wall results in a game over, while colliding with "food", yellow circles, picks it up.
The objective of the game is to collect "food" items that respawn randomly in the grid when the previous one is collected.
When the snake collects food, its length will increase by one, making movement progressively more difficult.

## Implementation Details
The game is laid out in a very OOP fashion. 

constants.py, effectively a configuration file, stored constants and values derived from them. 

A lightweight main class, Snake.py, implemented the animation cycle using a finite state machine to delegate behavior to additional state classes. 

A separate class, States.py, defined the behavior of those different states that would be processed by the main class. 

Finally, Player, Food, and Board classes defined relevant variables and the necessary methods to operate on them. Board also contained static methods to perform basic grid operations, such as collision checking and equality checks.

### constants.py 
This file holds game constants and values derived from them that determine layout and behavior details.

This allows for easy tweaking of parameters and avoids using "magic numbers". This also allows for potential implementation of dynamic resizing.

### Snake.py 
This file contains the runner main() method and the StateManager class.

The StateManager class points to the current state and also assigns itself as a field to that state. This reference allows for the current state to be changed by the state classes themselves.

main() follows the animation cycle and delegates behavior to one of several States, which encapsulate all the information for the current state of the game. 

### States.py 
This file holds the states that determine actual game behavior. An interface, State, contains update(), render(), and handle_event() methods. 

At each tick, update() updates all variables and game internals. render() displays necessary information to the screen, and handle_event() listens for keypresses that would change the state.

## Artificial Neural Network
I will use PyBrain.

## In Progress
Menu with player color selection <br>
Generalize game to Tron <br>
Dynamic resizing and maximizing window

