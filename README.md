# README
## Project Overview
This project implements an Intelligent Gomoku Game using a combination of a graphical user interface (GUI) and a Deep Q-Network (DQN) for AI-based gameplay. The game allows users to play against an AI opponent or watch the AI play against itself. The project consists of two main modules: Map.py and test.py.
## Files Description
**Map.py**

This module defines the game board, GUI, and core game logic for the Gomoku game. It includes:

**Game Board Setup**: Initializes an 8x8 game board with graphical representation using Tkinter.

**Game Logic**: Handles player moves, checks for win conditions, and manages game state.

**AI Integration**: Provides functions to integrate with an AI model for automated gameplay.

**Data Recording**: Saves game states and results to datasets for training the AI model.

**Auto-Play and Training**: Supports automated game playing and training modes.

**newtest.py**

This module implements a **Deep Q-Network (DQN)** using TensorFlow to serve as the AI opponent. It includes:

**DQN Model**: Defines the neural network architecture for the DQN.

**Training and Inference**: Implements methods for training the DQN using recorded game data and making moves based on the learned policy.

**Integration with Game**: Hooks into the game logic defined in Map.py to enable AI gameplay and training.

## Dependencies
**Python**: Tested with Python 3.8+.

**TensorFlow**: Required for the DQN implementation.

**NumPy**: Used for numerical operations.

**Tkinter**: Built-in Python library for creating the GUI.

## How to Run
**Install Dependencies**:

```pip install tensorflow numpy```

**Run the Game**:

Navigate to the project directory.

Execute the following command to start the game:

```python newtest.py```

**Game Controls**:

**Left Mouse Click**: Place a piece on the board.

**Auto-Training Button**: Triggers 1000 automated training games.

**Auto-Move Button**: Makes a single automated move by the AI.

## Features
**Graphical User Interface**: Interactive game board with Tkinter.

**AI Opponent**: Deep Q-Network (DQN) implemented in newtest.py.

**Auto-Play Mode**: Automated gameplay for training and demonstration.

**Data Recording**: Saves game states and results for training the DQN.

**Training Integration**: Trains the DQN using recorded game data.

## Notes
The game board size is set to 8x8 by default. Modify mapsize in Map.py to change the board size.

The DQN model uses convolutional layers to process the game board state. Adjust the model architecture in test.py if needed.

Ensure the DataSets and Saver directories exist or the script will create them automatically.
