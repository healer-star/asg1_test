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

**test.py**

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
