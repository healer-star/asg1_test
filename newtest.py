import tensorflow as tf
import numpy as np
import random
import os
import Map  # Assume Map is a module defining the game board and related parameters

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.n_input = Map.mapsize * Map.mapsize  # Input size based on the game board size
        self.n_output = 1  # Output size (Q-value for each action)
        self.current_q_step = 0  # Current step for Q-network
        self.avg_loss = 0  # Average loss for training
        self.train_times = 0  # Number of training steps

        # Build the Q-Network
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')  # First convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')  # Second convolutional layer
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')  # Third convolutional layer
        self.max_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')  # Max pooling layer
        self.flatten = tf.keras.layers.Flatten()  # Flatten layer to convert 2D output to 1D
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')  # First dense layer
        self.dense2 = tf.keras.layers.Dense(self.n_output)  # Output layer

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer with learning rate 0.001

        # Model saving
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)  # Checkpoint for saving model state
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory='Saver', max_to_keep=3)  # Manager for saving checkpoints


    def call(self, inputs):
        # Ensure input data is of type float32
        inputs = tf.cast(inputs, tf.float32)
        x = tf.reshape(inputs, [-1, Map.mapsize, Map.mapsize, 1])  # Reshape input to 4D tensor (batch_size, height, width, channels)
        x = self.conv1(x)  # Pass through the first convolutional layer
        x = self.max_pool(x)  # Apply max pooling
        x = self.conv2(x)  # Pass through the second convolutional layer
        x = self.max_pool(x)  # Apply max pooling
        x = self.conv3(x)  # Pass through the third convolutional layer
        x = self.max_pool(x)  # Apply max pooling
        x = self.flatten(x)  # Flatten the output
        x = self.dense1(x)  # Pass through the first dense layer
        x = self.dense2(x)  # Pass through the output layer
        return x

    def restore(self):
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saver")  # Directory for saving checkpoints
        print(f"Checkpoint directory: {checkpoint_dir}")

        if not os.path.exists(checkpoint_dir):  # Check if the checkpoint directory exists
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)  # Get the latest checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from latest checkpoint: {latest_checkpoint}")
        else:
            print(f"No checkpoint found in directory: {checkpoint_dir}")

    def train_step(self, boards, scores):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation
            predictions = self(boards, training=True)  # Get predictions from the model
            loss = tf.reduce_mean(tf.square(predictions - scores))  # Compute the mean squared error loss
        gradients = tape.gradient(loss, self.trainable_variables)  # Compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # Apply gradients to update model parameters
        return loss

    def computerPlay(self, IsTurnWhite):
        board = np.array(Map.blackBoard if IsTurnWhite else Map.whiteBoard)  # Get the current board state
        boards = []
        positions = []
        for i in range(Map.mapsize):  # Iterate over the board to find empty positions
            for j in range(Map.mapsize):
                if board[j][i] == Map.backcode:  # Check if the position is empty
                    predx = np.copy(board)  # Create a copy of the board
                    predx[j][i] = Map.blackcode  # Simulate placing a piece
                    boards.append(predx)  # Add the simulated board to the list
                    positions.append([i, j])  # Add the position to the list

        if not positions:  # If no empty positions are found, return 0
            return 0, 0, 0

        # Convert the list of boards to a NumPy array and ensure data type is float32
        boards = np.array(boards, dtype=np.float32)
        nextStep = self(boards)  # Get Q-values for each possible move
        maxValue = np.max(nextStep) + random.randint(0, 10) / 1000  # Add a small random value to break ties
        max_index = np.argmax(nextStep)  # Get the index of the best move
        maxx, maxy = positions[max_index]  # Get the coordinates of the best move

        return maxx, maxy, maxValue

    def TrainOnce(self, winner):
        board1 = np.array(Map.mapRecords1)  # Get recorded board states for player 1
        board2 = np.array(Map.mapRecords2)  # Get recorded board states for player 2
        step1 = np.array(Map.stepRecords1)  # Get recorded steps for player 1
        step2 = np.array(Map.stepRecords2)  # Get recorded steps for player 2
        scoreR1 = np.array(Map.scoreRecords1)  # Get recorded scores for player 1
        scoreR2 = np.array(Map.scoreRecords2)  # Get recorded scores for player 2

        # Reshape the recorded data to match the input shape of the model
        board1 = np.reshape(board1, [-1, Map.mapsize, Map.mapsize])
        board2 = np.reshape(board2, [-1, Map.mapsize, Map.mapsize])
        step1 = np.reshape(step1, [-1, Map.mapsize, Map.mapsize])
        step2 = np.reshape(step2, [-1, Map.mapsize, Map.mapsize])

        # Modify the boards to simulate the opponent's moves
        board1 = (board1 * (1 - step1)) + step1 * Map.blackcode
        board2 = (board2 * (1 - step2)) + step2 * Map.blackcode

        # Compute target scores for training
        scores1 = scoreR2[:-1] * -0.9
        scores2 = scoreR1[1:] * -0.9

        if winner == 2:  # If player 2 wins, append a reward of 1.0 to scores1
            scores1 = np.append(scores1, [1.0])
        if winner == 1:  # If player 1 wins, append a reward of 1.0 to scores2
            scores2 = np.append(scores2, [1.0])

        # Concatenate the boards and scores for both players
        boards = np.concatenate([board1, board2], axis=0)
        scores = np.concatenate([scores1, scores2], axis=0)

        # Convert the data to TensorFlow tensors
        boards = tf.convert_to_tensor(boards, dtype=tf.float32)
        scores = tf.convert_to_tensor(scores, dtype=tf.float32)

        # Perform a training step
        loss = self.train_step(boards, scores)
        self.avg_loss += loss  # Accumulate the loss
        self.train_times += 1  # Increment the training step counter

        # Print average loss and save the model every 100 steps
        if Map.AutoPlay % 100 == 0:
            print(f'Train avg loss: {self.avg_loss / self.train_times}, Total steps: {Map.AutoPlay}')
            self.avg_loss = 0  # Reset average loss
            self.train_times = 0  # Reset training step counter
            self.checkpoint_manager.save()  # Save the model checkpoint

    def PlayWithHuman(self):
        self.restore()  # Restore the model from the latest checkpoint
        Map.PlayWithComputer = self.computerPlay  # Set the computer's play function to the DQN's computerPlay method
        Map.TrainNet = self.TrainOnce  # Set the training function to the DQN's TrainOnce method
        Map.ShowWind()  # Call the function to display the game window (assumed to be defined in the Map module)
        
if __name__ == '__main__':
    dqn = DQN()
    dqn.PlayWithHuman()