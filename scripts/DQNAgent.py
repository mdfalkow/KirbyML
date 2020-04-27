import random
import time
from collections import deque

import numpy as np
from retro.retro_env import RetroEnv
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Dropout,
                                     Flatten, MaxPooling1D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from .ModifiedTensorBoard import ModifiedTensorBoard


class DQNAgent:
    """
    A General implementation of a Deep Q-learning Agent.
    From [pythonprogramming.net](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/)
    """

    def __init__(self, env: RetroEnv,
                 model_path: str,
                 discount: float,
                 rms_min: int,
                 rms_max: int,
                 minibatch_size: int,
                 target_update_freq: int,
                 model_prefix: str):
        """Construct a new DQN agent"""
        self.discount = discount  # Discount ratio for Q-learning
        self.rms_min = rms_min  # min replay memory size
        self.rms_max = rms_max  # max replay memory size
        self.minibatch_size = minibatch_size  # Size of the minibatch durign training
        # Number of episodes that must pass before updating the target_model
        self.target_update_freq = target_update_freq
        # Name of the model (for storing progress)
        self.model_prefix = model_prefix

        # Main model
        self.model = self.create_model(env, model_path)

        # Target network
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.rms_max)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(model_name=self.model_prefix,
                                               log_dir=f'logs/{self.model_prefix}-{int(time.time())}')

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, env, model_path=None):
        """Creates or loads a previous model"""
        if model_path:
            print(f'Loading model {model_path}')
            model = load_model(model_path)
            print(f'Loaded {model_path} successfully')
        else:
            model = Sequential()

            # OBSERVATION_SPACE_VALUES = 144 x 160
            model.add(Conv1D(32,
                             (3,),
                             input_shape=env.observation_space.shape[:2]))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))

            model.add(Conv1D(32, (3,)))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))

            # this converts our 3D feature maps to 1D feature vectors
            model.add(Flatten())
            model.add(Dense(32))

            # ACTION_SPACE_SIZE = how many choices (9)
            model.add(Dense(env.action_space.n, activation='linear'))
            model.compile(loss="mse",
                          optimizer=Adam(lr=0.01),
                          metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        """
        Adds step's data to a memory replay array 
        A transition is the tuple(observation space, action, reward, new observation space, done)
        """
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        """Trains main network every step during episode"""

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.rms_min:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([t[0] for t in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it,
        # otherwise main network should be queried
        new_current_states = np.array([t[3] for t in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for i, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states,
            # otherwise set it to 0
            # almost like with Q Learning,
            # but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255,
                       np.array(y),
                       batch_size=self.minibatch_size,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network
        # with weights of main network
        if self.target_update_counter > self.target_update_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        """
        Queries main network for Q values given  current observation space (environment state)
        """
        return self.model.predict(np.array(state)
                                  .reshape(-1, *state.shape)/255)[0]
