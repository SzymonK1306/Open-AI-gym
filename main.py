import random

import gym
import numpy as np
from collections import deque
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam

# Initialize the gym environment
env = gym.make('MsPacman-v0', render_mode='human')

# Set the parameters
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
height, width, channels = env.observation_space.shape
action_size = env.action_space.n
batch_size = 32
n_episodes = 1000
output_dir = 'models/'

# Define the DQN agent class
class DQNAgent:
    def __init__(self, state_size, action_size, height, width, channels):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.height = height
        self.width = width
        self.channels = channels
        self.model = self._build_model(self.height, self.width, self.channels, self.action_size)

    # def _build_model(self):
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(24, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    #     return model
    def _build_model(self, h, w, ch, actions):
        # model = Sequential()
        # model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(h, w, ch)))
        # model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
        # model.add(Convolution2D(64, (3, 3), activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(actions, activation='linear'))
        model = models.load_model('models/again_51.h5')
        # model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state, verbose=0)
        print(act_values)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Initialize the DQN agent
agent = DQNAgent(state_size, action_size, height, width, channels)
# agent.load('models/new_20.h5')

# Training loop
for episode in range(n_episodes):
    state, _ = env.reset()
    state = np.reshape(state, (1, height, width, channels))
    done = False
    time = 0
    score = 0

    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, (1, height, width, channels))
        # agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        time += 1

    print(f"Total score: {score}")
    #
    # if len(agent.memory) > batch_size:
    #     agent.replay(batch_size)
    #
    # # Save the model weights every 100 episodes
    # if (episode + 1) % 10 == 0:
    #     agent.save(output_dir + f"weights_{episode + 1}.hdf5")

