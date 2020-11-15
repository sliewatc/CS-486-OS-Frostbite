import gym
from gym.spaces import Discrete
from gym.spaces import Box
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as ko
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.optimizers import Adam
import random

class DQNFrogger:
    def __init__(self, env):
        self.env = env      # OpenAI Gym environment for Frogger (ie. Frostbite)

        # Redefine the action space from the OpenAI Gym
        # The OpenAI Gym defined 18 actions: ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT',
        # 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
        # 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
        # We only need 5 actions: ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN']
        self.numActions = 5
        self.actions = [0, 2, 3, 4, 5]
        self.observationSize = env.reset().shape[0]
        self.memory = deque(maxlen=10000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.995
        self.learningRate = 0.005

        self.batchSize = 32

        # We use both a model and target model to help the QDN
        # self.model performs the actual prediction of which action to take
        # self.targetModel tracks the action we want the model to make (ie. our eventual goal)
        self.model = self.build_model()
        self.targetModel = self.build_model()

    def build_model(self):
        model = Sequential()
        # Input into the model will be a 1x128 vector, which is how the RAM input is shaped
        model.add(Dense(24, input_dim=observationSize, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.numActions))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, newState, done):
        self.memory.append([state, action, reward, newState, done])

    def replay(self):
        # We want to randomly take <batchSize> samples from the agent's "memory"
        if len(self.memory) < self.batchSize:
            return

        miniBatch = random.sample(self.memory, self.batchSize)

        # For each of our samples,
        for sample in miniBatch:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)


 def customReward(reward, action, fellIntoWater):
        # We want to add intermediate rewards as well, since the openAI gym only gives rewards on winning

        newReward = reward
        if (reward != 1):
            if (fellIntoWater):
                newReward = 0
            else:
                if (action == 0 or action == 3 or action == 4): # Left/Right/NOOP
                    newReward = 0.3
                elif (action == 2): # Up (behind)
                    newReward = 0.1
                elif (action == 5): # Down (forward)
                    newReward = 0.7

        return newReward


env = gym.make("Frostbite-ramDeterministic-v4")
env.seed(0)

env.render() # Render 1 frame at the current state
obs = env.reset() # Reset the environment to the initial observation
print(obs)
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation)

prevInfo = ""
actions = [0, 2, 3, 4, 5]

maxSteps = 900

for step in range(maxSteps):
    frame = env.render()
        action = env.action_space.sample() # Take a random action, within our defined action list
        while (not(action in actions)):
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        # If prevInfo != info, this means we lost a life (ie. we fell into the water) from our action
        reward = customReward(reward, action, (prevInfo != info and prevInfo != "") )
        prevInfo = info

        if done:
            # We need to reset the environment again
            # We either won, or lost all our lives
            print("Finished after {} timesteps".format(t+1))
            break
