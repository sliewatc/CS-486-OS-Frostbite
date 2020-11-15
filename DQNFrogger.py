import gym
from gym.spaces import Discrete
from gym.spaces import Box
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as ko
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
import collections

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
        self.stateSize = env.reset().shape[0]
        self.memory = collections.deque(maxlen=10000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.999
        self.startLearning = 100 # At 100+ steps, we will decay the epsilon
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
        model.add(Dense(24, input_shape=(self.stateSize,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.numActions, activation='linear'))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learningRate))
        return model

    def remember(self, prevState, action, reward, newState, done):
        self.memory.append([prevState, action, reward, newState, done])

    def replay(self):
        # We want to randomly take <batchSize> samples from the agent's "memory"
        if len(self.memory) < self.batchSize:
            return

        miniBatch = random.sample(self.memory, self.batchSize)

        statesBatch = np.array([i[0] for i in miniBatch])
        actionsBatch = np.array([i[1] for i in miniBatch])
        rewardsBatch = np.array([i[2] for i in miniBatch])
        nextStatesBatch = np.array([i[3] for i in miniBatch])
        donesBatch = np.array([i[4] for i in miniBatch])

        statesBatch = np.squeeze(statesBatch)
        nextStatesBatch = np.squeeze(nextStatesBatch)

        targetQValue = rewardsBatch + self.gamma * (np.amax(self.model.predict_on_batch(nextStatesBatch), axis=1)) * (1 - donesBatch)
        targetsPrediction = self.model.predict_on_batch(statesBatch)

        for i, action in enumerate(actionsBatch):
            # Must decrement the action by 1 for actions 1,2,3,4, since we took away the FIRE action from the original action space
            if action == 0:
                targetsPrediction[i][action] = targetQValue[i]
            else:
                targetsPrediction[i][action - 1] = targetQValue[i]

        self.model.fit(statesBatch, targetsPrediction, epochs=1, verbose=0)

    def takeAction(self, state, step):
        # We use a decaying epsilon-greedy approach
        # If step >= 100, then we start decaying epsilon (since we want the agent to start using the Q network)
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()

            # Keep choosing an action until we get one in our smaller action space
            while (not (action in self.actions)):
                action = self.env.action_space.sample()
        else:
            state = np.reshape(state, (1, self.stateSize))
            action = np.argmax(self.model.predict(state)[0])

        if step >= self.startLearning:
            self.epsilon *= self.epsilonDecay
            if self.epsilon < self.epsilonMin:
                self.epsilon = self.epsilonMin

        return action

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

def main():
    env = gym.make("Frostbite-ramDeterministic-v4")
    env.seed(0)

    numTrials = 100
    maxSteps = 900
    prevInfo = ""

    dqnAgent = DQNFrogger(env=env)

    for trial in range(numTrials):
        currentState = env.reset() # Reset to the starting state

        for step in range(maxSteps):
            frame = env.render()
            action = dqnAgent.takeAction(currentState, step) # Take a random action, within our defined action list

            nextState, reward, done, info = env.step(action)
            reward = customReward(reward, action, (prevInfo != info and prevInfo != ""))
            prevInfo = info

            dqnAgent.remember(currentState, action, reward, nextState, done)
            dqnAgent.replay()

            currentState = nextState

            if done:
                # We need to reset the environment again
                # We either won, or lost all our lives
                print("Finished after {} timesteps".format(step+1))
                break

if __name__ == "__main__":
    main()
