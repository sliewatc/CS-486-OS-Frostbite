import numpy as np
import random
import time
import copy
import gym
import tensorflow as tf
import tensorflow.keras.optimizers as ko
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class Organism:
    def __init__(self, env):
        self.env = env
        self.game_len = 900
        self.game_runs = 5

        self.numActions = 5
        self.actions = [0, 2, 3, 4, 5]
        self.stateSize = env.reset().shape[0]
        self.learningRate = 0.005
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.999
        self.startLearning = 100 # At 100+ steps, we will decay the epsilon

        # genome is a neural network model
        self.genome = self.build_genome_model()

    def build_genome_model(self):
        # create the genome for this organism, which will be a sequential model of dense layers
        model = Sequential()
        # Input into the model will be a 1x128 vector, which is how the RAM input is shaped
        model.add(Dense(24, input_shape=(self.stateSize,), activation="relu", use_bias=False))
        model.add(Dense(24, activation="relu", use_bias=False))
        model.add(Dense(24, activation="relu", use_bias=False))
        model.add(Dense(self.numActions, activation='linear', use_bias=False))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learningRate))
        return model

    def run_game(self):
        total_reward = 0
        obs = self.env.reset() # obs is the observation of the initial state of the game

        # run game loop for game_len steps
        for step in range(self.game_len):
            # env.render()

            if np.random.random() < self.epsilon:
                # action = self.env.action_space.sample()

                action = random.choice(self.actions)

                # Keep choosing an action until we get one in our smaller action space
                # while (not (action in self.actions)):
                #     action = self.env.action_space.sample()

            else:
                obs = np.reshape(obs, (1, self.stateSize))
                action = np.argmax(self.genome.predict(obs)[0])

            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

            if step >= self.startLearning:
                self.epsilon *= self.epsilonDecay
                if self.epsilon < self.epsilonMin:
                    self.epsilon = self.epsilonMin


        return total_reward

    def run_game_with_render(self):
        total_reward = 0
        obs = self.env.reset() # obs is the observation of the initial state of the game

        # run game loop for game_len steps
        for t in range(self.game_len):
            self.env.render()

            obs = np.reshape(obs, (1, self.stateSize))
            action = np.argmax(self.genome.predict(obs)[0])

            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return total_reward

def calc_fitness(org):
        total = 0
        for i in range(org.game_runs):
            total += org.run_game()
        return total / org.game_runs

def crossover(org1, org2):
    # create child organism
    child_org = Organism(org1.env)

    # for each layer in the genome, randomly use some of either parent's values
    # for parent1_layer, parent2_layer in zip(org1.genome.layers, org2.genome.layers):
    for i in range(len(org1.genome.layers)):
        parent1_layer = org1.genome.layers[i]
        parent2_layer = org2.genome.layers[i]
        org1_weights = copy.deepcopy(parent1_layer.get_weights())
        org2_weights = copy.deepcopy(parent2_layer.get_weights())

        parent_layer_shape = np.array(org1_weights).shape

        org1_weights = np.array(org1_weights).flatten()
        org2_weights = np.array(org2_weights).flatten()

        child_layer_weights = org1_weights.copy()

        # randomly set child weights to use either parents' values
        for j, w in enumerate(org2_weights):
            rand = np.random.uniform()
            if rand > 0.5:
                child_layer_weights[j] = w
        
        child_layer_weights = child_layer_weights.reshape(parent_layer_shape)
        
        child_org.genome.layers[i].set_weights(child_layer_weights)

    return child_org

def mutation(org, mutation_rate=0.05):
    for i, layer in enumerate(org.genome.layers):
        weights = copy.deepcopy(layer.get_weights())

        layer_shape = np.array(weights).shape

        weights = np.array(weights).flatten()

        for j in range(len(weights)):
            rand = np.random.uniform()
            if rand < mutation_rate:
                weights[j] = random.uniform(-1.0, 1.0)

        weights = weights.reshape(layer_shape)

        org.genome.layers[i].set_weights(weights)

    return org
            
if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    env = gym.make('Frostbite-ram-v0')
    env.seed(0)

    generations = 5
    population_size = 5
    elites_num = 1
    start = time.time()
    population = [Organism(env) for _ in range(population_size)]

    for idx in range(generations):
        # calculate fitness of all organisms in population
        # fitness_scores = [o.calc_fitness() for o in population]

        # fitness_scores = []
        # for i, o in enumerate(population):
        #     print("Calculating fitness for organism " + str(i) + " in generation " + str(idx))
        #     fitness_scores.append(calc_fitness(o))

        fitness_scores = {}
        for i, o in enumerate(population):
            print("Calculating fitness for organism " + str(i) + " in generation " + str(idx))
            fitness_scores[o] = calc_fitness(o)



        print('Generation %d : max score = %0.2f' %(idx, max(fitness_scores.values())))

        # population_ranks = list(reversed(np.argsort(fitness_scores)))
        # elite_set = [population[x] for x in population_ranks[:5]]

        sorted_organisms = [k for k, v in sorted(fitness_scores.items(), key=lambda item: item[1])]
        elite_set = [org for org in sorted_organisms[:elites_num]]

        elite_set[0].genome.save("best_organism")

        select_probs = 0
        if (np.sum(fitness_scores) != 0):
            select_probs = np.array(list(fitness_scores.values())) / np.sum(list(fitness_scores.values()))

        child_set = [crossover(
            population[np.random.choice(range(population_size), p=select_probs)], 
            population[np.random.choice(range(population_size), p=select_probs)])
            for _ in range(population_size - elites_num)]
        
        mutated_list = [mutation(p) for p in child_set]
        population = elite_set
        population += mutated_list

    # fitness_scores = [o.calc_fitness() for o in population]
    fitness_scores = [calc_fitness(o) for o in population]
    best_organism = population[np.argmax(fitness_scores)]

    end = time.time()
    print('Best organism score = %0.2f. Time taken = %4.4f' %(np.max(fitness_scores), (end-start)))   

    best_organism.genome.save("best_organism")
