import gym
import tensorflow as tf
from geneticFrostbite import *
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    env = gym.make('Frostbite-ram-v0')
    env.seed(0)

    best_organism = Organism(env)
    best_organism.genome = load_model("best_organism")

    best_organism.run_game_with_render()