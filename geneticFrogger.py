import gym
import matplotlib.pyplot as plt

env = gym.make("Frostbite-v0")
env.render() # Render 1 frame at the current state
print(env.env.get_action_meanings()) # Meanings of the possible actions

observation = env.reset() # Reset the environment to the initial observation

frames = []
for t in range(10000):
        frame = env.render()
        frames.append(frame)
        action = env.action_space.sample() # Take a random action
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            # We need to reset the environment again
            # We either won, or lost all our lives
            print("Finished after {} timesteps".format(t+1))
            break