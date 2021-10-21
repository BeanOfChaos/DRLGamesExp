import gym
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import sys

import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
MAXIT = 1000
ACTIONS = np.arange(0, 9, 1)
DISCOUNT = 1
DO_RANDOM_STEPS = 10
VERBOSE = "--verbose" in sys.argv
DO_RENDER = "--render" in sys.argv

if __name__ == "__main__":
    sys.stdout.write("\n")
    #env = gym.make('MsPacman-v0')
    env = gym.make('MsPacman-ram-v0')

    sys.stdout.write("Creating model.\n")

    model = Sequential()
    """
    model.add(kl.Reshape((210, 160, 3), input_shape=(210, 160, 3)))
    model.add(kl.Conv2D(filters=4, kernel_size=3, strides=1))
    model.add(kl.MaxPooling2D(pool_size=4))
    model.add(kl.Conv2D(filters=8, kernel_size=3, strides=2))
    #model.add(kl.Conv2D(filters=8, kernel_size=3, strides=2))
    model.add(kl.MaxPooling2D(pool_size=4))
    model.add(kl.Flatten())
    """
    model.add(kl.Dense(256, activation="tanh", input_shape=(128,)))
    model.add(kl.Dense(len(ACTIONS)))

    opt = Adam(lr=0.05)
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    ep_rewards = []
    history = []
    for i in range(MAXIT):
        print(f"\nIteration number {i}")
        expl_prob = np.exp(- 64 * i / MAXIT)
        print(f"Random action with probability {expl_prob}")
        current_it_rewards = []
        done = False
        new_obs = env.reset().astype(np.float16)
        new_obs = new_obs.reshape(1, *new_obs.shape)
        j = 0
        while not done:
            obs = new_obs
            if DO_RENDER:
                env.render()
            estimate = model.predict(obs).flatten()
            if VERBOSE:
                print(estimate)
            #clear_session()
            if np.random.random() < expl_prob:
                if True or j % DO_RANDOM_STEPS == 0:
                    action = np.random.choice(ACTIONS)
                    if VERBOSE:
                        print(f"Selected random action {action}...")
                else:
                    if VERBOSE:
                        print(f"Redoing previous random action {action}...")
                j += 1
            else:
                j = 0
                action = ACTIONS[np.argmax(estimate)]
                if VERBOSE:
                    print(f"Selected predicted action {action}...")
            new_obs, reward, done, info = env.step(action)
            current_it_rewards.append(reward)
            new_obs = new_obs.astype(np.float16).reshape(1, *new_obs.shape)
            history.append((new_obs, obs, action, reward, estimate))

            if len(history) == BATCH_SIZE:
                if VERBOSE:
                    print(f"History full, training...")
                xs, ys = [], []
                for k, step in enumerate(history[:-1]):
                    new_obs, obs, action, reward, estimate = step
                    tmp = np.copy(estimate)
                    tmp[action] = reward + DISCOUNT * max(history[k+1][4].flatten())
                    xs.append(obs)
                    ys.append(tmp)
                if VERBOSE:
                    print("Resuming exploration.")
                model.train_on_batch(tf.convert_to_tensor(np.concatenate(xs)),
                                     tf.convert_to_tensor(np.asarray(ys)))
                history = []
        tmp = sum(current_it_rewards)
        print(f"Episode total reward: {tmp}")
        ep_rewards.append(tmp)
    model.save("pacman_model.json")
    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid = True
    plt.savefig("reward_over_time.png")
