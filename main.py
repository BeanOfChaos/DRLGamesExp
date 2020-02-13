import gym
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import sys

import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 1024
MAXIT = 1000
ACTIONS = np.arange(0, 9, 1)
DISCOUNT = 1


if __name__ == "__main__":
    sys.stdout.write("\n")
    env = gym.make('MsPacman-v0')

    sys.stdout.write("Creating model.\n")
    model = Sequential()
    model.add(kl.Reshape((210, 160, 3), input_shape=(210, 160, 3)))
    model.add(kl.Conv2D(filters=8, kernel_size=2, strides=2))
    model.add(kl.MaxPooling2D(pool_size=4))
    model.add(kl.Conv2D(filters=16, kernel_size=2, strides=2))
    model.add(kl.MaxPooling2D(pool_size=4))
    model.add(kl.Flatten())
    model.add(kl.Dense(256))
    model.add(kl.Dense(len(ACTIONS)))

    opt = Adam(lr=0.05)
    model.compile(optimizer=opt, loss='mse')
    model.summary()

    ep_rewards = []
    history = []
    for i in range(MAXIT):
        sys.stdout.write(f"\nIteration number {i}\n")
        expl_prob = np.exp(- 8 * i / MAXIT)
        sys.stdout.write(f"Random action with probability {expl_prob}\n")
        current_it_rewards = []
        done = False
        new_obs = env.reset().astype(np.float16)
        new_obs = new_obs.reshape(1, *new_obs.shape)
        while not done:
            obs = new_obs
            # uncomment to show bot playing
            # env.render()
            estimate = model.predict(obs).flatten()
            clear_session()
            if np.random.random() < expl_prob:
                action = np.random.choice(ACTIONS)
            else:
                action = ACTIONS[np.argmax(estimate)]
            new_obs, reward, done, info = env.step(action)
            current_it_rewards.append(reward)
            new_obs = new_obs.astype(np.float16).reshape(1, *new_obs.shape)
            history.append((new_obs, obs, action, reward, estimate))

            if len(history) == BATCH_SIZE:
                sys.stdout.write(f"History full, training...\n")
                xs, ys = [], []
                for i, step in enumerate(history):
                    new_obs, obs, action, reward, estimate = step
                    tmp = np.copy(estimate)
                    tmp[action] = reward + DISCOUNT * \
                        max(model.predict(new_obs).flatten())
                    xs.append(obs)
                    ys.append(tmp)
                sys.stdout.write("Resuming exploration.\n")
                model.train_on_batch(np.concatenate(xs), np.asarray(ys))
                history = []
        tmp = sum(current_it_rewards)
        sys.stdout.write(f"Episode total reward: {tmp}\n")
        ep_rewards.append(tmp)
    model.save("pacman_model.json")
    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid = True
    plt.savefig("reward_over_time.png")
