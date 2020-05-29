from os import sched_getaffinity
from pathlib import Path

import gym
import numpy as np
from pyvirtualdisplay import Display

ENV = 'CarRacing-v0'
AVAILABLE_CORES = len(sched_getaffinity(0))
ROLLOUTS_PER_CORE = 100 // AVAILABLE_CORES


def key_press(k, mod):
    from pyglet.window import key
    if k == key.ENTER:
        global restart
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8   # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    from pyglet.window import key
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0


def gauss_rand_walk(action_space, dt, seq_len=None):
    """
    Gaussian Random Walk simulating a Wiener process (Brownian motion):
    See: https://de.wikipedia.org/wiki/Wienerprozess#Gau%C3%9Fscher_Random_Walk
    """
    sqrt_dt = np.sqrt(dt)
    action = action_space.sample()
    rng = np.random.default_rng()
    cnt = 0

    while seq_len is None or cnt < seq_len:
        cnt += 1
        yield action
        action = (action + sqrt_dt * rng.standard_normal(size=action_space.shape)
                  ).clip(action_space.low, action_space.high, dtype=action_space.dtype)


def rollout(index):
    race_env = gym.make(ENV)
    fps = race_env.metadata['video.frames_per_second']
    # observation_space = race_env.observation_space
    dir = Path('observations') / f'thread_{index}'
    dir.mkdir(exist_ok=True)
    with Display(visible=False, size=(1400, 900)):
        for rollout in range(ROLLOUTS_PER_CORE):
            race_env.reset()
            total_reward = 0.0
            step = 0

            # vielleicht: rollout_observations = np.empty((1000,) + observation_space.shape)
            # und am Ende: observations=rollout_observations[:step]
            rollout_observations = []
            rollout_rewards = []
            rollout_terminals = []
            rollout_actions = []

            for action in gauss_rand_walk(race_env.action_space, 1. / fps):
                s, r, done, info = race_env.step(action)

                # append() vs += []
                # https://stackoverflow.com/a/725882
                rollout_observations.append(s)
                rollout_rewards.append(r)
                rollout_terminals.append(done)
                rollout_actions.append(action)

                total_reward += r
                step += 1
                if done or step == 1000:
                    print(f'shape: {np.array(rollout_observations).shape}')
                    np.savez_compressed(dir / f'rollout_{rollout}',
                                        observations=np.array(
                                            rollout_observations),
                                        rewards=np.array(rollout_rewards),
                                        actions=np.array(rollout_actions),
                                        terminals=np.array(rollout_terminals))
                    break
        race_env.close()


# if __name__ == "__main__":
#     race_env = gym.make(ENV)
#     fps = race_env.metadata['video.frames_per_second']
#     render_modes = race_env.metadata['render.modes']
#     print(f'FPS: {fps}')
#     print(f'Render modes: {render_modes}')
#     print(f'Available threads: {AVAILABLE_CORES}')
#     print(f'Rollouts per thread: {ROLLOUTS_PER_CORE}')

#     # race_env.render()
#     # race_env.viewer.window.on_key_press = key_press
#     # race_env.viewer.window.on_key_release = key_release

#     with Pool(AVAILABLE_CORES) as pool:
#         pool.map(rollout, range(AVAILABLE_CORES))
