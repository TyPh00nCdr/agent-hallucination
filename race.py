import numpy as np
import gym

ENV = 'CarRacing-v0'


def key_press(k, mod):
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


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])
    race_env = gym.make(ENV)
    fps = race_env.metadata['video.frames_per_second']
    render_modes = race_env.metadata['render.modes']
    print(f'FPS: {fps}')
    print(f'Render modes: {render_modes}')

    race_env.render()
    # race_env.viewer.window.on_key_press = key_press
    # race_env.viewer.window.on_key_release = key_release

    from pyglet.window import key
    isopen = True
    while isopen:
        race_env.reset()
        total_reward = 0.0
        restart = False
        step = 0
        for action in gauss_rand_walk(race_env.action_space, 1. / fps):
            a = action
            s, r, done, info = race_env.step(a)
            total_reward += r
            isopen = race_env.render()
            step += 1
            if done or restart or not isopen or step == 500:
                break
    race_env.close()
