import logging

import retro

from setup import setup, RECORDINGS_PATH

logging.basicConfig(level=logging.INFO)


def main():
    env = retro.make('KirbysDreamLand-GameBoy',
                     inttype=retro.data.Integrations.ALL,
                     record=RECORDINGS_PATH)
    obs = env.reset()
    i = 0
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            logging.info(f'Iteration {i} completed.')
            i += 1
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    setup()
    main()
