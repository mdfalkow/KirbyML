import os
import shutil

import numpy as np
import retro
from tensorflow.keras.models import load_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_INTEGRATION_PATH = f'{SCRIPT_DIR}/custom_integrations'


def parse_args():
    """Parse arguments"""
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Run model (no training)', allow_abbrev=False)

    parser.add_argument('ROM_PATH',
                        help='Path to Kirby\'s Dream Land ROM file')
    parser.add_argument('MODEL_PATH',
                        help='Load a previous model as a starting point')
    parser.add_argument('--show-emulation', action='store_true',
                        help='Show emulation')
    return parser.parse_args()


def integrate_rom(rom_path):
    """Setup the ROM integration. Required for setting up the OpenAI Gym environment"""
    print('Setting up ROM')
    if not os.path.exists(rom_path):
        raise RuntimeError('Error: rom file does not exist')

    elif not os.path.exists(f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy/rom.gb'):
        shutil.copy2(
            rom_path, f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy/')
    print('Adding Kirby\'s Dream Land to Retro integrations...')
    retro.data.Integrations.add_custom_path(CUSTOM_INTEGRATION_PATH)
    if 'KirbysDreamLand-GameBoy' in retro.data.list_games(inttype=retro.data.Integrations.ALL):
        print('Integration successful.')
    else:
        raise RuntimeError('Integration failed.')


def main():
    args = parse_args()
    env = retro.make('KirbysDreamLand-GameBoy',
                     inttype=retro.data.Integrations.ALL,
                     use_restricted_actions=retro.Actions.DISCRETE)

    model = load_model(args.MODEL_PATH)

    def get_action(state):
        return np.argmax(model.predict(np.array(state)
                                       .reshape(-1, *state.shape)/255)[0])

    episode = 0
    while True:
        done = False
        current_state = env.reset()[:, :, 0]
        cumulative_rew = 0
        while not done:
            action = get_action(current_state)
            new_state, reward, _, info = env.step(action)
            cumulative_rew += reward
            env.render()
            current_state = new_state[:, :, 0]
            if info['lives'] == 0:
                done = True
        print(f'Episode {episode}: total reward = {cumulative_rew}')
        episode += 1


if __name__ == "__main__":
    main()
