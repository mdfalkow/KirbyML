import logging
import os
import shutil

import retro

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROM_PATH = f'{SCRIPT_DIR}/ROMs'
CUSTOM_INTEGRATION_PATH = f'{SCRIPT_DIR}/custom_integrations'
RECORDINGS_PATH = f'{SCRIPT_DIR}/recordings'


def setup():
    logging.info('Setting up ROM')

    if not os.path.exists(f'{ROM_PATH}/rom.gb'):
        raise RuntimeError('Error: rom file does not exist')

    if not os.path.exists(f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy/rom.gb'):
        shutil.copy2(f'{ROM_PATH}/rom.gb',
                     f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy')
    logging.info('Adding Kirby\'s Dream Land to Retro integrations...')
    retro.data.Integrations.add_custom_path(CUSTOM_INTEGRATION_PATH)
    if 'KirbysDreamLand-GameBoy' in retro.data.list_games(inttype=retro.data.Integrations.ALL):
        logging.info('Integration successful.')
    else:
        raise RuntimeError('Integration failed.')
