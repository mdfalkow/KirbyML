import os
import shutil
from datetime import datetime

import numpy as np
import retro
from tqdm import tqdm

# from setup import setup
from scripts.DQNAgent import DQNAgent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_INTEGRATION_PATH = f'{SCRIPT_DIR}/custom_integrations'


def parse_args():
    """Parse arguments"""
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Model settings', allow_abbrev=False)

    # Path settings
    parser.add_argument('ROM_PATH',
                        help='Path to Kirby\'s Dream Land ROM file')
    parser.add_argument('--model-path', type=str, default='',
                        help='Load a previous model as a starting point')

    # Training result settings
    parser.add_argument('--show-emulation', action='store_true',
                        help='Show emulation')
    parser.add_argument('--stat-aggregation-frequency', type=int, default=50,
                        help='How frequently the model should store statistics (measured in episodes)')
    parser.add_argument('--model-prefix', type=str, default='2x32',
                        help='Name of the model (for storing progress)')

    # Q-learning parameters
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discounting ratio for Q-learning')

    # Training settings
    parser.add_argument('--episodes', type=int, default=5_000,
                        help='Number of episodes')
    parser.add_argument('--rms-max', type=int, default=10_000,
                        help='Maximum number of previous steps to keep for model training')
    parser.add_argument('--rms-min', type=int, default=500,
                        help='Minimum number of previous steps to keep for model training')
    parser.add_argument('--minibatch-size', type=int, default=16,
                        help='How many steps (samples) to use for training')
    parser.add_argument('--target-update-frequency', type=int, default='5',
                        help='How frequently the target model should be updated (measured in # of episodes)')
    parser.add_argument('--min-reward', type=int, default=-200,
                        help='Minimum reward value')

    # Exploration settings
    parser.add_argument('--initial-epsilon', type=float,
                        default=1, help='Initial epsilon value')
    parser.add_argument('--epsilon-decay', type=float,
                        default=0.99975, help='Epsilon decay rate')
    parser.add_argument('--min_epsilon', type=float, default=0.001,
                        help='Minimum epsilon value (triggers epsilon reset)')

    return parser.parse_args()


def integrate_rom(rom_path):
    """Setup the ROM integration. Required for setting up the OpenAI Gym environment"""
    print('Setting up ROM')
    if not os.path.exists(rom_path):
        raise RuntimeError('Error: rom file does not exist')

    elif not os.path.exists(f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy/rom.gb'):
        shutil.copy2(rom_path, f'{CUSTOM_INTEGRATION_PATH}/KirbysDreamLand-GameBoy/')
    print('Adding Kirby\'s Dream Land to Retro integrations...')
    retro.data.Integrations.add_custom_path(CUSTOM_INTEGRATION_PATH)
    if 'KirbysDreamLand-GameBoy' in retro.data.list_games(inttype=retro.data.Integrations.ALL):
        print('Integration successful.')
    else:
        raise RuntimeError('Integration failed.')


def main():
    args = parse_args()
    integrate_rom(args.ROM_PATH)
    env = retro.make('KirbysDreamLand-GameBoy',
                     inttype=retro.data.Integrations.ALL,
                     use_restricted_actions=retro.Actions.DISCRETE)

    agent = DQNAgent(env,
                     args.model_path,
                     args.discount,
                     args.rms_min,
                     args.rms_max,
                     args.minibatch_size,
                     args.target_update_frequency,
                     args.model_prefix)

    # For exploration
    epsilon = args.initial_epsilon
    # For stats
    ep_rewards = [args.min_reward]

    # Iterate over episodes
    for episode in tqdm(range(1, args.episodes + 1),
                        ascii=True,
                        unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()[:, :, 0]

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            new_state = new_state[:, :, 0]
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if args.show_emulation:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state,
                                        action,
                                        reward,
                                        new_state,
                                        done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats
        # (every given number of episodes)
        ep_rewards.append(episode_reward)
        if episode % args.stat_agregation_freq == 0 or episode == 1:
            average_reward = (sum(ep_rewards[-args.stat_agregation_freq:])
                              / len(ep_rewards[-args.stat_agregation_freq:]))
            min_reward = min(ep_rewards[-args.stat_agregation_freq:])
            max_reward = max(ep_rewards[-args.stat_agregation_freq:])
            agent.tensorboard.update_stats(reward_avg=average_reward,
                                           reward_min=min_reward,
                                           reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= args.min_reward:
                agent.model.save('__'.join([
                    f'models/{args.model_prefix}',
                    f'{max_reward:_>7.2f}max',
                    f'{average_reward:_>7.25}avg',
                    f'{min_reward:_>7.2f}min',
                    f'{datetime.now().isoformat()}.model'
                ]))
        # Decay epsilon
        if epsilon > args.min_epsilon:
            epsilon *= args.epsilon_decay
            epsilon = max(args.min_epsilon, epsilon)


if __name__ == "__main__":
    main()
