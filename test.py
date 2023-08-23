import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from src.metrics import MetricLogger
from src.agent import Mario
from src.wrappers import ResizeObservation, SkipFrame
import argparse

def main(args): 

    env = gym_super_mario_bros.make(f'SuperMarioBros-{args.world}-{args.stage}-v0')

    env = JoypadSpace(
        env,
        [['right'],
        ['right', 'A']]
    )

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    save_dir = Path(args.checkpoints_dir) / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = Path(args.checkpoints)
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
    mario.exploration_rate = mario.exploration_rate_min

    logger = MetricLogger(save_dir)

    episodes = args.episodes

    for e in range(episodes):

        state = env.reset()

        while True:

            image = env.render(mode="rgb_array")

            action = mario.act(state)

            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            logger.log_step(reward, None, None)

            state = next_state

            if done or info['flag_get']:
                break
            
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, default=1, help="The world in which agent will play, a number between 1 to 8")
    parser.add_argument('--stage', type=int, default=1, help="The stage in which agent will play, a number between 1 to 4")
    parser.add_argument('--checkpoints', type=str, default=None, help="path of the checkpoints file of the trained model")
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help="Directory where the testing checkpoints will be saved")
    parser.add_argument('--episodes', type=int, default=100, help="Number of times you want the trained agent to play the game Mario Game")
    args = parser.parse_args()

    assert args.checkpoints != None, "Please provide the checkpoints file of the trianed model"
    assert args.world >= 1 and args.world <= 8, "Please select a correct world"
    assert args.stage >= 1 and args.stage <= 4, "Please select a correct stage"

    main(args)