import argparse
import gym
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from stable_baselines3.common.multi_input_envs import (
    SimpleMultiObsEnv,
    NineRoomMultiObsEnv,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the multi_input_tests script")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=3000,
        help="Number of timesteps to train for (default: 3000)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=10,
        help="Number of environments to use (default: 10)",
    )
    parser.add_argument(
        "--frame_stacks",
        type=int,
        default=1,
        help="Number of stacked frames to use (default: 4)",
    )
    parser.add_argument(
        "--room9",
        action="store_true",
        help="If true, uses more complex 9 room environment",
    )
    args = parser.parse_args()

    ENV_CLS = NineRoomMultiObsEnv if args.room9 else SimpleMultiObsEnv

    make_env = lambda: ENV_CLS(random_start=True)

    env = DummyVecEnv([make_env for i in range(args.num_envs)])
    if args.frame_stacks > 1:
        env = VecFrameStack(env, n_stack=args.frame_stacks)

    model = PPO(MultiInputActorCriticPolicy, env)

    model.learn(args.timesteps)
    env.close()
    print("Done training, starting testing")

    make_env = lambda: ENV_CLS(random_start=False)
    test_env = DummyVecEnv([make_env])
    if args.frame_stacks > 1:
        test_env = VecFrameStack(test_env, n_stack=args.frame_stacks)

    obs = test_env.reset()
    num_episodes = 1
    trajectories = [[]]
    i_step, i_episode = 0, 0
    while i_episode < num_episodes:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        trajectories[-1].append((test_env.get_attr("state")[0], action[0]))

        i_step += 1

        if done[0]:
            if info[0]["got_to_end"]:
                print(f"Episode {i_episode} : Got to end in {i_step} steps")
            else:
                print(f"Episode {i_episode} : Did not get to end")
            obs = test_env.reset()
            i_step = 0
            trajectories.append([])
            i_episode += 1

    test_env.close()
