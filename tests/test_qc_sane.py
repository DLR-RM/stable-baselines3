import gym
import torch

from stable_baselines3 import qcsane
from stable_baselines3.qc_sane import SpikeActorDeepCritic

if __name__ == "__main__":
    import argparse
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v3")
    parser.add_argument("--encoder_pop_dim", type=int, default=10)
    parser.add_argument("--decoder_pop_dim", type=int, default=10)
    parser.add_argument("--encoder_var", type=float, default=0.15)
    parser.add_argument("--start_model_idx", type=int, default=0)
    parser.add_argument("--num_model", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_actor", type=int, default=3)
    parser.add_argument("--num_critic", type=int, default=2)
    parser.add_argument("--quantiles", type=int, nargs="+", default=[0.1, 0.5, 0.9])
    args = parser.parse_args()

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    AC_KWARGS = dict(
        hidden_sizes=[256, 256],
        encoder_pop_dim=args.encoder_pop_dim,
        decoder_pop_dim=args.decoder_pop_dim,
        mean_range=(-3, 3),
        std=math.sqrt(args.encoder_var),
        spike_ts=5,
        device=torch.device("cpu"),
        num_actor=args.num_actor,
        num_critic=args.num_critic,
        quantiles=args.quantiles,
    )
    COMMENT = (
        "sac-popsan-"
        + args.env
        + "-encoder-dim-"
        + str(AC_KWARGS["encoder_pop_dim"])
        + "-decoder-dim-"
        + str(AC_KWARGS["decoder_pop_dim"])
    )
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = 100
        qcsane(
            lambda: gym.make(args.env),
            actor_critic=SpikeActorDeepCritic,
            ac_kwargs=AC_KWARGS,
            popsan_lr=1e-4,
            gamma=0.99,
            batch_size=args.batch_size,
            seed=seed,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            norm_clip_limit=3.0,
            tb_comment=COMMENT,
            model_idx=num,
        )
        print("###########", num, "_modelcompleted")
