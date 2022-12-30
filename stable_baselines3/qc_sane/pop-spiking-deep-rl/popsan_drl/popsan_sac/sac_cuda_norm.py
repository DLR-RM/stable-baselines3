from copy import deepcopy
import itertools
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import os
import pickle
import sys
import csv
sys.path.append("../../")
from popsan_drl.popsan_sac.replay_buffer_norm import ReplayBuffer
from popsan_drl.popsan_sac.popsan import SquashedGaussianPopSpikeActor
from popsan_drl.popsan_sac.core_cuda import MLPQFunction


class SpikeActorDeepCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, device,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        print("%%%%%%%%%%%%%%%%%",device)

        # build policy and value functions
        self.popsan = SquashedGaussianPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                                    mean_range, std, spike_ts, act_limit, device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size, deterministic=False):
        with torch.no_grad():
            a, _ = self.popsan(obs, batch_size, deterministic, False)
            a = a.to('cpu')
            return a.numpy()



def spike_sac(env_fn, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), seed=100,
              steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, popsan_lr=1e-4, q_lr=3e-4, alpha=0.2, batch_size=256, start_steps=10000,
              update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
              save_freq=5, norm_clip_limit=3, norm_update=50, tb_comment='', model_idx=0, use_cuda=False):
    """
    Spike Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``popsan`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``popsan`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``popsan`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        popsan_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        norm_clip_limit (float): Clip limit for normalize observation

        norm_update (int): Number of steps to update running mean and var in memory

        tb_comment (str): Comment for tensorboard writer

        model_idx (int): Index of training model

        use_cuda (bool): If true use cuda for computation
    """
    # Set device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    env.seed(seed)
    test_env.seed(seed)
    env.action_space.seed(seed)
    test_env.action_space.seed(seed)
    print("#################",device)
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # List of parameters for PopSAN parameters (save this for convenience)
    popsan_params = itertools.chain(ac.popsan.encoder.parameters(),
                                    ac.popsan.snn.parameters(),
                                    ac.popsan.decoder.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)

    # Set up function for computing Spike-SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.popsan(o2, batch_size)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.to('cpu').detach().numpy(),
                      Q2Vals=q2.to('cpu').detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.popsan(o, batch_size)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.to('cpu').detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    popsan_mean_optimizer = Adam(popsan_params, lr=popsan_lr)
    pi_std_optimizer = Adam(ac.popsan.log_std_network.parameters(), lr=q_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        popsan_mean_optimizer.zero_grad()
        pi_std_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        popsan_mean_optimizer.step()
        pi_std_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1,
                      deterministic)

    def test_agent():
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o), True))
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes


    ###
    # add tensorboard support and save rewards
    # Also create dir for saving parameters
    ###
    writer = SummaryWriter(comment="_" + tb_comment + "_" + str(model_idx))
    save_test_reward = []
    save_test_reward_steps = []
    try:
        os.mkdir("./params")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./params/spike-sac_" + tb_comment
    try:
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    with open(model_dir + '/' + "data" + str(model_idx) + '_traincsv.csv', 'w', newline='\n') as csvfile:
        fieldnames = ["t","a","o2", "r", "d", "_"]
        writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer2.writeheader()
        writer2.writerow({'t':-1, 'a':"", 'o2':o, 'r':"", 'd':"", '_':""})
    pickle.dump([-1,"",o, "","" ,"" ],
                            open(model_dir + '/' + "data" + str(model_idx) + '_train.p', "wb+"))
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(replay_buffer.normalize_obs(o))
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        with open(model_dir + '/' + "data" + str(model_idx) + '_traincsv.csv', 'a', newline='\n') as csvfile:
            fieldnames = ["t","a","o2", "r", "d", "_"]
            writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            writer2.writerow({'t':t, 'a':a,'o2':o2, 'r':r, 'd':d, '_':_})
        pickle.dump([t,a,o2, r, d, _],
                            open(model_dir + '/' + "data" + str(model_idx) + '_train.p', "wb+"))

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            writer.add_scalar(tb_comment + '/Train-Reward', ep_ret, t + 1)
            o, ep_ret, ep_len = env.reset(), 0, 0
            with open(model_dir + '/' + "data" + str(model_idx) + '_traincsv.csv', 'a', newline='\n') as csvfile:
                fieldnames = ["t","a","o2", "r", "d", "_"]
                writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
                #writer.writeheader()
                writer2.writerow({'t':t,'a':"", 'o2':o2, 'r':r, 'd':d, '_':_})
            pickle.dump([t,"",o, "","" ,"" ],
                            open(model_dir + '/' + "data" + str(model_idx) + '_train.p', "wb+"))

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                ac.popsan.to('cpu')
                torch.save(ac.popsan.state_dict(),
                           model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')
                print("Learned Mean for encoder population: ")
                print(ac.popsan.encoder.mean.data)
                print("Learned STD for encoder population: ")
                print(ac.popsan.encoder.std.data)
                ac.popsan.to(device)
                pickle.dump([replay_buffer.mean, replay_buffer.var],
                            open(model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '_mean_var.p', "wb+"))
                #print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')
                print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')

            # Test the performance of the deterministic version of the agent.
            test_mean_reward = test_agent()
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            writer.add_scalar(tb_comment + '/Test-Mean-Reward', test_mean_reward, t + 1)
            print("Model: ", model_idx, " Steps: ", t + 1, " Mean Reward: ", test_mean_reward)

    # Save Test Reward List
    pickle.dump([save_test_reward, save_test_reward_steps],
                open(model_dir + '/' + "model" + str(model_idx) + "_test_rewards.p", "wb+"))

if __name__ == '__main__':
    import math
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--start_model_idx', type=int, default=0)
    parser.add_argument('--num_model', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=torch.device('cpu'))
    COMMENT = "sac-popsan-" + args.env + "-encoder-dim-" + str(AC_KWARGS['encoder_pop_dim']) + \
              "-decoder-dim-" + str(AC_KWARGS['decoder_pop_dim'])
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = 100
        spike_sac(lambda: gym.make(args.env), actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
                  popsan_lr=1e-4, gamma=0.99, seed=seed, epochs=args.epochs,
                  norm_clip_limit=3.0, tb_comment=COMMENT, model_idx=num)
        print("###########",num,"_modelcompleted")

