import csv
import itertools
import os
import pickle
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")
from stable_baselines3.qc_sane.core_cuda import MLPQFunction_quantile
from stable_baselines3.qc_sane.popsan import SquashedGaussianPopSpikeActor
from stable_baselines3.qc_sane.replay_buffer_norm import ReplayBuffer


class SpikeActorDeepCritic(nn.Module):
    def __init__(
        self,
        test_env,
        observation_space,
        action_space,
        encoder_pop_dim,
        decoder_pop_dim,
        mean_range,
        std,
        spike_ts,
        device,
        num_actor,
        num_critic,
        quantiles,
        hidden_sizes=(256, 256),
        activation=nn.SELU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        print("%%%%%%%%%%%%%%%%%", device)
        # self.quantiles = [.1, .5, .9]
        # self.quantiles = [.1, .9]
        self.quantiles = quantiles  # [.1,.9,.2,.8]

        # build policy and value functions
        # self.popsan1 = SquashedGaussianPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
        #                                             mean_range, std, spike_ts, act_limit, device)
        for actor_index in range(num_actor):
            print("Creating Actor " + str(actor_index + 1))
            popsan_params_actor_index = """SquashedGaussianPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, 
decoder_pop_dim, hidden_sizes, mean_range, std, spike_ts, act_limit, device)"""

            exec("self.popsan%d = %s" % (actor_index + 1, popsan_params_actor_index))

        self.popsan = self.popsan1
        for critic_idx in range(num_critic):
            print("Creating Critic " + str(critic_idx + 1))
            exec(
                "self.q%d = %s"
                % (
                    critic_idx + 1,
                    """MLPQFunction_quantile(obs_dim, act_dim, hidden_sizes, activation,self.quantiles)""",
                )
            )
        # self.q1 = MLPQFunction_quantile(obs_dim, act_dim, hidden_sizes, activation,self.quantiles)
        # self.q2 = MLPQFunction_quantile(obs_dim, act_dim, hidden_sizes, activation,self.quantiles)

    def act(self, popsan, obs, batch_size, deterministic=False):
        with torch.no_grad():
            a, _ = popsan(obs, batch_size, deterministic, False)
            a = a.to("cpu")
            return a.numpy()


def qcsane(
    env_fn,
    actor_critic=SpikeActorDeepCritic,
    ac_kwargs=dict(),
    seed=100,
    steps_per_epoch=10000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    popsan_lr=1e-4,
    q_lr=3e-4,
    alpha=0.2,
    batch_size=256,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    save_freq=5,
    norm_clip_limit=3,
    norm_update=50,
    tb_comment="",
    model_idx=0,
    use_cuda=False,
):

    """
    QC_SANE: Authors:(Surbhi Gupta, et.al.)


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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    env.seed(seed)
    test_env.seed(seed)
    env.action_space.seed(seed)
    test_env.action_space.seed(seed)
    print("#################", device)
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    global ac, ac_targ
    ac = actor_critic(test_env, env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # List of parameters for PopSAN parameters (save this for convenience)
    print("kwargs", ac_kwargs)
    for actor_index in range(ac_kwargs["num_actor"]):
        popsan_params_actor_index = """itertools.chain(eval("ac.popsan%d.encoder.parameters()"%(actor_index+1)),
                                        eval("ac.popsan%d.snn.parameters()"%(actor_index+1)),
                                        eval("ac.popsan%d.decoder.parameters()"%(actor_index+1)))"""
        exec("popsan_params%d = %s" % (actor_index + 1, popsan_params_actor_index))

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        clip_limit=norm_clip_limit,
        norm_update_every=norm_update,
    )

    # Set up function for computing Spike-SAC Q-losses
    quantiles = ac.quantiles

    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        for critic_idx in range(ac_kwargs["num_critic"]):
            exec(
                "q%d = %s"
                % (
                    critic_idx + 1,
                    """torch.mean( eval("ac.q%d(o,a)"%(critic_idx+1)),-1)""",
                )
            )
        # q1 = torch.mean(ac.q1(o,a),-1)
        # q2 = torch.mean(ac.q2(o,a),-1)
        # print("$$$$",torch.mean(q1,-1).shape)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.popsan(o2, batch_size)

            # Target Q-values
            q_pi_targ = []
            for critic_idx in range(ac_kwargs["num_critic"]):
                q_pi_targ.append(torch.mean(eval("ac_targ.q%d(o2, a2)" % (critic_idx + 1)), -1))
            # q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            q_pi_targ = torch.min(torch.stack(q_pi_targ, axis=-1), axis=-1).values

            # r=torch.stack([r,r,r],dim=1)
            # d=torch.stack([d,d,d],dim=1)
            # logp_a2=torch.stack([logp_a2,logp_a2,logp_a2],dim=1)
            # print("SHApe of q1",r.shape,gamma,d.shape,q_pi_targ.shape,alpha,logp_a2.shape)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
            # print("SHApe of q1",backup.shape)
        q_info = dict()  # Useful info for logging
        loss_q = 0
        for critic_idx in range(ac_kwargs["num_critic"]):
            exec("qf%d_losses = %s" % (critic_idx + 1, []))
            for _i, quantile in enumerate(quantiles):
                # error = backup - eval("q%d"%(critic_idx+1))
                loss = """torch.maximum(quantile*(backup - eval("q%d"%(critic_idx+1))), 
(quantile-1)*(backup - eval("q%d"%(critic_idx+1))))"""
                # print("##$$$$",loss)
                exec("qf%d_losses.append(%s)" % (critic_idx + 1, loss))
            loss_q_idx = """torch.mean(torch.sum(torch.stack(eval("qf%d_losses" % (critic_idx + 1)),dim=-1),-1))"""
            exec("loss_q%d = %s" % (critic_idx + 1, loss_q_idx))
            loss_q += eval("loss_q%d" % (critic_idx + 1))    # loss_q = loss_q1 + loss_q2
            exec(
                "q_info['Q%dVals'] = %s"
                % (
                    critic_idx + 1,
                    """eval("q%d.to('cpu').detach().numpy()"%(critic_idx+1))""",
                )
            )
            # print("##$$$$",eval("q_info['Q%dVals']" % (critic_idx + 1)))
            # q_info = dict(Q1Vals=q1.to('cpu').detach().numpy(),
            #             Q2Vals=q2.to('cpu').detach().numpy())
        return loss_q, q_info

    # Set up function for computing SAC pi loss
    global all_Actr

    def all_Actr(popsan, o, batch_size):

        pi, logp_pi = popsan(o, batch_size)
        q_pi = []
        for critic_idx in range(ac_kwargs["num_critic"]):
            q_pi.append(torch.mean(eval("ac.q%d(o, pi)" % (critic_idx + 1)), -1))
            # print(q_pi[critic_idx].shape)
        # q_pi = torch.min(q1_pi_targ, q2_pi_targ)
        q_pi = torch.min(torch.stack(q_pi, axis=-1), axis=-1).values
        # print(q_pi.shape)
        # q1_pi = torch.mean(ac.q1(o, pi),-1)
        # q2_pi = torch.mean(ac.q2(o, pi),-1)
        # q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        # logp_pi=torch.stack([logp_pi,logp_pi,logp_pi],dim=1)
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.to("cpu").detach().numpy())
        return loss_pi, pi_info

    def compute_loss_pi(data):
        o = data["obs"]
        pi_info = []
        loss_pi_all = []
        batch_size
        for actor_index in range(ac_kwargs["num_actor"]):
            # all_Actr(ac.popsan1,o)
            t = eval("ac.popsan%d" % (actor_index + 1))
            # a,b=all_Actr(t ,o)
            # a,b=exec("""global all_Actr(t ,o,batch_size)""")
            # print(batch_size)
            exec(
                "loss_pi%d,inf%d = %s"
                % (
                    actor_index + 1,
                    actor_index + 1,
                    """eval("all_Actr(t ,o, batch_size)")""",
                )
            )
            pi_info.append(eval("inf%d" % (actor_index + 1)))
            loss_pi_all.append(eval("loss_pi%d" % (actor_index + 1)))
        # pi_info=[inf1,inf2,inf3,inf4]
        return loss_pi_all, pi_info  # loss_pi3,

    # Set up optimizers for policy and q-function
    for actor_idx in range(ac_kwargs["num_actor"]):
        exec(
            "globals()['popsan_mean_optimizer%d'] = %s"
            % (
                actor_idx + 1,
                """Adam(eval("popsan_params%d"%(actor_idx + 1)), lr=popsan_lr)""",
            )
        )
        exec(
            "globals()['pi_std_optimizer%d'] = %s"
            % (
                actor_idx + 1,
                """Adam(eval("ac.popsan%d.log_std_network.parameters()"%(actor_idx + 1)), lr=q_lr)""",
            )
        )

    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # print("%%%",eval("popsan_mean_optimizer%d"%(1)))
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        for actor_idx in range(ac_kwargs["num_actor"]):
            eval("popsan_mean_optimizer%d.zero_grad()" % (actor_idx + 1))
            eval("pi_std_optimizer%d.zero_grad()" % (actor_idx + 1))

        loss_pi_all, pi_info = compute_loss_pi(data)  ##loss_pi1,loss_pi2,loss_pi3, pi_info

        for loss_pi_idx in range(len(loss_pi_all)):
            loss_pi_all[loss_pi_idx].backward()
            eval("popsan_mean_optimizer%d.step()" % (loss_pi_idx + 1))
            eval("pi_std_optimizer%d.step()" % (loss_pi_idx + 1))
        # loss_pi2.backward()
        # popsan_mean_optimizer2.step()
        # pi_std_optimizer2.step()

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

    def get_action(popsan, o, deterministic=False):
        return ac.act(
            popsan,
            torch.as_tensor(o, dtype=torch.float32, device=device),
            1,
            deterministic,
        )

    def test_agent(popsan):
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for _j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(popsan, replay_buffer.normalize_obs(o), True))
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

    def updt(par1, par2):
        with torch.no_grad():
            for bst, pop in zip(par1.parameters(), par2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                pop.data.mul_(0.6)
                pop.data.add_((0.4) * bst.data)

    ###
    # add tensorboard support and save rewards
    # Also create dir for saving parameters
    ###
    writer = SummaryWriter(comment="_" + tb_comment + "_" + str(model_idx))
    save_test_reward = []
    save_test_reward_steps = []
    try:
        os.makedirs("./results/params")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./results/params/spike-sac_" + tb_comment
    try:
        os.makedirs(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    with open(model_dir + "/" + "data" + str(model_idx) + "_train.csv", "w", newline="\n") as csvfile:
        fieldnames = ["t", "a", "o2", "r", "d", "_"]

        writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer2.writeheader()
        writer2.writerow({"t": -1, "a": "", "o2": o, "r": "", "d": "", "_": ""})
    pickle.dump(
        [-1, "", o, "", "", ""],
        open(model_dir + "/" + "data" + str(model_idx) + "_train.p", "wb+"),
    )

    with open(
        model_dir + "/" + "data" + str(model_idx) + "_Test-Mean-Reward.csv",
        "a",
        newline="\n",
    ) as csvfile:
        fieldnames = ["t"] + ["Actor_" + str(idx + 1) for idx in range(ac_kwargs["num_actor"])]
        writer3 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer3.writeheader()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(ac.popsan, replay_buffer.normalize_obs(o))
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        with open(model_dir + "/" + "data" + str(model_idx) + "_train.csv", "a", newline="\n") as csvfile:
            fieldnames = ["t", "a", "o2", "r", "d", "_"]
            writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            writer2.writerow({"t": t, "a": a, "o2": o2, "r": r, "d": d, "_": _})
        # pickle.dump([t,a,o2, r, d, _],
        # open(model_dir + '/' + "data" + str(model_idx) + '_train.p', "wb+"))

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            writer.add_scalar(tb_comment + "/Train-Reward", ep_ret, t + 1)
            o, ep_ret, ep_len = env.reset(), 0, 0
            with open(
                model_dir + "/" + "data" + str(model_idx) + "_train.csv",
                "a",
                newline="\n",
            ) as csvfile:
                fieldnames = ["t", "a", "o2", "r", "d", "_"]
                writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # writer.writeheader()
                writer2.writerow({"t": t, "a": "", "o2": o2, "r": r, "d": d, "_": _})
            pickle.dump(
                [t, "", o, "", "", ""],
                open(model_dir + "/" + "data" + str(model_idx) + "_train.p", "wb+"),
            )

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print("##epoch", epoch, " completed")
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                ac.popsan.to("cpu")
                torch.save(
                    ac.popsan.state_dict(),
                    model_dir + "/" + "model" + str(model_idx) + "_e" + str(epoch) + ".pt",
                )
                print("Learned Mean for encoder population: ")
                print(ac.popsan.encoder.mean.data)
                print("Learned STD for encoder population: ")
                print(ac.popsan.encoder.std.data)
                ac.popsan.to(device)
                pickle.dump(
                    [replay_buffer.mean, replay_buffer.var],
                    open(
                        model_dir + "/" + "model" + str(model_idx) + "_e" + str(epoch) + "_mean_var.p",
                        "wb+",
                    ),
                )
                # print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')
                print(
                    "Weights saved in ",
                    model_dir + "/" + "model" + str(model_idx) + "_e" + str(epoch) + ".pt",
                )

            # Test the performance of the deterministic version of the agent.
            # test_mean_reward0 = test_agent(ac.popsan)
            test_mean_reward = []
            for actor_idx in range(ac_kwargs["num_actor"]):
                # exec("test_mean_reward%d = %s"% (actor_idx+1, test_agent(eval("ac.popsan%d"%(actor_idx+1)) ))
                test_mean_reward.append(test_agent(eval("ac.popsan%d" % (actor_idx + 1))))
            test_mean_reward = np.array(test_mean_reward)
            # test_mean_reward=np.array([test_mean_reward1,test_mean_reward2,test_mean_reward3,test_mean_reward4])#,test_mean_reward3])
            idx = np.argmax(test_mean_reward)
            exec("selected_actr = %s" % ("""eval("ac.popsan%d"%(idx+1))"""))
            ac.popsan = eval("selected_actr")
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            with open(
                model_dir + "/" + "data" + str(model_idx) + "_Test-Mean-Reward.csv",
                "a",
                newline="\n",
            ) as csvfile:
                fieldnames = ["t"] + ["Actor_" + str(idx + 1) for idx in range(ac_kwargs["num_actor"])]
                writer3 = csv.DictWriter(csvfile, fieldnames=fieldnames)
                actor_test_performnce = {"t": t + 1}
                for actor_idx in range(ac_kwargs["num_actor"]):
                    actor_test_performnce[fieldnames[actor_idx + 1]] = test_mean_reward[actor_idx]
                writer3.writerow(actor_test_performnce)

            print(
                "Model: ",
                model_idx,
                " Steps: ",
                t + 1,
                " Mean Reward: ",
                test_mean_reward,
                "Selected Actor ",
                idx + 1,
            )

    # Save Test Reward List
    pickle.dump(
        [save_test_reward, save_test_reward_steps],
        open(model_dir + "/" + "model" + str(model_idx) + "_test_rewards.p", "wb+"),
    )
