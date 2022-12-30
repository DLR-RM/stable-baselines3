# DRL with Population Coded Spiking Neural Network

This package is the PyTorch implementation of the **Pop**ulation-coded **S**piking **A**ctor **N**etwork (**PopSAN**) that integrates with both on-policy (PPO) and off-policy (DDPG, TD3, SAC) DRL algorithms for learning optimal and energy-efficient continuous control policies.

The paper has been accepted at CoRL 2020.
The arXiv preprint is available [here](https://arxiv.org/abs/2010.09635).

**New**: We have created a new GitHub repo to demonstrate the online runtime interaction with Loihi. If you are interested in using Loihi for real-time robot control, please [check it out](https://github.com/michaelgzt/loihi-control-loop-demo).

## Citation ##

Guangzhi Tang, Neelesh Kumar, Raymond Yoo, and Konstantinos P. Michmizos. 
"Deep Reinforcement Learning with Population-Coded Spiking Neural Network for Continuous Control." *4th Conference on Robot Learning (CoRL 2020)*, Cambridge MA. USA.

```bibtex
@inproceedings{tang2020deep,
  title={Deep Reinforcement Learning with Population-Coded Spiking Neural Network for Continuous Control},
  author={Tang, Guangzhi and Kumar, Neelesh and Yoo, Raymond and Michmizos, Konstantinos P},
  booktitle={4th Conference on Robot Learning (CoRL 2020)},
  pages={1--10},
  year={2020}
}
```

## Software Installation ##

* Ubuntu 16.04
* Python 3.5.2
* MuJoCo 2.0
* OpenAI GYM 0.15.3 (with mujoco_py 2.0.2.5)
* PyTorch 1.2 (with CUDA 10.0 and tensorboard 2.1)
* NxSDK 0.9

A CUDA enabled GPU is not required but preferred for training. 
The results in the paper are generated from models trained using both Nvidia Tesla K40c and Nvidia GeForce RTX 2080Ti.

Intel's neuromorphic library NxSDK is only required for SNN deployment on the Loihi neuromorphic chip. 
If you are interested in deploying the trained SNN on Loihi, please contact the [Intel Neuromorphic Lab](https://www.intel.com/content/www/us/en/research/neuromorphic-community.html).

We have provided the `requirements.txt` for the python environment without NxSDK. In addition, we recommend setting up the environment using [virtualenv](https://pypi.org/project/virtualenv/).

## Example Usage ##

#### 1. Training PopSAN ####

To train PopSAN using TD3 algorithm, execute the following commands:

```bash
cd <Dir>/<Project Name>/popsan_drl/popsan_td3
python td3_cuda_norm.py --env HalfCheetah-v3
```

This will automatically train 1 million steps and save the trained models. The steps to train DDPG, SAC, and PPO are the same as above.

#### 2. Deploy the trained PopSAN on Loihi ####

To evaluate PopSAN realization on Loihi, execute the following commands to start testing:

```bash
cd <Dir>/<Project Name>/loihi_realization
python test_loihi.py
```

This will test the 10 trained models on Loihi. To run the code correctly, `data_dir` value in the script needs to be changed to the folder that stores all trained models.

### Acknowledgment ###

This work is supported by Intel's Neuromorphic Research Community Grant Award. Part of our code, including DRL training and PPO multiprocessing environments, were built upon [OpenAI Spinning Up](https://github.com/openai/spinningup), [OpenAI Baselines](https://github.com/openai/baselines), and [Stable Baselines](https://github.com/hill-a/stable-baselines). We would like to thank their contribution to the community.
