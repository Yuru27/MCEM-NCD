 # Multi-Agent Cross-Entropy Method with Monotonic Nonlinear Critic Decomposition (MCEM-NCD)

This repository provides the official PyTorch implementation for the paper: "**Multi-Agent Cross-Entropy Method with Monotonic Nonlinear Critic Decomposition**." Our implementation is built upon the [PyMARL](https://github.com/oxwhirl/pymarl) framework. It is evaluated on two benchmarks: the [SMAC](https://github.com/oxwhirl/smac) for discrete action spaces and the [Continuous Predator-Prey](https://github.com/oxwhirl/facmac) for continuous action spaces. For further details on the underlying framework and evaluation environments, please refer to their respective repositories.


## Setup

Set up the working environment:

```shell
# require Anaconda 3 or Miniconda 3
conda create -n mcemncd python=3.8 -y
conda activate mcemncd

bash install_dependencies.sh
```

Set up the StarCraft II and SMAC:

```shell
bash install_sc2.sh
```


## Run an experiment

This command is used to train the MCEM-NCD for discrete action on the `5m_vs_6m` scenario of SMAC. To conduct experiments on alternative scenarios, the 'map_name' parameter must be adjusted accordingly.

```shell
python3 src/discrete/main.py --config=mcemncd --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2050000
```

This command is used to train the MCEM-NCD model in the `continuous_pred_prey_3a` scenario of Continuous Predator-Prey. To conduct experiments on alternative scenarios, the `scenario_name` parameter must be adjusted accordingly.

```shell
python3 src/continuous/main.py --config=mcemncd_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a t_max=2050000
```

All results will be saved in the `results` folder.