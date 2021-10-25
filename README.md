This repo includes baseline learning code for all the PettingZoo [Butterfly](https://www.pettingzoo.ml/butterfly) environments (except Prison, which is a toy debugging environment) based on parameter shared PPO via [Stable Baslines 3](https://github.com/DLR-RM/stable-baselines3) and [SuperSuit](https://github.com/Farama-Foundation/SuperSuit). 

To train all four Butterfly environments for five runs each:

```sh train_all.sh```

To train individual environments:

```python train.py --env-name=pistonball_v4 --n-runs=5 --n-evaluations=100 --timesteps=2000000  --num-cpus=8 --num-eval-cpus=4 --num-vec-envs=4```

The above example trains pistonball_v4 for 5 runs, with 2000000 timesteps and 100 evaluations per run, on 8 cpus, with four more cpus for the evaluations, and four parallel environments per cpu, and saves the results of the evaluations to data/ENV_NAME/run_x.

To modify other hyperparameters e.g. learning rate, activation function, network size: modify config/ENV_NAME.json

To plot learning and evaluations in an environment from the data folder:

```python plot.py --env-name=pistonball_v4 --n-runs=10```

To plot learning curves for all four environments:

```sh plot_all.sh```
