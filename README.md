# Berkeley-Deep-RL-Homework
Implementations of Reinforcement Learning algorithms from the Berkeley Deep Reinforcement Learning Course (http://rail.eecs.berkeley.edu/deeprlcourse/)

### HW1
Model Architecture :
- 2 dense layers of 64 neurons with Tanh activation
- Learning rate (0.001) and epochs (30) tuned by grid search with batch size of 128
- 30 rollouts of expert

Example hyperparameter grid :

Learning rate = 0.001 and epochs = 30 performed well for all models except Reacher-v2

![alt]https://github.com/ieshanvaidya/Berkeley-Deep-RL-Homework/blob/master/hw1/figures/Ant-v2_hyperparams_tuning.png

![alt]https://github.com/ieshanvaidya/Berkeley-Deep-RL-Homework/blob/master/hw1/figures/Reacher-v2_hyperparams_tuning.png

Behavior Cloning Results :

| Expert         | Mean Expert Reward | Mean Model Reward | Std Dev Expert Reward | Std Dev Model Reward |
|----------------|--------------------|-------------------|-----------------------|----------------------|
| Ant-v2         | 4816.54            | 4768.42           | 103.12                | 98.73                |
| HalfCheetah-v2 | 4116.33            | 4085.92           | 83.12                 | 71.80                |
| Hopper-v2      | 3778.54            | 3778.15           | 3.67                  | 4.71                 |
| Humanoid-v2    | 10410.32           | 9342.13           | 53.65                 | 2844.41              |
| Reacher-v2     | -3.71              | -7.98             | 1.79                  | 3.91                 |
| Walker2d-v2    | 5536.69            | 5507.58           | 54.58                 | 116.72               |

DAgger is performed on Reacher-v2 since it performs poorly compared to the expert.

DAgger Results :
