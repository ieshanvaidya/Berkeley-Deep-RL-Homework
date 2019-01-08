# Berkeley-Deep-RL-Homework
Implementations of Reinforcement Learning algorithms from the Berkeley Deep Reinforcement Learning Course (http://rail.eecs.berkeley.edu/deeprlcourse/)

### HW1
Model Architecture :
- 2 dense layers of 32 neurons with ReLU activation
- Adam Optimizer with default learning rate 0.001
- 20 epochs and batch_size of 100
- 20 rollouts of expert

Behavior Cloning Results :

| Expert         | Mean Expert Reward | Mean Model Reward | Std Expert Reward | Std Model Reward |
|----------------|--------------------|-------------------|-------------------|------------------|
| Ant-v2         | 4746.57            | 4729.87           | 612.66            | 97.51            |
| HalfCheetah-v2 | 4139.85            | 4026.78           | 73.55             | 87.58            |
| Hopper-v2      | 3778.58            | 2155.87           | 3.93              | 598.73           |
| Humanoid-v2    | 10411.17           | 380.00            | 51.17             | 92.55            |
| Reacher-v2     | -3.74              | -7.56             | 1.61              | 3.11             |
| Walker2d-v2    | 5454.00            | 4855.07           | 485.88            | 498.47           |

DAgger is performed on Hopper-v2 and Humanoid-v2 since they perform poorly compared to the expert.

DAgger Results :
