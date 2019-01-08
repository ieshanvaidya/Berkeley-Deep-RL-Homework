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
| Ant-v2         | 4796.73            | 4732.91           | 106.47            | 132.16           |
| HalfCheetah-v2 | 4129.50            | 3682.37           | 115.14            | 153.92           |
| Hopper-v2      | 3778.59            | 165.06            | 3.34              | 1.43             |
| Humanoid-v2    | 10408.70           | 251.39            | 49.33             | 36.62            |
| Reacher-v2     | -4.01              | -11.63            | 1.73              | 4.08             |
| Walker2d-v2    | 5448.71            | 828.94            | 294.95            | 659.87           |
