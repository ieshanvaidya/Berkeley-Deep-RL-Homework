from functions import *
import matplotlib.pyplot as plt

plt.rc('font', family = "serif")
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#experts = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]

def runDAgger(expert, iterations, rollouts, lr = 0.001, epochs = 50, batch_size = 128) :
    expert_data = getExpertData(expert)
    observations, actions, expert_returns = expert_data["observations"], expert_data["actions"], expert_data["returns"]
    actions = np.squeeze(actions)

    retrun(DAgger(expert, observations, actions, iterations, rollouts, lr, epochs, batch_size))
    #obs_dataset[expert], act_dataset[expert], rewards_dataset[expert] = DAgger(expert, observations, actions, iterations, rollouts, lr, epochs, batch_size)

iterations = 30
rollouts = 30

for expert in experts :
    observation_dataset, actions_dataset, rewards_dataset = runDAgger(expert, iterations, rollouts)

    with open("expert_data/{}_dagger_observations.pkl", "wb".format(expert)) as f :
        pickle.dump(observation_dataset)

    with open("expert_data/{}_dagger_actions.pkl", "wb".format(expert)) as f :
        pickle.dump(actions_dataset)

    with open("expert_data/{}_dagger_rewards.pkl".format(expert), "wb") as f :
        pickle.dump(rewards_dataset)