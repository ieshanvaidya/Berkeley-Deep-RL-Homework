from functions import *
import matplotlib.pyplot as plt

plt.rc('font', family = "serif")
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#experts = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]
experts = ["Reacher-v2"]

def runDAgger(expert, iterations, rollouts, lr = 0.001, epochs = 30, batch_size = 128) :
    expert_data = getExpertData(expert)
    observations, actions, expert_returns = expert_data["observations"], expert_data["actions"], expert_data["returns"]
    actions = np.squeeze(actions)

    return(DAgger(expert, observations, actions, iterations, rollouts, lr, epochs, batch_size))
    #obs_dataset[expert], act_dataset[expert], rewards_dataset[expert] = DAgger(expert, observations, actions, iterations, rollouts, lr, epochs, batch_size)

def plotDagger(expert) :
    with open("expert_data/{}_dagger_rewards.pkl".format(expert), "rb") as f :
        rewards = pickle.load(f)

    with open("expert_data/behaviorcloning.pkl", "rb") as f :
        data = pickle.load(f)

    for row in data :
        if row[0] == expert :
            expertreward = row[1]
            modelreward = row[2]
            break

    fig, ax = plt.subplots()
    plotr = [np.mean(i) for i in rewards]
    plots = [sem(i) for i in rewards]
    x = np.arange(1, len(plotr) + 1, 1)
    ax.errorbar(x, plotr, yerr = plots, color = "blue", alpha = 0.7, ecolor = "black", capsize = 2, label = "DAgger")
    ax.plot(x, modelreward * np.ones(len(x)), color = "red", alpha = 0.7, linestyle = "--", label = "Behavior Cloning")
    ax.plot(x, expertreward * np.ones(len(x)), color = "green", alpha = 0.7, linestyle = "--", label = "Expert")
    ax.set_xlabel("DAgger Iterations")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    ax.set_title(expert)
    fig.savefig("figures/{}_dagger.png".format(expert), dpi = 300, bbox_inches = "tight")
    plt.show()

iterations = 50
rollouts = 30

dagger_run = False

if dagger_run :
    for expert in experts :
        print("Expert : {}".format(expert))
        observation_dataset, actions_dataset, rewards_dataset = runDAgger(expert, iterations, rollouts)

        with open("expert_data/{}_dagger_observations.pkl".format(expert), "wb") as f :
            pickle.dump(observation_dataset, f)

        with open("expert_data/{}_dagger_actions.pkl".format(expert), "wb") as f :
            pickle.dump(actions_dataset, f)

        with open("expert_data/{}_dagger_rewards.pkl".format(expert), "wb") as f :
            pickle.dump(rewards_dataset, f)

dagger_plot = True
if dagger_plot :
    plotDagger("Reacher-v2")