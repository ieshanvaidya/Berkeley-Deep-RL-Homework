import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

learning_rates = [10 ** i for i in range(-6, 2, 1)]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80]

def getMatrix(expert, data) :
    raw = data[expert]
    matrix = np.zeros(shape = (len(learning_rates), len(epochs_list)))
    for entry in raw :
        lr, epochs, rewards = entry[0], entry[1], entry[2]
        mean_reward = np.mean(rewards)
        lr_index, epochs_index = learning_rates.index(lr), epochs_list.index(epochs)
        matrix[lr_index][epochs_index] = mean_reward

    return(matrix)

plt.rc('font', family = "serif")
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

experts = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]

thresholds = {"Ant-v2" : 0, "HalfCheetah-v2" : 0, "Hopper-v2" : 0, "Humanoid-v2" : 0, "Reacher-v2" : -20, "Walker2d-v2" : 0}
norm = {"Ant-v2" : True, "HalfCheetah-v2" : True, "Hopper-v2" : True, "Humanoid-v2" : True, "Reacher-v2" : False, "Walker2d-v2" : True}

for expert in experts :
    print("Expert : {}".format(expert))
    with open("expert_data/{}_hyperparams_tuning.pkl".format(expert), "rb") as f :
        data = pickle.load(f)

    matrix = getMatrix(expert, data)
    matrix[matrix < thresholds[expert]] = thresholds[expert]
    if norm[expert] :
        matrix /= np.max(matrix)

    fig, ax = plt.subplots()
    sns.heatmap(matrix, cmap ="Greens", annot = True, yticklabels = learning_rates, xticklabels = epochs_list)
    ax.set_title(expert)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    fig.savefig("figures/{}_hyperparams_tuning.png".format(expert), dpi = 300)
    #lt.show()