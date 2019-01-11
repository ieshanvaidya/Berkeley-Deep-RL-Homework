from functions import *
import matplotlib.pyplot as plt

plt.rc('font', family = "serif")
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

experts = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]
#experts = ["Ant-v2"]

rollouts = 30

def tuneHyperparams(expert, hyperparams, rollouts) :
    '''
    Expects dictionary of learning_rates and epochs_list
    '''
    rewards = defaultdict(list)
    for hp in ParameterGrid(hyperparams) :
        lr, epochs = hp["lr"], hp["epochs"]
        print("\rLearning Rate : {}, Epochs : {}".format(lr, epochs), end = "")
        expert_data = getExpertData(expert)
        observations, actions, expert_returns = expert_data["observations"], expert_data["actions"], expert_data["returns"]
        actions = np.squeeze(actions)
        model = behaviorCloning(observations, actions, lr = lr, epochs = epochs)
        model_returns = getReturns(expert, model, rollouts)
        rewards[expert].append([lr, epochs, model_returns])

    with open("expert_data/{}_hyperparams_tuning.pkl".format(expert), "wb") as f :
        pickle.dump(rewards, f)


#run_expert.py needs to be ran first to generate data
generate_expert = False
if generate_expert :
    for expert in experts :
        runExpert(expert, rollouts)


#Hyperparam grid
learning_rates = [10 ** i for i in range(-6, 2, 1)]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80]
hyperparams = {"lr" : learning_rates, "epochs" : epochs_list}

#Using 10 rollouts to tune (compuational constraint, takes long time)
perform_tuning = False
if perform_tuning :
    for expert in experts :
        print("Expert : {}".format(expert))
        tuneHyperparams(expert, hyperparams, 10)


#Using optimal learning rate to tune epochs (behavior cloning function has been modified so that optimal params are default)
generate_data = True
if generate_data :
    results = []
    for expert in experts :
        print("Expert : {}".format(expert))
        expert_data = getExpertData(expert)
        observations, actions, expert_returns = expert_data["observations"], expert_data["actions"], expert_data["returns"]
        actions = np.squeeze(actions)
        model = behaviorCloning(observations, actions)
        model_returns = getReturns(expert, model, rollouts)
        results.append([expert, np.mean(expert_returns), np.mean(model_returns), np.std(expert_returns), np.std(model_returns)]) 

    with open("expert_data/behaviorcloning.pkl", "wb") as f :
        pickle.dump(results, f)

    print(results)