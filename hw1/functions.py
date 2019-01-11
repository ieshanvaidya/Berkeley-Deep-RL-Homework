import pickle
import subprocess
import tensorflow as tf
import gym
from sklearn.model_selection import train_test_split, ParameterGrid
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
import load_policy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from scipy.stats import sem

def runExpert(expert, rollouts) :
    process = subprocess.Popen("python run_expert.py experts/{0}.pkl {1} --num_rollouts {2}".format(expert, expert, rollouts).split(" "), shell = False)
    process.wait()
    
def getExpertData(expert) :
    with open("expert_data/{}.pkl".format(expert), "rb") as f :
        data = pickle.load(f)
    return(data)

def prepareData(expert) :
    expert_data = getExpertData(expert)
    observations, actions, returns = expert_data.values()
    #Reshaping action to 2d
    actions = actions.reshape((actions.shape[0], actions.shape[-1]))
    #Train-validation split
    X_train, X_test, y_train, y_test = train_test_split(observations, actions, test_size = 0.1, random_state = 1)
    
    return(X_train, X_test, y_train, y_test)

def behaviorCloning(observations, actions, lr = 0.001, epochs = 30, batch_size = 128, verbose = 0) :
    #Getting shapes which are used in input and output layers of the network
    obs_shape = observations.shape
    act_shape = actions.shape
    obs_mean = np.mean(observations, axis = 0)
    obs_std = np.std(observations, axis = 0) + 1e-9
    
    model = Sequential()
    #model.add(InputLayer(input_shape = obs_shape[1:]))
    model.add(Lambda(lambda x : (x - obs_mean) / obs_std, input_shape = obs_shape[1:]))
    model.add(Dense(units = 64, activation = "tanh"))
    model.add(Dense(units = 64, activation = "tanh"))
    model.add(Dense(units = act_shape[-1]))
    
    optimizer = Adam(lr)
    
    model.compile(optimizer = optimizer, loss = "mse", metrics = ["mse"])
    
    model.fit(x = observations, y = actions, epochs = epochs, batch_size = batch_size, verbose = verbose)

    return(model)

def getReturns(expert, model, rollouts) :
    model_returns = []
    env = gym.make(expert)
    max_steps = env.spec.timestep_limit
    for i in tqdm(range(rollouts), desc = "Rollout", leave = False):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = obs.reshape((1, obs.shape[0]))
            action = model.predict(obs)
            action = action.reshape((action.shape[0], 1, action.shape[1]))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps >= max_steps:
                    break
        model_returns.append(totalr)
    return(model_returns)

def DAgger(expert, observations, actions, iterations, rollouts, lr = 0.001, epochs = 50, batch_size = 128) :
    obs_dataset = []
    act_dataset = []
    rewards = []
    #lr, epochs, batch_size = parameters["lr"], parameters["epochs"], parameters["batch_size"]
    train_observations, train_actions = observations, actions
    env = gym.make(expert)

    #Expert policy
    policy_fn = load_policy.load_policy("experts/{}.pkl".format(expert))

    for i in tqdm(range(iterations), desc = "DAgger Iteration", leave = False) :
        #Train model on training set D
        model = behaviorCloning(train_observations, train_actions, lr, epochs, batch_size)
        max_steps = env.spec.timestep_limit
        
        #Using trained model to get mean rewards
        returns = []
        for j in range(rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = obs.reshape((1, obs.shape[0]))
                model_action = model.predict(obs)
                model_action = model_action.reshape((model_action.shape[0], 1, model_action.shape[1]))
                obs, r, done, _ = env.step(model_action)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
            returns.append(totalr)
        rewards.append(returns)
        
        
        #Start with initial observation and run model to get [o1, o2, o3]
        new_observations = []
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            new_observations.append(obs)
            obs = obs.reshape((1, obs.shape[0]))
            model_action = model.predict(obs)
            model_action = model_action.reshape((model_action.shape[0], 1, model_action.shape[1]))
            obs, r, done, _ = env.step(model_action)
            steps += 1
            if steps >= max_steps:
                break
                    
        
        #Use these observations as input to expert and get expert actions
        with tf.Session() :
            new_actions = []

            for nobs in new_observations :
                expert_action = policy_fn(nobs[None,:])
                new_actions.append(expert_action)
            
        #Get labeled set D_exp = [(o1, a1), (o2, a2), ...]
        #Aggregate this to training set (D = D + D_exp) and retrain model
        obs_dataset.append(train_observations)
        act_dataset.append(train_actions)
        train_observations = np.concatenate((train_observations, np.array(new_observations)))
        train_actions = np.concatenate((train_actions, np.squeeze(np.array(new_actions))))
        
    return(obs_dataset, act_dataset, rewards)
