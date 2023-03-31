import matplotlib.pyplot as plt
import json
import os
import sys
import gymnasium as gym
import time
import seaborn as sns
import itertools
from agents import *


def show_results(agent):
    scores = agent.results["scores"]
    epsilons = agent.results["epsilon"]

    fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    ax[0].plot(scores)
    ax[1].plot(epsilons)
    fig.tight_layout()



def read_results(filename, results_path = "./results/"):
    with open(results_path + filename + ".json", "r") as f:
        res = json.load(f)
    
    key_value = list(res["Q"].items())
    for key, val in key_value:
        res["Q"][eval(key)] = val
        res["Q"].pop(key)

    key_value = list(res["N"].items())
    for key, val in key_value:
        res["N"][eval(key)] = val
        res["N"].pop(key)

    return res

def train_agent(agent, show_progress = True, save_res = True, results_path = "./results/"):
    best_score = 0
    best_episode = 0
    # Get infos
    n_episodes, height, width = agent.n_episodes, agent.height, agent.width
    # Initiate environment
    env = gym.make('TextFlappyBird-v0', height = height, width = width, pipe_gap = 4)
    for episode in range(n_episodes):
        obs = env.reset()
        obs = agent.init_agent(obs)
        # iterate
        while True:
            # Select next action
            action = agent.get_action(obs)
            # Apply action and return new observation of the environment
            next_obs, reward, done, _, info = env.step(action)
            
            agent.update(obs, next_obs, action, reward, done)
            
            # If player is dead break
            if done:
                break
            
            #action = next_action
            obs = next_obs
        
        score = info["score"]
        
        agent.update_history(episode, score, agent.epsilon)
        agent.update_epsilon()

        if score > best_score:
            best_score,best_episode = score, episode

        if show_progress:
            print(f"Episode {episode+1}/{n_episodes} *** High score: {best_score} obtained in episode {best_episode}", end = "\r")
    

    q = agent.Q.copy()
    key_value = list(q.items())
    for key, val in key_value:
        q[str(key)] = val
        q.pop(key)
    n = agent.N.copy()
    key_value = list(n.items())
    for key, val in key_value:
        n[str(key)] = val
        n.pop(key)

    res = {
        "agent_type": str(agent),
        "params": agent.params,
        "n_episodes": n_episodes,
        "best_score": best_score,
        "best_episode": best_episode,
        "scores": list(agent.history["scores"]),
        "epsilon": list(agent.history["epsilon"]),
        "Q": q,
        "N": n
    }

    filename = str(agent)+ "lr" + str(agent.params["step_size"]) + "_eps" + str(agent.params["epsilon"]) + "_dec" + agent.params["epsilon_decay"] + "_gamma" +  str(agent.params["gamma"]) + "_" + str(n_episodes) 
    if filename + ".json" in os.listdir(path=results_path):
        filename = filename + "bis"
    if save_res:
        with open(results_path + filename + ".json", "w") as f:
            json.dump(res, f)

    return res


def hyperparam_tuning(grid, n_values = None, save_res = True, results_path="./results/"):
    keys = list(grid.keys())
    combs = list(itertools.product(*[grid[key] for key in keys]))
    for agent_type, n_episodes, epsilon_decay, step_size, epsilon, gamma in combs:
        if agent_type == "QLearningAgent":
            params = {
                "epsilon": epsilon,
                "epsilon_decay": epsilon_decay,
                "step_size": step_size,
                "gamma": gamma
            }
            agent = QLearningAgent(n_episodes, params)
            res = agent.train(show_progress=False, save_res=save_res, results_path=results_path)
        elif agent_type == "ExpectedSARSAAgent":
            params = {
                "epsilon": epsilon,
                "epsilon_decay": epsilon_decay,
                "step_size": step_size,
                "gamma": gamma
            }
            agent = ExpectedSARSAAgent(n_episodes, params)
            res = agent.train(show_progress=False, save_res=save_res, results_path=results_path)
        elif agent_type == "nStepSARSAAgent":
            for n in n_values:
                params = {
                    "epsilon": epsilon,
                    "epsilon_decay": epsilon_decay,
                    "step_size": step_size,
                    "gamma": gamma,
                    "n": n
                }
                agent = nStepSARSAAgent(n_episodes, params)
                res = agent.train(show_progress=False, save_res=save_res, results_path=results_path)
                if n != n_values[-1]:
                    print(f"{agent_type} * Best: {res['best_score']} * Params: {params} * Episodes: {n_episodes}")
        print(f"{agent_type} * Best: {res['best_score']} * Params: {params} * Episodes: {n_episodes}")
        

def plot_results(res_dict):
    sns.set_style('whitegrid')
    scores = res_dict["scores"]
    epsilons = res_dict["epsilon"]
    agent_type = res_dict["agent_type"]
    epsilon_decay = res_dict["params"]["epsilon_decay"]
    fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    fig.suptitle(f"{agent_type} with epsilon decay {epsilon_decay}")
    ax[0].plot(scores, "b")
    ax[0].set_xlabel("Episodes")
    ax[0].set_title(f"Score (best: {max(scores): .0f})")
    ax[1].plot(epsilons, "b")
    ax[1].set_xlabel("Episodes")
    ax[1].set_title(f"Epsilon (final value: {epsilons[-1]: .1e})")
    fig.tight_layout()

def show_results(agent):
    sns.set_style('whitegrid')
    scores = agent.results["scores"]
    epsilons = agent.results["epsilon"]

    fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    ax[0].plot(scores, "b")
    ax[0].set_xlabel("Episodes")
    ax[0].set_title(f"Score (best: {max(scores): .0f})")
    ax[1].plot(epsilons, "b")
    ax[1].set_xlabel("Episodes")
    ax[1].set_title(f"Epsilon (final value: {epsilons[-1]: .1e})")
    fig.tight_layout()


def get_state_value_plot(q, title):
    q_filtered = {k: v for k, v in q.items() if v != [0, 0]}
    dx_list = []
    dy_list = []
    for (dx, dy), (q_idle, q_flap) in q_filtered.items():
        if dx not in dx_list:
            dx_list.append(dx)
        if dy not in dy_list:
            dy_list.append(dy)


    x_range = np.arange(np.min(dx_list), np.max(dx_list)+1)
    y_range = np.arange(min(dy_list), max(dy_list)+1)

    data = np.array([[max(q[(dx, dy)]) for dx in x_range] for dy in y_range])

    # Create plot
    fig, ax = plt.subplots()

    # Create heatmap using imshow
    im = ax.imshow(data, cmap='viridis')
    plt.xticks(x_range)
    ax.set_xticklabels(x_range)
    plt.yticks(range(len(y_range)))
    ax.set_yticklabels([y for y in y_range])

    plt.gca().invert_xaxis()

    plt.xlabel("dx")
    plt.ylabel("dy")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.title(title)
    # Show plot
    plt.show()# Add colorbar

def plot_optimal_policy(q, title, gradient):
    q_filtered = {k: v for k, v in q.items() if sum(v) > 0}
    dx_list = []
    dy_list = []
    for (dx, dy), (q_idle, q_flap) in q_filtered.items():
        if dx not in dx_list:
            dx_list.append(dx)
        if dy not in dy_list:
            dy_list.append(dy)


    x_range = np.arange(np.min(dx_list), np.max(dx_list)+1)
    y_range = np.arange(np.min(dy_list), np.max(dy_list)+1)
    data = []
    j = 0
    i = 0
    for j, dy in enumerate(y_range):
        row = []
        for i, dx in enumerate(x_range):
            q_idle, q_flap = q[(dx, dy)]
            if gradient:
                sum_val = q_idle + q_flap
                el = q_flap / sum_val if sum_val > 0 else 0.5
            else:
                if q_idle == q_flap:
                    el = None
                el = q_flap > q_idle

            row.append(el)
        data.append(row)


    # Create plot
    fig, ax = plt.subplots()

    # Create heatmap using imshow
    im = ax.imshow(data, cmap='viridis')
    plt.xticks(x_range)
    ax.set_xticklabels(x_range)
    plt.yticks(range(len(y_range)))
    ax.set_yticklabels([y for y in y_range])

    plt.gca().invert_xaxis()

    plt.xlabel("dx")
    plt.ylabel("dy")

    # Add colorbar
    if gradient:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Tendency to flap")
    plt.title(title)
    # Show plot
    plt.show()# Add colorbar