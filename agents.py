import os
import sys
import gymnasium as gym
import time
import numpy as np
import random
from collections import defaultdict
import text_flappy_bird_gym
import itertools
import json


class Agent:
    def __init__(self, n_episodes, width, height):
        self.action_space = np.array([0, 1])
        self.num_actions = len(self.action_space)
        self.history = {
            "scores": np.zeros(n_episodes),
            "epsilon" : np.zeros(n_episodes)
        }
        poss_obs = list(itertools.product([_ for _ in range(width)], [_ for _ in range(-height, height)]))
        self.Q = {tuple(coordinates): [0, 0] for coordinates in poss_obs}
        self.N = {tuple(coordinates): 0 for coordinates in poss_obs}
        self.height = height
        self.width = width
        self.n_episodes = n_episodes

    def random_action(self):
        rand_action = np.random.choice(self.action_space)
        return rand_action

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.RandomState().choice(ties)
    
    def init_agent(self, obs):
        obs, info = obs
        
        return obs
    
    def update_history(self, episode, score, epsilon):
        self.history["scores"][episode] = score
        self.history["epsilon"][episode] = epsilon

    def get_action(self, obs):
        """
        Choose action epsilon-greedily.
        
        obs: (dx, dy) to nearest pipe

        """
        if random.random() < self.epsilon:
            return self.random_action()

        return self.argmax(self.Q[obs])
    

    def update_epsilon(self):
        if self.epsilon_decay == "linear":
            self.epsilon -= self.epsilon_decay_increment
        elif self.epsilon_decay == "exponential":
            self.epsilon *= self.epsilon_decay_factor

    
    def train(self, show_progress = True, save_res = True, results_path = "./results/"):
        best_score = 0
        best_episode = 0
        # Get infos
        n_episodes, height, width = self.n_episodes, self.height, self.width
        # Initiate environment
        env = gym.make('TextFlappyBird-v0', height = height, width = width, pipe_gap = 4)
        for episode in range(n_episodes):
            obs = env.reset()
            obs = self.init_agent(obs)
            # iterate
            while True:
                # Select next action
                action = self.get_action(obs)
                # Apply action and return new observation of the environment
                next_obs, reward, done, _, info = env.step(action)
                
                self.update(obs, next_obs, action, reward, done)
                
                # If player is dead break
                if done:
                    break
                
                #action = next_action
                obs = next_obs
            
            score = info["score"]
            
            self.update_history(episode, score, self.epsilon)
            self.update_epsilon()

            if score > best_score:
                best_score,best_episode = score, episode

            if show_progress:
                print(f"Episode {episode+1}/{self.n_episodes} *** High score: {best_score} obtained in episode {best_episode}", end = "\r")
        

        q = self.Q.copy()
        key_value = list(q.items())
        for key, val in key_value:
            q[str(key)] = val
            q.pop(key)
        n = self.N.copy()
        key_value = list(n.items())
        for key, val in key_value:
            n[str(key)] = val
            n.pop(key)

        res = {
            "agent_type": str(self),
            "params": self.params,
            "n_episodes": self.n_episodes,
            "best_score": best_score,
            "best_episode": best_episode,
            "scores": list(self.history["scores"]),
            "epsilon": list(self.history["epsilon"]),
            "Q": q,
            "N": n
        }

        filename = str(self)+ "lr" + str(self.params["step_size"]) + "_eps" + str(self.params["epsilon"]) + "_dec" + self.params["epsilon_decay"] + "_gamma" +  str(self.params["gamma"]) + "_" + str(self.n_episodes) 
        if filename + ".json" in os.listdir(path=results_path):
            filename = filename + "bis"
        if save_res:
            with open(results_path + filename + ".json", "w") as f:
                json.dump(res, f)

        return res



class QLearningAgent(Agent):

    def __init__(self, n_episodes, params: dict,  height= 15, width = 20):
        super().__init__(n_episodes, width, height)
        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.init_epsilon = params["epsilon"]

        if self.epsilon_decay:
            self.epsilon_decay_factor = (0.001/self.init_epsilon)**(1/n_episodes)
            self.epsilon_decay_increment = self.init_epsilon/n_episodes


    def __str__(self):
        return "QLearningAgent"
    

    def update(self, obs, next_obs, action, reward, done):
        self.N[obs] += 1
        self.Q[obs][action] += self.step_size * (reward + self.gamma*np.amax(self.Q[next_obs]) - self.Q[obs][action])



class ExpectedSARSAAgent(Agent):

    def __init__(self, n_episodes, params: dict,  height= 15, width = 20):
        super().__init__(n_episodes, width, height)
        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.init_epsilon = params["epsilon"]

        if self.epsilon_decay:
            self.epsilon_decay_factor = (0.001/self.init_epsilon)**(1/n_episodes)
            self.epsilon_decay_increment = self.init_epsilon/n_episodes

    def __str__(self):
        return "ExpectedSARSAAgent"
    

    def update(self, obs, next_obs, action, reward, done):
        probs = [self.epsilon/self.num_actions for _ in range(self.num_actions)]
        if len(set(self.Q[next_obs])) == 1:
            probs = [1/self.num_actions for _ in range(self.num_actions)]
        else:
            probs[np.argmax(self.Q[next_obs])] += 1 - self.epsilon
        weight_q = np.dot(probs, self.Q[next_obs])

        self.Q[obs][action] += self.step_size * (reward + self.gamma*weight_q - self.Q[obs][action])

    

class nStepSARSAAgent(Agent):
    def __init__(self, n_episodes, params, width = 20, height = 15):
        super().__init__(n_episodes, width, height)

        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.init_epsilon = params["epsilon"]


        if self.epsilon_decay:
            self.epsilon_decay_factor = (0.001/self.init_epsilon)**(1/n_episodes)
            self.epsilon_decay_increment = (self.init_epsilon - 0.001)/n_episodes

 
    def __str__(self):
        return f"nStepSARSAAgent"
    
    
    def get_prob(self, state, action):
        """
        Returns the probability of choosing an action given a state
        """
        if len(set(self.Q[state])) == 1:
            return 1/self.num_actions
        if action == np.argmax(self.Q["state"]):
            return self.epsilon/self.num_actions + 1 - self.epsilon
        else:
            return self.epsilon/self.num_actions


    def train(self, save_res = True, results_path = "./results/", show_progress = True):
        reward_store = []

        # Initiate environment
        env = gym.make('TextFlappyBird-v0', height = self.height, width = self.width, pipe_gap = 4)
        best_score, best_episode = 0, 0
        best_scores = []
        for episode in range(self.n_episodes):
            state = env.reset()
            state = self.init_agent(state)
            best_score_episode = 0

            action = self.get_action(state)
            states = [state]
            actions = [action]
            rewards = [0.0]

            T = float("inf")
            for t in itertools.count():
                if t < T:
                    
                    next_state, reward, done, _ , info = env.step(action)
                    states.append(next_state)
                    rewards.append(reward)

                    score = info["score"]

                    if score > best_score:
                        best_score,best_episode = score, episode

                    if score > best_score_episode:
                        best_score_episode = score

                    if done:
                        T = t + 1
                    
                    else:
                        next_action = self.get_action(next_state)
                        actions.append(next_action)
                        
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T)):
                        G += self.gamma**(i - tau -1)*rewards[i]

                    if tau + self.n < T:
                        G += self.gamma**self.n * self.Q[states[tau + self.n]][actions[tau + self.n]]
                        
                    self.Q[states[tau]][actions[tau]] += self.step_size*(G - self.Q[states[tau]][actions[tau]])

                if tau == T - 1:
                    break

                state = next_state
                action = next_action

            self.update_epsilon()
            self.update_history(episode, best_score_episode, self.epsilon)
            ret = np.sum(rewards)
            reward_store.append(ret)
            best_scores.append(best_score_episode)

            if show_progress:
                print(f"Episode {episode+1}/{self.n_episodes} *** High score: {best_score} obtained in episode {best_episode}", end = "\r")


        q = self.Q.copy()
        key_value = list(q.items())
        for key, val in key_value:
            q[str(key)] = val
            q.pop(key)
        n = self.N.copy()
        key_value = list(n.items())
        for key, val in key_value:
            n[str(key)] = val
            n.pop(key)

        res = {
            "agent_type": str(self),
            "params": self.params,
            "n_episodes": self.n_episodes,
            "best_score": best_score,
            "best_episode": best_episode,
            "scores": list(self.history["scores"]),
            "epsilon": list(self.history["epsilon"]),
            "Q": q,
            "N": n
        }

        filename = str(self) +str(self.n) + "_" + "lr" + str(self.params["step_size"]) + "_eps" + str(self.params["epsilon"]) + "_dec" + self.params["epsilon_decay"] + "_gamma" +  str(self.params["gamma"]) + "_" + str(self.n_episodes) 
        if filename + ".json" in os.listdir(path=results_path):
            filename = filename + "bis"
        if save_res:
            with open(results_path + filename + ".json", "w") as f:
                json.dump(res, f)


        return res

