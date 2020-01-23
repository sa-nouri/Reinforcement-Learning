#### SALAR NOURI ------ 810194422 ####
#### Homework (3-2) ----- Machine Learning ###
####  =================================== #####

import numpy as np
from MapBuilder_2 import *
import csv
import netwrokx as nx
import random
import matplotlib.pyplot as plt

#++++++++++++++++++++++++++++++++++++++++++++++++=
# CSV file Reading
def csv_reading(filename, Num_Features, Num_days):
    # Data Capturing Format
    Data = []
    for i in range( Num_Features * Num_days):
        Data.append([])
    # Reading CSV file
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        Feature_Sep = 0 
        for row in csv_reader:
            if (Feature_Sep >= Num_Features):
                for i in range (Num_Features) :
                    Data[i].append(np.array([float(row[i]), float(row[i + Num_Features]), float(row[i + (2 * Num_Features)])]))
    
    return Data

    #==========================================
    Map = MapBuilder()

    # Boltzamn Policy

def Policy (State, Temperature) :
    if (State == Map.terminal_state()) :
        As = np.random.choice(Map.terminal_state())
        return [Map.terminal_state(), As]

    if (State != Map.terminal_state()) :
        As = np.random.choice(len(Map.next_State(State)), p = \
                              np.exp(-Q_Values[State - 1]/Temperature)/sum(np.exp(-Q_Values[State - 1]/Temperature)))
    return [Map.next_State(State)[As], As]
        
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    class Sarsa:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward 
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
self.learnQ(state1, action1, reward, reward + self.gamma * qnext)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
        
        # The policy e_greedy
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        state = env.reset()
        for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            next_state, reward, done, _ = env.step(action)
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            estimator.update(state, action, td_target)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state
    
    return stats

#******************************************************************


for i in range (2500) :

    state = q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0)

    Episode_Return = 0
    Current_State = Map.initial_state()
    Next_State, Action = Policy(Current_State, Temperature)

    if Temperature > 0.1 :
        Temperature *= 0.99