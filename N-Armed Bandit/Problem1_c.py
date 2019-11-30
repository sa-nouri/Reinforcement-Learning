# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:21:55 2018

@author: Salar
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Initialization

Players = 10000
Games = 3

confidence_level = 0.9
confidence_intervals = np.zeros((Games, 2))

Players_PwsPls = np.array([
                          [0.5, 0.5],
                          [0.9, 0.3],
                          [0.9, 0.9]
                          ])

Players_Prob = np.array([0.2, 0.2, 0.6])
Games_Prob = np.array([0.7, 0.5, 0.3])

std_rewards = np.zeros((Games, ))
mean_rewards = np.zeros((Games, ))
rewards = []
for i in range (Games):
    rewards.append([])

# Calculation Procedure

for i in range (Games) :
    for j in range (Players) :
        current_reward = 0
        current_player = np.random.choice(3, p = Players_Prob)
        for k in range (10) :
            current_reward -= 1
            if (np.random.uniform() < Games_Prob[i]) :
                if (np.random.uniform() < (1 - Players_PwsPls[current_player, 0])) :
                    break
            else :
                if (np.random.uniform() < Players_PwsPls[current_player, 1]) :
                    break
                else :
                    current_reward += 11
        rewards[i].append(current_reward)

for i in range (Games) :
    mean_rewards[i] = np.mean(rewards[i])
    std_rewards[i] = np.std(rewards[i])/np.sqrt(len(rewards[i]))
    confidence_intervals[i, ] = [mean_rewards[i] - norm.ppf((1 + confidence_level)/2)*std_rewards[i], mean_rewards[i] +\
                                norm.ppf((1 + confidence_level)/2)*std_rewards[i]]

# Printing the Results

print('The Average Profit of the First Game is Equal to {}\nAnd the Corresponding Confidence Interval is Equal to {}'.format(mean_rewards[0], confidence_intervals[0, ]))
print('The Average Profit of the Second Game is Equal to {}\nAnd the Corresponding Confidence Interval is Equal to {}'.format(mean_rewards[1], confidence_intervals[1, ]))
print('The Average Profit of the Third Game is Equal to {}\nAnd the Corresponding Confidence Interval is Equal to {}'.format(mean_rewards[2], confidence_intervals[2, ]))
