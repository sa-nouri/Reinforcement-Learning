###  Salar Nouri ---- 810194422 ###
### Problem(1) == Part(a & b ) ---- HOMEWORK#2   ---- Machine Learning ###

# ************************************************ #

# Libraries which is Used

import numpy as np
import matplotlib.pyplot as plt
import random


# ************   Functions and Clsses ************* #####



# Reward or Punishment Functionality
def reward_punishment(Gaussian_parameters):
    return np.random.normal(Gaussian_parameters[0],Gaussian_parameters[1])
            


#  Bandit ---- Which chooses arm and gets reward or punishment 
class Bandit:
    
    def __init__(self, bandit_problabilties, inst_reward, S, F):
        self.S = S
        self.F = F
        self.N = len(bandit_problabilties)
        self.Probability = bandit_problabilties
        self.inst_reward = inst_reward

    def get_rewards(self, arm):
        choice = 1 if( random.random() < self.Probability [arm] ) else 0 
        return reward_punishment(self.inst_reward[arm][choice])


# Learning Agent 
class Agent:

    # armes_bandit Information

    def __init__(self, bandit, epsilon, S, F):
        self.epsilon = epsilon
        self.k = np.zeros(bandit.N, dtype = np.int)  # number of times action was chosen
        self.Q_V = np.zeros(bandit.N, dtype = np.float)  # estimated value

    # Calculating Q-Value 
    def Q_Value(self, arm, reward):
        self.k[arm] += 1
        self.Q_V[arm]  += ( reward - self.Q_V[arm]) * (1./self.k[arm])
        

    # Return Maximum Index
    def ind_max(self, x):
        m = max(x)
        return x.index(m)


    # Epsilon Greedy Policy Functionality
    def Epsilon_Greedy(self, bandit):
            if random.random() > self.epsilon :
                return np.argmax(self.Q_V)
            else:
                return random.randrange(bandit.N)       

    
    # Upper Confidence Bound 1 Policy Functionality
    def UCB_1(self,R, k, N):
        return np. argmax(self.Q_V + R * np.sqrt(np.log(k)/N))
    
    
# Enviorment And Experiments 
def Experiments( agent, bandit, Num_Episodes, Policy):
    action_history = []
    reward_history = []
    R = 2 
    for Trial in range (Num_Episodes):
        if Policy == 'E_Greedy' :
            action = agent.Epsilon_Greedy(bandit)
        elif Policy == 'UCB1' :
            action = agent.UCB_1(R, Trial, Num_Episodes)
            
        reward = bandit.get_rewards(action) # Gets Reward or Punishment according to Choosen Action
        agent.Q_Value(action, reward) # Updating Q_Value accordning to Reward or Punishment 

        action_history.append(action)
        reward_history.append(reward)
    
    return [ np.array(action_history), np.array(reward_history)]


def Thompson_Samp( bandit):
        x = np.random.beta(bandit.S + 1, bandit.F + 1, bandit.N)
        y = np.argmax(x)
        if( random.random() > bandit.Probability[np.argmax(x)]):
            bandit.S +=1
        else:
            bandit.F +=1
        return [bandit.S, bandit.F]

#****** Main Funtion ****** #

bandit_Prob = [ 0.5 , 0.6 , 0.6]
Num_Exp = 1000 
Num_Episodes = 10000 
Epsilon = 0.1
Reward_Punish = [[ [60, 8], [-40, 8] ],
                    [ [40, 60], [-40, 70]],
                    [ [20, 8], [-10, 10]]
                    ]

reward_history_avg = np.zeros(Num_Episodes)  # reward history experiment-averaged
action_history_sum = np.zeros((Num_Episodes, len(bandit_Prob)))  # sum action history
Policy = 'UCB1'

for i in range(Num_Exp):
    bandit = Bandit(bandit_Prob, Reward_Punish)  # initialize bandits
    agent = Agent(bandit, Epsilon)  # initialize agent
    (action_history, reward_history) = Experiments(agent, bandit, Num_Episodes, Policy)  # perform experiment
    
    # Sum up experiment reward (later to be divided to represent an average)
    reward_history_avg += reward_history
    
    # Sum up action history
    for j, (a) in enumerate(action_history):
        action_history_sum[j][a] += 1

    reward_history_avg /= np.float(Num_Exp)

 #######   


N_bandits = len(bandit_Prob)

# =========================
# Plot

plt.plot(reward_history_avg)
plt.xlabel("Episode number")
plt.ylabel("Rewards collected".format(Num_Exp))
plt.title("Bandit reward history averaged over {} experiments (epsilon = {})".format(Num_Exp, Epsilon))
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
plt.xlim([1, Num_Episodes])
plt.show()

# =========================
# Plot

plt.figure(figsize=(18, 12))
for i in range(N_bandits):
    action_history_sum_plot = 100 * action_history_sum[:,i] / Num_Exp
    plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
                action_history_sum_plot,
                linewidth=5.0,
                label="Bandit #{}".format(i+1))
plt.title("Bandit action history averaged over {} experiments (epsilon = {})".format(Num_Exp, Epsilon), fontsize=26)
plt.xlabel("Episode Number", fontsize=26)
plt.ylabel("Bandit Action Choices (%)", fontsize=26)
leg = plt.legend(loc='upper left', shadow=True, fontsize=26)
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
plt.xlim([1, Num_Episodes])
plt.ylim([0, 100])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
for legobj in leg.legendHandles:
    legobj.set_linewidth(16.0)
plt.show()
