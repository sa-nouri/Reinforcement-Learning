#### SALAR NOURI ------ 810194422 ####
#### Homework (4) ----- Machine Learning ###
####  =================================== #####

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import gym
plt.style.use('ggplot')

# Parameters
Possible_Terminals = [[3, 0], [4, 0], [3, 1], [4, 1]]
Camera = [[1, 1], [1, 2], [4, 3], [2, 4], [0, 4], [1, 5]]
Key = [[3, 4], [3, 0]]

Q_Values = np.zeros((30, 4))
alpha_values = np.ones((30, 4))
Epsilon = 1
n = 4
Gamma = 0.97

Optimal_Return = 170
Regret = [0]
env = gym.make("MountainCar-v0")
lambda = 0.9


#=================== Functions================#
# Policy 

def make_epsilon_greedy_policy(estimator, epsilon, num_actions):

    def policy_fn(observation):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - epsilon)
        return action_probs
        
    return policy_fn


# n- step lambda sarsa
def sarsa_lambda(lmbda, env, estimator, gamma, epsilon):

    estimator.reset(z_only=True)

    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    ret = 0
    # Step through episode
    for t in itertools.count():
        # Take a step
        next_state, reward, done, _ = env.step(action)
        ret += reward

        if done:
            target = reward
            estimator.update(state, action, target)
            break

        else:
            # Take next step
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs)

            # Estimate q-value at next state-action
            q_new = estimator.predict(
                next_state, next_action)[0]
            target = reward + gamma * q_new
            # Update step
            estimator.update(state, action, target)
            estimator.z *= gamma * lmbda

        state = next_state
        action = next_action    
    
    return t, ret





    ### Main



for i in range (50000) :

    T, t = 30, 0
    State= []
    Reward= [0]
    Action = []
    Key = False

    while True :

        if (t < T) :

            if (Next_State in Terminal_States) :
                T = t + 1
            else :
                Actio.append(Policy(State[-1], epsilon))

        Tau = int(t - n + 1)
        if (Tau >= 0) :
            G = sum(Reward_[Tau + 1 : np.min([Tau + n, T]) + 1]*np.array([Gamma**x for x in range (len(Reward_Memory[Tau + 1 : np.min([Tau + n, T]) + 1]))]))
            if (Tau + n < T) :
            sarsa_lambda(n, env, estimator, gamma, epsilon)
            alpha[State_Memory[Tau][0] + 5*State_Memory[Tau][1], Action_Memory[Tau]] *= 0.9999

        if (Tau == T - 1) :
            break

        t += 1

    Regret.append(Regret[-1] + Optimal_Return - sum(Reward_Memory))

    if (np.mod(i, 10) == 0) :
        Epsilon *= 0.999

    if (Epsilon < 0.1) :
        Epsilon = 0

## Result
while True:

    print(Current_State)
    As = np.argmax(Q_Values[Current_State[0] + 5*Current_State[1]])
    Current_State = Next_State
    if (Current_State in Terminal_States or t > 30) :
        print(Current_State)
        break
    t += 1


plt.plot(Regret)
plt.xlabel('Episodes')
plt.ylabel('Regret')
plt.title('n-step SARSA TD(lambda = 0.9)')
plt.show()



##+===============================================+=
