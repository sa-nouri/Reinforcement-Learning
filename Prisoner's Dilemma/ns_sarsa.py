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


# n- step sarsa


def sarsa_n(n, env, estimator, gamma=1.0, epsilon=0):
    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)
    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    states = [state]
    actions = [action]
    rewards = [0.0]

    T = float('inf')
    for t in itertools.count():
        if t < T:           
            # Take a step
            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)

            if done:
                T = t + 1

            else:
                # Take next step
                next_action_probs = policy(next_state)
                next_action = np.random.choice(
                    np.arange(len(next_action_probs)), p=next_action_probs)

                actions.append(next_action)

        update_time = t + 1 - n  # Specifies state to be updated
        if update_time >= 0:       
            # Build target
            target = 0
            for i in range(update_time + 1, min(T, update_time + n) + 1):
                target += np.power(gamma, i - update_time - 1) * rewards[i]
            if update_time + n < T:
                q_values_next = estimator.predict(states[update_time + n])
                target += q_values_next[actions[update_time + n]]
            
            # Update step
            estimator.update(states[update_time], actions[update_time], target)
        
        if update_time == T - 1:
            break

        state = next_state
        action = next_action

    ret  np.sum(rewards)
    
    return t, ret


    
## Main f


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
            sarsa_n(n, env, estimator, gamma, epsilon)
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
plt.title('n-step SARSA TD(0)')
plt.show()



##+===============================================+=
