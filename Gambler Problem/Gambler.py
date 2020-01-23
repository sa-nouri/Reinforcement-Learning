## Salar Nouri ---- 810194422 ###
# Machine Learning ---- Homework (3 - 1)  ###

## ************************* ##

import numpy as np
import matplotlib.pyplot as plt


# =========================================


# Initializing Parameter

gamma = 1  # Discount Factor
P_Prof = 0.4 # Profitability Probiable
Num_Sates = 100 # The number of available States
Theta = 1e-6   # Threshold for comparing the differences


# Variable Initializing

Reward = [ 0 for _ in range ( Num_Sates + 1 )]
Reward[100] = 1 
Value = [ 0 for _  in range (  (2 * Num_Sates) + 1 )]
Policy = [ 0 for _  in range ( (2 *  Num_Sates )+ 1 )]


def BellmanOperator(num):
    Optimal_Value = 0 # Initializing Optimal Value

    for bet in range (0, min( num , 100 - num ) + 1):
        # Win and lose effect on value
        win = num + ( 2 *  bet ) 
        loss = num - bet 
        # Calculating Average value of possible states
        sum = P_Prof * ( Reward[win] + gamma * Value[win] ) + ( 1 - P_Prof) * ( Reward[loss] + gamma * Value[loss] )

        if sum > Optimal_Value :
            Optimal_Value = sum 
            Value[num] = sum
            Policy[num] = bet


def gambler():
    delta = 1 # Initial Capital

    while delta > Theta :
        delta = 0
        for i in range (1, Num_Sates):
            Value_Temp = Value[i]
            BellmanOperator(i)
            diff = abs( Value_Temp - Value[i] )
            delta = max ( diff, delta)


# Main Function

gambler()