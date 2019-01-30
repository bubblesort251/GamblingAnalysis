#Import libraries

import numpy as np #for numerical array data
import pandas as pd #for tabular data
import matplotlib.pyplot as plt #for plotting purposes
import cvxpy as cp
import math
from scipy.special import comb
import itertools

#First function, return optimal bet as a function of win probabilities
def optimal_bet_given_probs(dfProb, vig):
    '''optimal_bet_given_probs takes a pandas df of probabilities dfProb and a vig, returns the optimal set of bets and expected return
    INPUTS:
        dfProb: pandas df, should have one column, rows should be probabilities that the bet will pay off
    OUTPUTS:
        1: array of weights to bet
        2: expected return
    '''

    #Calculate the number of bets
    numBets = dfProb.shape[0]
    inds = [x for x in range(numBets)]
    #Calculate the return of a winning bet (equals 1-vig)
    ret = 1-vig
    #Set up cvxpy problem
    w = cp.Variable(numBets)
    #enforce positivity contraint and that the amount bet is less than or equal to the wealth
    constraints = [w.T*np.ones((numBets,1)) <= 1, w >= 0]
    problem = cp.Problem(cp.Maximize(objective_given_probs(dfProb, w, ret)), constraints)
    #Solve the problem
    problem.solve()

    #return the solution, w is the optimal betting weights, the objective is the optimal return
    #here I convert the optimal return to a nominal return instead of a log return
    return w.value, math.exp(problem.value)-1

#helper function for optimal_bet_given_probs
def objective_given_probs(dfProb, w, ret):
    '''objective_given_probs is a helper function for optimal_bet_given_probs: returns the expected log return for a set of bets and probabilites of winning the bets
    INPUTS:
        dfProb: pandas df, should have 1 column, values should be probability of winning bets
        w: cvxpy variable object, should be numBets x 1 in length
        ret: return of a winning bet, should be 1-vig
    Outputs:
        obj: cvxpy expression, calculation of the expected log return for that set of bets
    '''
    numBets = dfProb.shape[0]
    inds = [x for x in range(dfProb.shape[0])]
    obj = 0.
    #Iterate over all possible combinations
    for r in range(numBets+1):
        for comb in itertools.combinations(inds, r=r):
            prob = 0.
            trues = np.zeros((numBets, 1))
            for i in range(numBets):
                if i in comb:
                    prob = prob + math.log(dfProb.values[i][0])
                    trues[i] = 1
                else:
                    prob = prob + math.log(1 - dfProb.values[i][0])
            prob = math.exp(prob)
            #now add the  the return for this combination
            obj = obj + prob*(cp.log(1 + ret*cp.matmul(w.T,trues) - cp.matmul(w.T, 1-trues)))
    return obj


#Now, define a function that calculates the break even probability
def calc_breakeven_prob(vig, n, cutoff = .01, tol=1e-4):
    '''calc_breakeven_prob calculates the probability you need to win a given bet in order to expect a certain level of compensation as a function of the vig
    INPUTS:
        vig: float, expressed as a decimal, percent the house takes of your money
        n: number of bets you can make in a given period
        cutoff: amount you need to win in order for the return to be considered positive
        tol: tolerance, controls how precision of the stopping condition
    OUTPUTS:
        prob: float, probability you need to win the bet in order to make an expected return above the cutoff
        sum(weights): float, amount of your money you bet in the period in order to realize that optimal solution
    ASSUMPTIONS:
        This function assumes that all the bets have the same payoff
        It uses binary search
    '''
    prob = .5
    h = .25
    dfProb = make_df_prob(prob, n)
    weights, obj = optimal_bet_given_probs(dfProb, vig)
    while (abs(obj - cutoff) > tol):
        if (obj - cutoff > 0):
            prob = prob - h
            h = h/2.
        else:
            prob = prob + h
            h = h/2.
        dfProb = make_df_prob(prob, n)
        weights, obj = optimal_bet_given_probs(dfProb, vig)
    return prob, sum(weights)

#Define a helper function for calc_breakeven_prob
def make_df_prob(prob, n):
    l = [prob]*n
    return pd.DataFrame(l, columns=['Probabilities'])


