#Import libraries

import numpy as np #for numerical array data
import pandas as pd #for tabular data
import matplotlib.pyplot as plt #for plotting purposes
import cvxpy as cp
import math
from scipy.special import comb
import itertools
import random

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

#Second function, return optimal bet given a set of bets with the same win probability
def optimal_bet_given_same_probs(dfProb, numBets, vig):
    '''optimal_bet_given_same_probs takes a pandas df defining the win probability, returns the optimal bet size and expected return
    INPUTS:
        dfProb: pandas df, should have 1 column and all rows being the same value, the probability a given bet will pay off
        numBets: int, number of bets one can place given that probability
        vig: float, amount lost due to transaction cost
    OUTPUTS:
        1: amount to place for one of the bets
        2: expected return
    '''
    ret = 1-vig
    w = cp.Variable(1)
    constraints = [w*numBets <= 1, w >= 0]
    problem = cp.Problem(cp.Maximize(objective_given_same_probs(dfProb, w, numBets, ret)))
    #Solve the problem
    problem.solve()

    #return the solution, w is the optimal betting weight for a given bet, the objective is the optimal return
    #Here I convert the optimal return to a nominal return instead of a log return
    return w.value[0], math.exp(problem.value)-1

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

def objective_given_same_probs(dfProb, w, numBets, ret):
    '''objective_given_same_probs takes a probability for winning, a number of bets and returns the objective function for log optimal growth
    INPUTS:
        dfProb: pandas df, should have 1 row and 1 column.  Should be a probability
        w: cvxpy variable, length 1
        numBets: integer, should be number of bets that you can place with the specific win probabbility
        ret: float, return of a winning bet
    OUTPUTS:
        obj: cvxpy expression, objective function
    '''
    obj = 0.
    winProb = dfProb.values[0][0]
    for r in range(numBets + 1):
        logProb = r*math.log(winProb) + (numBets - r)*math.log(1 - winProb)
        prob = comb(numBets, r)*math.exp(logProb)
        obj = obj + prob*(cp.log(1 + ret*(r*w) - (numBets-r)*w))

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
    dfProb = make_df_prob(prob)
    weights, obj = optimal_bet_given_same_probs(dfProb, n, vig)
    while (abs(obj - cutoff) > tol):
        if (obj - cutoff > 0):
            prob = prob - h
            h = h/2.
        else:
            prob = prob + h
            h = h/2.
        dfProb = make_df_prob(prob)
        weights, obj = optimal_bet_given_same_probs(dfProb, n, vig)
    return prob, n*weights

#Define a helper function for calc_breakeven_prob
def make_df_prob(prob):
    l = [prob]
    return pd.DataFrame(l, columns=['Probabilities'])

#Helper functions to run monte-carlo simulations
#Define probability cutoffs for different events
def probability_cutoffs(dfProb, numBets):
    '''probability_cutoffs is a helper function, returns a list of probabilites
    INTPUTS:
        dfProb: pandas df object, should have 1 column and at least 1 row.  The value should be the probability of winning a bet
        numBets: int, number of bets available to the etter this time period
    OUTPUTS:
        l: list object, values should be the probability cutoffs for the bets
        It assumes that the first cutoff is winning 0 bets, and the last cutoff is winning all bets
    '''
    l = list()
    winProb = dfProb.values[0][0]
    prob = 0
    for r in range(numBets+1):
        logProb = r*math.log(winProb) + (numBets - r)*math.log(1 - winProb)
        prob = prob + comb(numBets, r)*math.exp(logProb)
        l.append(prob)
    return l

#Helper function for calculating return
def get_return(placeholder, amountPerBet, numBets, probCutoffs, ret):
    realization = random.random()
    for numWins in range(len(probCutoffs)):
        if (realization <= probCutoffs[numWins]):
            #The return is this amount
            return (1. + ret*numWins*amountPerBet - (numBets - numWins)*amountPerBet)

    return 1+ret*nBets*amountPerBet


#Helper function to print output
def print_conditions_for_wealth_paths(dfProb, nBets, optimalAmountPerBet, listOfTimeStepsToRecord, vig, numSim):
    print('Here are the simulation conditions')
    print('During each time period, the player places ' + str(nBets) + ' bets')
    print('Each bet has a win probability of ' + str(100*dfProb.values[0][0]) + '%')
    print('Each bet made will be for ' + str(100*optimalAmountPerBet) + '% of the player''s wealth')
    print('Each winning bet will return ' + str(100*(1-vig)) + '% of the amount bet ')
    print('It runs ' + str(numSim) + ' independent monte-carlo simulations')
    print('It records the wealth distribution after the following time steps ' + str(listOfTimeStepsToRecord))

def simulate_wealth_paths(dfProb, numBets, listOfTimeStepsToRecord, vig=.02, numSim=10000):
    '''simulate_wealth_paths simulates wealth paths given the optimal betting framework
    INPUTS:
        dfProb: pandas df object, should have 1 column and at least 1 row.  The value should be the probability of winning a bet
        numBets: int, number of bets one can make per week
        listoftimeStepsToRecord: list object, time points to record the wealth distribution 
        nSim: integer, number of wealth path simulations to perform
    OUTPUTS:
        dfOut: pandas df, should have number of columns equal to the number of time steps
    '''
    #Step 1: Define helpful quantities
    optimalAmountPerBet, _ = optimal_bet_given_same_probs(dfProb, numBets, vig)
    probCutoffs = probability_cutoffs(dfProb, numBets)
    ret = 1-vig

    #Print out what the simulation will do
    print_conditions_for_wealth_paths(dfProb, numBets, optimalAmountPerBet, listOfTimeStepsToRecord, vig, numSim)

    #Step 2: initialize dfOut object
    dfOut = pd.DataFrame(np.ones((numSim,1)), columns=['Time 0'])
    
    #Step 3: Iterate over all the time periods
    for time in range(1,max(listOfTimeStepsToRecord)+1):
        #Create a df array of ones
        dfTimeStep = pd.DataFrame(np.ones((numSim, 1)), columns=['Returns'])
        dfTimeStep = dfTimeStep.apply(get_return, args=(optimalAmountPerBet, numBets, probCutoffs, ret,), axis=1)
        if(time == 1):
            dfNewTime = dfTimeStep
        else:
            dfNewTime = dfNewTime*dfTimeStep
        if(time in listOfTimeStepsToRecord):
            dfOut['Time ' + str(time)] = dfNewTime

    return dfOut
    #dfOut = pd.DataFrame(pd)



