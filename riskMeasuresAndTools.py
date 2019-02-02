import pandas as pd
import numpy as np
import math

'''This section calculates risk measures'''
def calc_var(dataset, alpha):
    '''funciton computes the value at risk at a risk level alpha
    INPUTS:
        dataset: dataframe object, dataset where columns are the variables
        alpha: float, level, should be between 0 and 1
    Outputs:
        out: output, should be a dataframe object, with 1 row of the value at risk for the given alpha
    '''
    out = pd.DataFrame(-1.0*dataset.quantile(alpha)).transpose()
    new_name = 'VaR at alpha = ' + str(alpha)
    out = out.rename({alpha: new_name}, axis='index')
    return out

def calc_cvar(dataset, alpha):
    '''calcCVar calculates the conditional value at risk for each column given a dataframe at the level alpha'''
    cols = dataset.columns
    vals = np.zeros((1,len(cols)))
    for i in range(len(cols)):
        col = cols[i]
        q = dataset[col].quantile(alpha)
        data2 = dataset[[col]].copy()
        data2 = data2[data2[col] <= q]
        vals[0][i] = data2.mean().values
    vals = pd.DataFrame(-vals, columns=cols, index=['CVaR at alpha = ' + str(alpha)])
    return vals


def calc_conditional_loss(dataset):
    '''calcCVar calculates the conditional value at risk for each column given a dataframe at the level alpha'''
    cols = dataset.columns
    vals = np.zeros((1,len(cols)))
    for i in range(len(cols)):
        col = cols[i]
        '''figure out probability you have lost money'''
        probWonMoney = (dataset[col] > 1).mean()
        probLostMoney = 1 - probWonMoney
        q = dataset[col].quantile(probLostMoney)
        data2 = dataset[[col]].copy()
        data2 = data2[data2[col] <= q]
        vals[0][i] = data2.mean().values
    vals = pd.DataFrame(vals, columns=cols, index=['Average Wealth Conditional On Losing Money '])
    return vals

def calc_var_goal(dataset, alpha, goal):
    '''function computes VaR with respect to the goal'''
    dataset2 = dataset - goal
    out = calc_var(dataset2, alpha)
    prev_name = 'VaR at alpha = ' + str(alpha)
    new_name = 'VaR wrt goal, alpha = ' + str(alpha)
    out = out.rename({prev_name: new_name}, axis='index')
    return out

def calc_cvar_goal(dataset, alpha, goal):
    '''function computes conditional value at risk for each column of dataset with respect to the goal at the the level alpha'''
    dataset2 = dataset - goal
    out = calc_cvar(dataset2, alpha)
    prev_name = 'CVaR at alpha = ' + str(alpha)
    new_name = 'CVaR wrt goal, alpha = ' + str(alpha)
    out = out.rename({prev_name: new_name}, axis='index')
    return out

def calc_prob_meeting_goal(dataset, goal):
    '''function computes the probability of meeting the goal for each folumn of the data set'''
    return pd.DataFrame((dataset >= goal).mean(), columns = ['Prob of Meeting Goal']).transpose()

def calc_prob_make_money(dataset):
    '''function computes the probability of meeting the goal for each folumn of the data set'''
    return pd.DataFrame((dataset > 1).mean(), columns = ['Prob of Making Money']).transpose()

def calc_downside_risk(dataset, goal):
    '''function calculates the expected downside risk of the columns of a pd dataframe (dataset), with respect to a goal'''
    temp = np.maximum(goal - dataset, 0)
    return pd.DataFrame(temp.mean(), columns = ['Expected Downside Risk']).transpose()

def calc_average_wealth(dataset):
    '''function computes the average of the columns of the dataset'''
    return pd.DataFrame(dataset.mean(), columns = ['Average Wealth']).transpose()

def calc_variance_wealth(dataset):
    '''function computes the average of the columns of the dataset'''
    return pd.DataFrame(dataset.var(), columns = ['Variance of Wealth']).transpose()

def calc_vol_wealth(dataset):
    '''function computes the average of the columns of the dataset'''
    return pd.DataFrame(dataset.std(), columns = ['Vol of Wealth']).transpose()



def full_report(dataset, goal, alpha=.05):
    '''function reports all of the risk measures for dataset, uses alpha for the VaR and CVaR calculations
    INPUTS:
        dataset: pandas df object, columns are the policy, rows are the scenarios
        goal: float, goal for the policy
        alpha: significance level for VaR and CVar calculations
    OUTPUTS:
        fullRiskdf: columns are the porfolios, rows are the risk measures'''

    '''Finally, compute measures only using the dataset itself'''
    fullRiskdf = calc_average_wealth(dataset)
    fullRiskdf = fullRiskdf.append(calc_vol_wealth(dataset))

    ''' Compute probability you will make money'''
    fullRiskdf = fullRiskdf.append(calc_prob_make_money(dataset))

    '''Next compute the risk measures involving the goal only'''
    fullRiskdf = fullRiskdf.append(calc_prob_meeting_goal(dataset, goal))

    '''Compute, conditional on the fact that you lose money, what is your final wealth on average'''
    fullRiskdf = fullRiskdf.append(calc_conditional_loss(dataset))

    return fullRiskdf