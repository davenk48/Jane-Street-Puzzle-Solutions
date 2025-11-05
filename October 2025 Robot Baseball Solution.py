"""
The Artificial Automaton Athletics Association (Quad-A) is at it again, to compete with postseason baseball they are developing a Robot Baseball competition. Games are composed of a series of independent at-bats in which the batter is trying to maximize expected score and the pitcher is trying to minimize expected score. 

An at-bat is a series of pitches with a running count of balls and strikes, both starting at zero. For each pitch, the pitcher decides whether to throw a ball or strike, and the batter decides whether to wait or swing; these decisions are made secretly and simultaneously. The results of these choices are as follows. 
1. If the pitcher throws a ball and the batter waits, the count of balls is incremented by 1. 
2. If the pitcher throws a strike and the batter waits, the count of strikes is incremented by 1. 
3. If the pitcher throws a ball and the batter swings, the count of strikes is incremented by 1. 
4. If the pitcher throws a strike and the batter swings, with probability p the batter hits a home run and with probability 1-p the count of strikes is incremented by 1. 

An at-bat ends when either: 
1. The count of balls reaches 4, in which case the batter receives 1 point. 
2. The count of strikes reaches 3, in which case the batter receives 0 points. 
3. The batter hits a home run, in which case the batter receives 4 points. 

By varying the size of the strike zone, Quad-A can adjust the value p, the probability a pitched strike that is swung at results in a home run. They have found that viewers are most excited by at-bats that reach a full count, that is, the at-bats that reach the state of three balls and two strikes. Let q be the probability of at-bats reaching full count; q is dependent on p. Assume the batter and pitcher are both using optimal mixed strategies and Quad-A has chosen the p that maximizes q. Find this q, the maximal probability at-bats reach full count, to ten decimal places.
"""


"""
The strategy here is to model the at-bat as a series zero sum games, since the batter and pitcher play optimal mixed strategies
Note that the the at-bat has scores assigned to the batter at the end depending on the outcome
The batter will try to maximise their expected score, the pitcher will try to minimize the batter's expected score
Write a function to recursively play out the game from the start to every possible ending, for a given value of P
And then return the full count probability, this is when there have been 3 balls and 2 strikes
Then use an optimisation method to maximise the full count probability
"""

import numpy as np
import scipy
import random

def V(no_balls, no_strikes, p):
    if no_balls == 4:
        return 1
    if no_strikes == 3:
        return 0
    
    wait_ball = V(no_balls+1, no_strikes, p)
    wait_strike = V(no_balls, no_strikes+1, p)
    swing_ball = V(no_balls, no_strikes+1, p)
    swing_strike = p*4 + (1-p)*V(no_balls, no_strikes+1, p)
    
    payoff = np.array([
        [wait_ball, wait_strike],
        [swing_ball, swing_strike]
    ])
    
    value, batter_strategy, pitcher_strategy = solve_zero_sum_game(payoff)
    return value

def solve_zero_sum_game(payoff):
    a, b = payoff[0, 0], payoff[0, 1]
    c, d = payoff[1, 0], payoff[1, 1]
    
    denominator = a - b - c + d
    if denominator == 0:
        return (a + d) / 2, [0.5, 0.5], [0.5, 0.5]
    
    prob_wait = (d - b) / denominator    
    prob_ball = (d - c) / denominator
    
    prob_wait = max(0, min(1, prob_wait))
    prob_ball = max(0, min(1, prob_ball))
    
    prob_swing = 1 - prob_wait
    prob_strike = 1 - prob_ball
    
    batter_strategy = [prob_wait, prob_swing]
    pitcher_strategy = [prob_ball, prob_strike]
    
    value = prob_wait * (prob_ball*a + prob_strike*b) + prob_swing * (prob_ball*c + prob_strike*d)
    return value, batter_strategy, pitcher_strategy

def compute_full_count_probability(p):
    V_memo = {}
    Q = np.zeros((5, 4))
    Q[0][0] = 1.0

    for no_balls in range(4):
        for no_strikes in range(3):
            if no_balls == 4 or no_strikes == 3:
                continue

            wait_ball = V(no_balls+1, no_strikes, p)
            wait_strike = V(no_balls, no_strikes+1, p)
            swing_ball = V(no_balls, no_strikes+1, p)
            swing_strike = p*4 + (1-p)*V(no_balls, no_strikes+1, p)

            payoff = np.array([
                [wait_ball, wait_strike],
                [swing_ball, swing_strike]
            ])

            value, batter_strat, pitcher_strat = solve_zero_sum_game(payoff)
            prob_wait, prob_swing = batter_strat
            prob_ball, prob_strike = pitcher_strat

            current_prob = Q[no_balls][no_strikes]

            Q[no_balls+1][no_strikes] += current_prob * prob_wait * prob_ball
            Q[no_balls][no_strikes+1] += current_prob * (
                prob_wait * prob_strike +
                prob_swing * prob_ball +
                prob_swing * prob_strike * (1 - p)
                )
    return Q[3][2] 

result = scipy.optimize.minimize_scalar(lambda p: -compute_full_count_probability(p), bounds=(0, 1), method='bounded')
optimal_p = result.x
max_q = -result.fun
print(f"Optimal p: {optimal_p:.10f}, Max full count probability: {max_q:.10f}")

"""
Maximum Q is 0.2959679934
Achieved when P is 0.2269743429
"""


