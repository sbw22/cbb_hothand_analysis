import http.client
import json
from datetime import datetime, timedelta
from math import exp
import numpy as np
from fetch_ncaa_data import NCAADataFetcher

# β
# η

# Process for BLHMM:

# Initial Distribution
    # Create initial state distribution for before any shots are taken. When I am making it, I am going 
    # to assume an equal probability of being in each state, but later maybe I can include player stats in here.

# Hidden States
    # Zin = ∈ {C, N, H} where C = cold, N = neutral, H = hot. These states are hidden because we can't directly 
    # observe whether a player is in a hot or cold state, we can only infer it from their shooting performance.
    # i = game index, n = shot number, Zin = hidden state during shot n of game i.
    # Yin = Observed shot outcome, made or missed ∈ {0, 1} where 0 = made shot, 1 = missed shot.

# Transition Matrix
    # Stores probabilities for moving from state (C, H, N) to state (C, H, N). 
    # Row = Current State, Column = Next State
    # Looks like

    #       ( p(CC) p(CN) p(CH) )
    # L =   ( p(NC) p(NN) p(NH) )
    #       ( p(HC) p(HN) p(HH) )

    # Each row should sum to 1 because the player has to transition to some state after each shot.

# Transisiton Distance
    # Zi,n+1 | Zin ~ Categorical( pi(jC), pi(jN), pi(jH), j ∈ {C, H, N} )

# Softmax Regression
    # Used for state transision probabilities:
        # j ∈ {C, H, N}
    # ηi(jC) = Xi + β(jC) + bi(jC) -   ( pi(jC) = e^(ηi(jC) / 1 + e^(ηi(jH)) + e^(ηi(jH))) )
    # ηi(jH) = Xi + β(jH) + bi(jH) -   ( pi(jH) = e^(ηi(jH) / 1 + e^(ηi(jC)) + e^(ηi(jH))) )
    #                                  ( pi(jN) = 1 / 1 + e^(ηi(jC)) + e^(ηi(jH)) )

    # Xi = Vector of features from dataset. 
        # Ex.
        #     [        1          ]
        #     [   shot distance   ]
        #     [  time remaining   ]
        #     [     fatigue       ]
        #     [ defender distance ]
        

    # β = Global coefficients for features in X. 
        # Same size as X.
        # Learned probabilities by the model so that the transition probabilities match the observed
        #  data well (as closely as possible). [ . . . ] form
    # bi = How much a game deviates from the average. Given its game i do we know if the player will 
    # be more streaky? If so, bi(jH) = +0.8 -> increase probability of transitioning to hot


# def feature_vector(n=1):
    # This is a placeholder function that should return the feature vector Xi for a given shot. 
    # In a real implementation, this function would extract relevant features from the dataset, such as shot distance, time remaining, fatigue level, defender distance, etc.
    # For now, it just returns a dummy vector of ones for testing purposes.
    
    #What we are currently using
    #     [       Current Score       ]
    #     [       Time Left       ]
    #     [       Time Between Shots        ]
    #     [       Home/Away        ]
    
    
    # return [1 for _ in range(n)] # Example feature vector with three features (intercept, shot distance, time remaining)

def B(j, len):
    # This is a placeholder function that should return the global coefficients β for the given state j. 
    # In a real implementation, these coefficients would be learned from the data during model fitting. 
    # For now, it just returns a dummy vector of ones for testing purposes.
    return [1 for _ in range(len)] # Example coefficient vector with one feature (intercept, shot distance, time remaining) (?)

def bi(j):
    # This is a placeholder function that should return the game-specific deviation bi for the given state j. 
    # In a real implementation, this would capture how much game i deviates from the average in terms of transition probabilities. 
    # For now, it just returns a dummy value of zero for testing purposes.
    return 0 # Example game-specific deviation (no deviation from average)

def dot_product(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def calculate_posterior(prior, likelihood):
    pass

def softmax_regression(j, C, H, Xi_n): 
    # print(f"Xi_n = {Xi_n}")
    # Xi = feature_vector()
    ni_jc = dot_product(Xi_n, B(j*C, len(Xi_n))) + bi(j*C)
    ni_jh = dot_product(Xi_n, B(j*H, len(Xi_n))) + bi(j*H)
    # print(f"Xi_n = {Xi_n}, B(j*C, len(Xi_n)) = {B(j*C, len(Xi_n))}, B(j*H, len(Xi_n)) = {B(j*H, len(Xi_n))}")
    # print(f"ni_jc={ni_jc}, ni_jh={ni_jh}, j = {j}")
    denom = 1 + exp(ni_jc) + exp(ni_jh)

    pjC = exp(ni_jc) / denom # a logit function
    pjH = exp(ni_jh) / denom
    pjN = 1 / denom

    return pjC, pjH, pjN

def process(shot_sequence, clock_sequence, is_home_sequence, initial_distribution=[0.5, 0.33, 0.33], gamma_C=0.2, gamma_N=0.32, gamma_H=0.48):
    # This is where the main processing of the data will happen. The steps will be:
    # 1. Fetch play-by-play data for a sample of games.
    # 2. Extract player-level shooting sequences from the play-by-play data.
    # 3. Fit the BLHMM to the shooting sequences to estimate transition probabilities and state distributions.
    # 4. Analyze the fitted model to identify hot hand patterns and evaluate their statistical significance.

    # posterior ∝ likelihood × prior

    # P(state | shot outcome) ∝ P(shot outcome | state) × P(state)

    # gamma_J = Probability of making a shot in state J, where J ∈ {C, N, H}. These are the likelihoods P(shot outcome | state) for each state.
    #implement 
    gammas = [gamma_C, gamma_N, gamma_H]

    C, N, H = initial_distribution
    belief = initial_distribution
    Xi = [clock_sequence, is_home_sequence]
    # print(f"clock_sequence len = {len(clock_sequence)}, shot_sequence len = {len(shot_sequence)}, is_home_sequence len = {len(is_home_sequence)}")
    '''# Transition matrix
    pCC, pCH, pCN = softmax_regression(C, C, H, Xi)
    pNC, pNH, pNN = softmax_regression(N, C, H, Xi)
    pHC, pHH, pHN = softmax_regression(H, C, H, Xi)

    # Note: Each row is a transition distance (I think) from the notes
    #    -> C  -> N  -> H
    p = [[pCC, pCN, pCH], # C
        [pNC, pNN, pNH],  # N
        [pHC, pHN, pHH]]  # H'''
    '''p = [
        [0.6, 0.3, 0.1],  # from Cold:    likely to stay cold
        [0.2, 0.6, 0.2],  # from Neutral: likely to stay neutral
        [0.1, 0.3, 0.6],  # from Hot:     likely to stay hot
    ]'''
    # Initialize p
    p = [[0, 0, 0], # C
        [0, 0, 0], # N
        [0, 0, 0]] # H

    for i, shot in enumerate(shot_sequence):
        # Here we would update the state distribution based on the observed shot outcome and the transition probabilities. 
        # This would involve calculating the likelihood of the observed shot given each possible hidden state, and then using Bayes' theorem to update our beliefs about the player's current state.
        made = (shot == 0) # Assuming shot_sequence is a list of 0s and 1s where 0 = made shot, 1 = missed shot
        # Xi = [feature[i] for feature in X]
        #change in the future to get player by player priors
        prior = [0, 0, 0] # The prior
        Xi_n = [stat_list[i] for stat_list in Xi]
        C, N, H = belief
        pCC, pCH, pCN = softmax_regression(C, C, H, Xi_n)
        pNC, pNH, pNN = softmax_regression(N, C, H, Xi_n)
        pHC, pHH, pHN = softmax_regression(H, C, H, Xi_n)
        # print(f"")

        # Note: Each row is a transition distance (I think) from the notes
        #    -> C  -> N  -> H
        p = [[pCC, pCN, pCH], # C
            [pNC, pNN, pNH],  # N
            [pHC, pHN, pHH]]  # H
        
        # print(f"Transition matrix on shot {i}: {p}")

        for next_s in range(3):
            for curr_s in range(3):
                prior[next_s] += belief[curr_s] * p[curr_s][next_s]
    
        updated = [0, 0, 0]
        for s in range(3):
            likelihood = gammas[s] if made else (1 - gammas[s])
            updated[s] = likelihood * prior[s]
        
        total = sum(updated)
        belief = [u / total for u in updated]

        print(f"Shot: {'make' if made else 'miss'} | P(C)={belief[0]:.3f}  P(N)={belief[1]:.3f}  P(H)={belief[2]:.3f} | {'make' if made else 'miss'}")

    
    print(f"\n\nTransition matrix:")
    for row in p:
        print(f"{row}")

    t_step_transition_probabilities(p, 10)
    occupancy_times(p, n=len(shot_sequence) - 1, initial_state=0)  # starting from Cold - 0 = cold, 1 = neutral, 2 = hot
    sojourn_times(p, t=1)
    return

def t_step_transition_probabilities(p, t=2):
    np_p = np.array(p)
    p_t = np.linalg.matrix_power(np_p, t)
    p_t = list(p_t)
    print(f"Transition probabilities in {t} step(s)")
    for row in p_t:
        print(f"{row}")

    '''pi_t = [[0,0,0],[0,0,0],[0,0,0]]
    for next_s in range(3):
        for curr_s in range(3):'''

def sojourn_times(p, t=1):
    # The sojourn time in state j is the expected number of consecutive shots a player will take in state j before transitioning to a different state. 
    # For a Markov chain, the sojourn time in state j can be calculated as 1 / (1 - p(jj)), where p(jj) is the probability of staying in state j.
    # TO-DO for this function: Calculate the posterior distribution
    sojourn_times = [0, 0, 0]
    for j in range(3):
        sojourn_times[j] = (p[j][j]**t) * (1 - p[j][j])
    
    print(f"\nSojourn times for each state for {t} step(s):")
    print(f"  Cold: {sojourn_times[0]:.3f} shots")
    print(f"  Neutral: {sojourn_times[1]:.3f} shots")
    print(f"  Hot: {sojourn_times[2]:.3f} shots")

    return sojourn_times


def occupancy_times(p, n, initial_state=None):  # LOOK OVER THIS FUNCTION CAREFULLY, CLAUDE GENERATED IT AND I WANT TO MAKE SURE IT IS CORRECT AND UNDERSTAND IT DEEPLY
    """
    Compute the occupancy time matrix for an n-shot sequence given a fixed
    transition matrix p.

    In the paper (Section 3.3, Eq. 9), the occupancy time m_i^(jk)(n) is the
    expected number of visits to state k starting from state j in the first
    n transitions of the chain.

    For 2 states, the paper uses a closed-form expression derived from the
    eigenvalues of the 2x2 transition matrix:

        M(n) = ((n+1)/(p_CH+p_HC)) * [[p_HC, p_CH],[p_HC, p_CH]]
             + (1-(p_HC+p_CH-1)^(n+1))/(p_CH+p_HC)^2 * [[p_CH,-p_CH],[-p_HC,p_HC]]

    For 3 states {C=0, N=1, H=2}, the equivalent general result is the matrix
    geometric series:

        M(n) = I + P + P^2 + ... + P^n  =  sum_{t=0}^{n} P^t

    where M[j][k] is the expected number of visits to state k when starting
    from state j, over n transitions (i.e. n+1 shots including the starting one).

    This collapses to the paper's 2-state closed form when applied to a 2x2
    matrix, and generalizes naturally to any number of states.

    Args:
        p:             3x3 transition matrix as a list of lists (rows sum to 1)
                       p[j][k] = P(next state = k | current state = j)
        n:             number of transitions (shots after the first)
        initial_state: optional int (0=C, 1=N, 2=H). If provided, returns a
                       1D array of expected visits to each state from that
                       starting state. If None, returns the full 3x3 matrix.

    Returns:
        If initial_state is None: 3x3 numpy array M where M[j][k] = expected
            visits to state k starting from state j in n transitions.
        If initial_state is int: 1D numpy array of length 3, the j-th row of M.
    """
    np_p = np.array(p)
    M = np.zeros((3, 3))
    # @ = matrix multiplication operator in numpy
    # Sum P^t for t = 0, 1, ..., n
    # P^0 = I (the starting state counts as visit 1)
    # np.eye = identity matrix of size 3x3
    P_power = np.eye(3)
    for t in range(n + 1):
        M += P_power
        P_power = P_power @ np_p

    state_names = ['C', 'N', 'H']
    print(f"\nOccupancy time matrix over {n} transitions ({n+1} shots):")
    print(f"  {'':6} {'->C':>8} {'->N':>8} {'->H':>8}")
    for j in range(3):
        row_str = "  ".join(f"{M[j][k]:8.3f}" for k in range(3))
        print(f"  {state_names[j]}: {row_str}  (sums to {M[j].sum():.3f}, should be {n+1})")

    if initial_state is not None:
        print(f"\nStarting from state {state_names[initial_state]}:")
        for k in range(3):
            print(f"  Expected visits to {state_names[k]}: {M[initial_state][k]:.3f}")
        return M[initial_state]

    return M
    


def shot_probability_regression(n):
    # This function would implement the regression model for the shot success probabilities in each state. 
    # It would take the design features of the shots (e.g., distance, time remaining) and the state-specific coefficients to calculate the probability of making a shot in each state.
    Xi = [feature_vector() for _ in range(len(shot_sequence))] # This is a placeholder for the actual design matrix of shot features.
    logit_yiC = dot_product(Xi, B(C)) + bi(C)
    logit_yiH = dot_product(Xi, B(H)) + bi(H)




# Initial Distribution
    # ∂i = (∂i(C), ∂i(N), ∂i(H)) starting with ∂i ~ Dirichlet(1, 1, 1)
    # So all states are equally likely at the start of game i. 
    # If a player is more likely to start cold, adjust to Dirichlet(5, 1, 1) for example, so cold state
    # is higher probability.

# Shot Outcome Model
    # Yin | Zin = C ~ Bernoulli(yin(C))
    # Yin | Zin = N ~ Bernoulli(yin(N))
    # Yin | Zin = H ~ Bernoulli(yin(H))
    # yin is the success probability in state. 

# Shot Prob. Regression
    # logit( yi(C) ) = Xi⍺(C) + Ξi ai(C) (?)
    # Xi = design features of observed features in game i.
    # Ex.
    #        [  1  distance1  time1  ]
    #  Xi =  [  1  distance2  time2  ]
    #        [  1  distance3  time3  ]
    # Where each row is a shot and the columns are features of the shot.
    # ⍺(C) = fixed-effect coefficients

def main():
    ncaa_data_fetcher = NCAADataFetcher()
    # Fetch the season sample, then summarize the first game as a quick smoke test.
    pbp_data = ncaa_data_fetcher.get_pbp_data()
    print(f"Finished getting play-by-play data for {len(pbp_data)} games.")

    # lists that hold info from all games in the sample.
    # Each item in total_game_stats is a dictionary of player stats for one game, structured like the output of get_all_player_stats_of_game().
    appended_game_stats = []
    # Each item in extended_game_stats is a player's shooting performance over multiple games, structured like the value for one player in 
    # the output of get_all_player_stats_of_game().
    extended_game_stats = []

    # Analyze the first returned contest because the script is still in exploratory mode.

    print(f"Inspecting the first game's play-by-play data:")
    # print(f"php_data[0]: {pbp_data[0].keys()}\n")
    # print(f"php_data[0]['periods'] has {len(pbp_data[0]['periods'])} periods, with keys {list(pbp_data[0]['periods'][0].keys())}\n")
    print(f"pbp_data[0]['periods'][0]['playbyplayStats'] has {len(pbp_data[0]['periods'][0]['playbyplayStats'])} events, with keys {pbp_data[0]['periods'][0]['playbyplayStats'][23]['homeText']}\n")

    # game_stats = get_all_player_stats_of_game(pbp_data[0])

    for game_info in pbp_data:
        game_stats = ncaa_data_fetcher.get_all_player_stats_of_game(game_info)

        appended_game_stats.append(game_stats)

        for player, stats in game_stats.items():

            if player not in extended_game_stats:
                # structure of extended_game_stats = [{player: {'two_point_sequence': [...], 'three_point_sequence': [...]}}]
                extended_game_stats.append({player: stats})
                continue

            extended_game_stats[player]['two_point_sequence'].extend(stats['two_point_sequence'])
            extended_game_stats[player]['three_point_sequence'].extend(stats['three_point_sequence'])
        # print(f"Total number of players in the first game: {len(game_stats)}")

    print("Player stats for the first game:")
    for player, stats in appended_game_stats[0].items():
        total_player_actions = len(stats['two_point_sequence']) + len(stats['three_point_sequence'])
        print(f"  {player}: {total_player_actions} total actions (2pt: {len(stats['two_point_sequence'])}, 3pt: {len(stats['three_point_sequence'])})")
    
    print(f"games in appended_game_stats: {len(appended_game_stats)}")

    print(f"extended_game_stats[0]: {extended_game_stats[5]}")
    test_player = extended_game_stats[5]['Milan Momcilovic']
    test_shot_sequence = test_player['three_point_sequence']
    test_clock_sequence = test_player['clock_time_sequence_three_point']
    test_is_home_sequence = test_player['is_home_sequence_three_point']

    print(f"test_shot_sequence: {test_shot_sequence}")
    process(test_shot_sequence, test_clock_sequence, test_is_home_sequence)
    return
    
    for player, actions in game_stats.items():
        # Show how many recorded events were attached to each player-like key.
        print(f"  {player}: {len(actions)} actions")
    return 
    # print(f"game_stats['Team'] = {game_stats['Team']}")
    print(f"\n\nDetailed actions for Team:")
    for item in game_stats['Team']:
        # print(f"  - {item['description']} at {item['time']} with score {item['score']}")
        print(f"{item}")
        print(f"homeText: {item['homeText']}")
        print(f"visitorText: {item['visitorText']}\n")
    
    # Dissect the event descriptioins for player. For each shot, we should be getting shot type 
    # (3pt, 2pt, layup, free throw), whether the shot was made or missed, time of event, and score of game).


    



if __name__ == "__main__":
    main()