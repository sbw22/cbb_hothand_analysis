import http.client
import json
from datetime import datetime, timedelta
from math import exp
import numpy as np


# Centralize the API host and season boundaries so the rest of the script can
# refer to a single source of truth instead of repeating literal values.
API_HOST = "ncaa-api.henrygd.me"
# The first day we want to inspect for the 2025-26 college basketball season.
SEASON_START = datetime(year=2025, month=11, day=3)
SEASON_START = datetime(year=2026, month=1, day=1) # switched to Jan 1 for testing so we get conference games in the sample
# The cutoff date for the season loop; the code walks day by day until this date.
SEASON_END = datetime(year=2026, month=4, day=15)
# Cap the number of games we fetch during testing so the script stays fast.
MAX_TEST_GAMES = 50


def get_game_ids():

    # Open a reusable HTTPS connection to the NCAA API host.
    conn = http.client.HTTPSConnection(API_HOST)

    # Start from the first day of the season and collect game IDs one date at a time.
    specific_date = SEASON_START
    # This list accumulates every game ID discovered from the scoreboard endpoint.
    list_of_game_ids = []
    # Temporary guard that keeps the script from iterating too far while testing.
    counter = 0
    while specific_date < SEASON_END:
        # Only inspect the first date in this reduced testing mode.
        if counter > 10: #####################################################################################################################
            break
        # Format the date to match the API path shape YYYY/MM/DD.
        date_str = specific_date.strftime("%Y/%m/%d")
        # Request the daily scoreboard for Division I men's basketball.
        conn.request("GET", f"/scoreboard/basketball-men/d1/{date_str}/all-conf") # switched all-conf to big-12
        # Read the HTTP response body so it can be parsed as JSON.
        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")
        try:
            # Convert the raw response text into a Python dictionary.
            json_data = json.loads(decoded_data)
        except json.JSONDecodeError:
            # If the API returns HTML or any non-JSON payload, stop and surface it.
            print("Response was not valid JSON")
            print(decoded_data)
            return

        # The scoreboard response contains a `games` array; each item wraps a game object.

        for game in json_data['games']:
            # Pull out the unique game identifier that will be used for play-by-play calls.
            game_id = game['game']['gameID']
            home_team = game['game']['home']
            away_team = game['game']['away']
            home_team_conference = home_team['conferences'][0]['conferenceSeo']
            away_team_conference = away_team['conferences'][0]['conferenceSeo']
            if away_team_conference != "big-12" and home_team_conference != "big-12":
                continue
            counter += 1 # Only trigger counter if we find a big 12 game
            if counter > 10: ######################################################################################################################
                break
            list_of_game_ids.append(game_id)
            print(f"len of list_of_game_ids: {len(list_of_game_ids)}, counter value: {counter}")
            # print(f"keys in game object: {game['game'].keys()}")
            # print(f"home team: {game['game']['home']}")#  , visitor team: {game['game']['away']}")
            # print(f"big 12 team found")

        # Move forward one day and continue scanning the season schedule.
        specific_date += timedelta(days=1)
        if len(json_data['games']) == 0:
            print(f"No games found for {date_str}")
    
    # Close the connection before returning the accumulated game IDs.
    conn.close()
    
    return list_of_game_ids


def get_pbp_data():

    # Open a second connection for the play-by-play requests.
    conn = http.client.HTTPSConnection(API_HOST)

    # Use only a small subset of games while debugging so the API load stays low.
    list_of_game_ids = get_game_ids()[:MAX_TEST_GAMES]

    # Give a quick progress summary before making the per-game requests.
    print(f"Found {len(list_of_game_ids)} game IDs for testing.")

    # Store one parsed play-by-play payload per game.
    list_of_pbp_data = []
    for game_id in list_of_game_ids:
        # Request the play-by-play endpoint for one specific contest.
        conn.request("GET", f"/game/{game_id}/play-by-play")
        # Read and decode the response so JSON parsing can happen safely.
        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")
        try:
            # Parse the JSON payload into a Python dictionary.
            pbp_json_data = json.loads(decoded_data)
            list_of_pbp_data.append(pbp_json_data)
        except json.JSONDecodeError:
            # If a game request fails, print the raw response and keep going.
            print(f"Response for game ID {game_id} was not valid JSON")
            print(decoded_data)
            continue

    
    # If every request failed, return an empty list instead of indexing into nothing.
    if not list_of_pbp_data:
        print("No play-by-play data was returned for any game.")
        conn.close()
        return []

    # Close the connection after the batch is complete.
    conn.close()

    return list_of_pbp_data


def get_all_player_stats_of_game(pbp_data):
    # This dictionary is the output of the function: each key is a player-like label,
    # and each value is the chronological list of events we were able to associate
    # with that label.
    player_stats = {}

    # The play-by-play payload is organized by periods, so if that top-level field
    # is missing there is nothing meaningful to walk and the function can exit early.
    if 'periods' not in pbp_data:
        print(f"'periods' key not found in pbp_data\n")
        return {}

    
    # For some reason, each play is duplicated in the play-by-play data, so for now I am just going to skip the next event when we record an event.
    skip_counter = False

    # Walk every period, then each play within the period, and attach the play to the
    # best actor label we can infer from the payload.
    for period in pbp_data['periods']:

        # My code starts here
        counter = 0
        for play in period['playbyplayStats']:
            # print(f"")
            
            if skip_counter:
                skip_counter = False
                continue

            play_list = play['eventDescription'].split()
            is_home_bool = bool(play['isHome'])
            is_home = 0 if is_home_bool else 1 # 0 if at home, 1 if away
            # clock_time_str = play['clock']
            clock_time_list = play['clock'].split(":")
            clock_time = (int(clock_time_list[0]) * 60) + int(clock_time_list[1]) # The total amount of seconds left on the clock
            clock_time /= 1200 # Normalizing the clock value to be between 0 and 1
            # Sometimes, the player's last name is concatenated with 'makes' or 'misses' in the event description, 
            # so we need to split those apart to get the player's name and the shot result as separate tokens.
            if play_list[1][-6:] == "misses":
                # split apart players last name and the 'misses' string
                play_list[1:2] = [play_list[1][:-6], 'misses']
            elif play_list[1][-5:] == "makes":
                # split apart players last name and the 'makes' string
                play_list[1:2] = [play_list[1][:-5], 'makes']

            player_name = play_list[0] + " " + play_list[1]


            # Check if the event description contains the keywords that indicate a shot event, and print it if found.
            # If not, skip to the next event. This is a quick way to filter for shot events without needing to parse 
            # every single event description in detail.
            if not (set(['three', 'two', 'point', 'shot']) & (set(play_list))):
                continue
            if (set(['blocks', 'layup', 'free', 'throw']) & (set(play_list))):
                continue
            # If the player name is one of the following, skip it because those are not actual players and will just 
            # add noise to our player-level stats.
            if player_name in ["TV timeout", "Subbing out", "Subbing in", "End of", "Team defensive", "Team offensive"]:
                continue

            '''if counter == 10:
                print(f"play: {play}")
                wefds'''
            if player_name not in player_stats:
                player_stats[player_name] = {'two_point_sequence': [], 'three_point_sequence': [], 'clock_time_sequence_two_point': [], 'clock_time_sequence_three_point': [], 'is_home_sequence_two_point': [], 'is_home_sequence_three_point': []} # 0 = make, 1 = miss

            
            # Record whether the shot was a make or miss (0 or 1) in the appropriate sequence list for the player. This is a simple way to 
            # track the player's shooting performance over time, and we can analyze these sequences later to identify hot hand patterns.
            if 'makes' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                # print(f"Found a made 3pt shot: {play}")
                player_stats[player_name]['three_point_sequence'].append(0)
                player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
            elif 'misses' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                # print(f"Found a missed 3pt shot: {play}")
                player_stats[player_name]['three_point_sequence'].append(1)
                player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
            elif 'makes' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                # print(f"Found a made 2pt shot: {play}")
                player_stats[player_name]['two_point_sequence'].append(0)
                player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
            elif 'misses' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                # print(f"Found a missed 2pt shot: {play}")
                player_stats[player_name]['two_point_sequence'].append(1)
                player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
            else:
                continue
            
            
            skip_counter = True # Set the skip counter to True so that the next event will be skipped, 
            # which should prevent us from processing the duplicate event in the play-by-play data.
            counter += 1

    return player_stats
    # My code ends here

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
    
    
    return [1 for _ in range(n)] # Example feature vector with three features (intercept, shot distance, time remaining)

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

    t_step_transition_probabilities(p, 4)
    occupancy_times(p, n=len(shot_sequence) - 1, initial_state=0)  # starting from Cold - 0 = cold, 1 = neutral, 2 = hot
    return

def t_step_transition_probabilities(p, t=1):
    np_p = np.array(p)
    p_t = np.linalg.matrix_power(np_p, t)
    p_t = list(p_t)
    print(f"Transition probabilities in {t} step(s)")
    for row in p_t:
        print(f"{row}")

    '''pi_t = [[0,0,0],[0,0,0],[0,0,0]]
    for next_s in range(3):
        for curr_s in range(3):'''


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

    # Sum P^t for t = 0, 1, ..., n
    # P^0 = I (the starting state counts as visit 1)
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
    # Fetch the season sample, then summarize the first game as a quick smoke test.
    pbp_data = get_pbp_data()
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
        game_stats = get_all_player_stats_of_game(game_info)

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
    # Run the script only when executed directly, not when imported as a module.
    main()