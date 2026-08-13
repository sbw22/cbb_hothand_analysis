import http.client
import json
import copy
from collections import defaultdict
from datetime import datetime, timedelta
from math import exp, log
import numpy as np
from scipy.stats import invwishart
from fetch_ncaa_data import NCAADataFetcher
import pickle
import os
from dash import Dash, html, dcc
import matplotlib.pyplot as plt
import hashlib
import csv
from log_functions import log_empirical_transitions_by_momentum, log_state_make_rates_by_momentum

# =============================================================================
# NOTATION REFERENCE
# =============================================================================
# β (beta)     : Global fixed-effect coefficient vectors, one per transition pair.
#                Shape: (NUM_FEATURES,) per transition.
#                Shared across all games — learned from data via MH.
#
# b_i          : Game-level random effect scalars, one per transition pair per game.
#                Captures how much game i deviates from the average transition tendency.
#                Jointly sampled as a 6-vector per game from N(0, Sigma_b).
#
# Sigma_b      : 6x6 covariance matrix over the 6 random effects per game.
#                Prior: Inverse-Wishart(nu_0, Psi_0). Updated each MCMC iteration.
#
# eta_i^(jk)  : Log-odds for transitioning from state j to state k in game i.
#                eta = Xi_n · beta^(jk) + b_i^(jk)
#
# gamma_j      : Probability of making a shot when in state j ∈ {C, N, H}.
#
# Z_in         : Hidden state at shot n of game i ∈ {0=C, 1=N, 2=H}.
#
# TRANSITION PAIRS (6 total for 3-state extension):
#   'CH': Cold  → Hot      'HC': Hot   → Cold
#   'CN': Cold  → Neutral  'NC': Neutral → Cold
#   'NH': Neutral → Hot    'HN': Hot   → Neutral
#
# Ordering used for b_i vectors: [CH, HC, CN, NC, NH, HN]
# =============================================================================

NUM_FEATURES = 6   # must match len(Xi_n): [clock_time, is_home, opp_def_3pt_pct_avg, three_point_game_num, three_point_momentum, three_point_intercept]
TRANSITIONS = ['CH', 'HC', 'CN', 'NC', 'NH', 'HN']

# State index constants

GAMMAS = [0.20, 0.32, 0.48]   # gamma_C, gamma_N, gamma_H
# STATE_SHOT_COUNTS = {0: 0, 1: 0, 2: 0}
# STATE_MADE_SHOT_COUNTS = {0: 0, 1: 0, 2: 0}

C_IDX, N_IDX, H_IDX = 0, 1, 2
STATE_NAMES = ['C', 'N', 'H']
T_IDX = {t: i for i, t in enumerate(TRANSITIONS)}   # e.g. 'CH' -> 0


def compute_model_version() -> str:
    """
    Fingerprint the model's structural assumptions. If any of these change
    (feature count, transition set/ordering, or their index mapping), a
    saved MCMC state is no longer valid to warm-start from — the beta
    vectors and b_i slots would silently misassign to the wrong quantities.
    """
    ALGO_VERSION = "belief-chaining-v5"
    
    fingerprint = f"{NUM_FEATURES}|{TRANSITIONS}|{sorted(T_IDX.items())}|{ALGO_VERSION}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

MODEL_VERSION = compute_model_version()


# =============================================================================
# GLOBAL MODEL PARAMETERS  (updated in-place during MCMC)
# =============================================================================

# β: global fixed-effect coefficients, one vector per transition pair.
# Initialized to zeros → all transition probabilities start at 1/3 each.
beta = {t: np.zeros(NUM_FEATURES) for t in TRANSITIONS}

# Transition prior means for [clock, home, opp_def, game_num, momentum, intercept]
# Favoring transitions into Neutral ('CN', 'HN') and staying in Neutral ('NC', 'NH' near zero or negative)
'''BETA_PRIOR_MEANS = {
    'CH': np.array([0.0, 0.0, 0.0, 0.0,  0.0, -0.25]),  # Cold -> Hot
    'CN': np.array([0.0, 0.0, 0.0, 0.0,  0.0, 0.5]),  # Cold -> Neutral
    'HC': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Hot -> Cold
    'HN': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25]),  # Hot -> Neutral
    'NC': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Neutral -> Cold
    'NH': np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.25]),  # Neutral -> Hot
}'''

BETA_PRIOR_MEANS = {t: np.zeros(NUM_FEATURES) for t in TRANSITIONS}
for t in TRANSITIONS:
    BETA_PRIOR_MEANS[t][5] = -1.5   # intercept: favor staying in current state

# Direct Cold<->Hot jumps: rare, strongly discouraged
BETA_PRIOR_MEANS['CH'][5] = -2.7
BETA_PRIOR_MEANS['HC'][5] = -2.3
# Routing through Neutral: the "normal" pathway, mildly discouraged (still persistent, but reachable)
BETA_PRIOR_MEANS['CN'][5] = -1.0
BETA_PRIOR_MEANS['NC'][5] = -0.8
BETA_PRIOR_MEANS['NH'][5] = -1.2
BETA_PRIOR_MEANS['HN'][5] = -1.0

# b_i storage: dict keyed by game_id, value = np.array of shape (6,)
# b_i[i][T_IDX['CH']] gives b_i^(CH) for game i.
b_i_store = {}   # populated lazily at MCMC init

# Sigma_b: 6x6 covariance matrix for the joint random effect vector per game.
# Prior: Inverse-Wishart(nu_0=8, Psi_0=I).  Start at identity.
Sigma_b = np.eye(6)

# =============================================================================
# HELPER: BUILD TRANSITION MATRIX FOR ONE SHOT
# =============================================================================

def build_transition_row(current_state: str, Xi_n: np.ndarray, game_id: int) -> np.ndarray:
    """
    Return the 3-element probability row [p_C, p_N, p_H] for transitioning
    out of current_state, using the current global beta and b_i for game_id.

    The reference category (η=0) is always "stay in current state", so
    only the two off-diagonal etas are computed. This ensures identifiability.
    """
    b_i = b_i_store.get(game_id, np.zeros(6)) # b_i: The game-level random effect scalar for the transition. IT HAS 6 ELEMENTS, ONE FOR EACH TRANSITION PAIR.

    if current_state == 'C':
        eta_H = Xi_n @ beta['CH'] + b_i[T_IDX['CH']]
        eta_N = Xi_n @ beta['CN'] + b_i[T_IDX['CN']]
        denom = 1.0 + exp(eta_H) + exp(eta_N)
        return np.array([1.0/denom, exp(eta_N)/denom, exp(eta_H)/denom])

    elif current_state == 'N':
        eta_C = Xi_n @ beta['NC'] + b_i[T_IDX['NC']]
        eta_H = Xi_n @ beta['NH'] + b_i[T_IDX['NH']]
        denom = 1.0 + exp(eta_C) + exp(eta_H)
        return np.array([exp(eta_C)/denom, 1.0/denom, exp(eta_H)/denom])

    elif current_state == 'H':
        eta_C = Xi_n @ beta['HC'] + b_i[T_IDX['HC']]
        eta_N = Xi_n @ beta['HN'] + b_i[T_IDX['HN']]
        denom = 1.0 + exp(eta_C) + exp(eta_N)
        return np.array([exp(eta_C)/denom, exp(eta_N)/denom, 1.0/denom])

    raise ValueError(f"Unknown state: {current_state}")


def build_transition_matrix(Xi_n: np.ndarray, game_id: int) -> np.ndarray:
    """
    Full 3x3 transition matrix P where P[j, k] = p(next=k | curr=j).
    Rows: C, N, H.  Columns: C, N, H.
    """
    P = np.zeros((3, 3))
    for j, s in enumerate(STATE_NAMES):
        P[j] = build_transition_row(s, Xi_n, game_id)
    return P


# =============================================================================
# MCMC STEP 1 — FORWARD-FILTERING BACKWARD-SAMPLING (FFBS)
# =============================================================================

def forward_pass(shot_sequence: list, Xi_sequence: np.ndarray,
                 game_id: int, gammas: list,
                 initial_dist: np.ndarray, iteration: int) -> list:
    """
    Run the HMM forward filter for one game.

    At each shot n, compute alpha_n = P(Z_n | Y_1,...,Y_n) (normalized).
    These are saved and used in the backward sampling step.

    Args:
        shot_sequence : list of ints, 0=made 1=missed
        Xi_sequence   : (M_i, NUM_FEATURES) array of shot features
        game_id       : index i
        gammas        : [gamma_C, gamma_N, gamma_H]
        initial_dist  : (3,) initial state distribution

    Returns:
        alphas : list of M_i normalized belief vectors, each shape (3,)
    """
    belief = initial_dist.copy()
    alphas = []

    for n, shot in enumerate(shot_sequence):
        made = (shot == 0)   # 0 = made, 1 = missed
        Xi_n = Xi_sequence[n]

        # --- Predict: propagate belief through transition matrix ---
        P = build_transition_matrix(Xi_n, game_id)
        
        predicted = belief @ P   # shape (3,)

        # --- Update: weight by emission likelihood ---
        likelihoods = np.array([
            gammas[s] if made else (1.0 - gammas[s]) for s in range(3)
        ])

        updated = likelihoods * predicted
        total = updated.sum()
        if total == 0:
            # Numerical guard: fall back to uniform
            updated = np.ones(3) / 3.0
        else:
            updated /= total

        belief = updated
        alphas.append(belief.copy())

        '''if iteration % 200 == 0:

            print("Transition matrix:\n", P)

            print("Predicted:", predicted)
            print("Likelihoods:", likelihoods)

            print("Updated:", updated)'''

    return alphas


def backward_sample(alphas: list, shot_sequence: list,
                    Xi_sequence: np.ndarray, game_id: int) -> np.ndarray:
    """
    Backward sampling pass: given the forward messages, sample one complete
    hidden state path Z = [Z_1, ..., Z_M] for the game.

    P(Z_{n-1} = j | Z_n = k, alpha_{n-1}) ∝ alpha_{n-1}[j] * P[j, k]

    Returns:
        Z : (M_i,) integer array of sampled states (0=C, 1=N, 2=H)
    """
    M = len(shot_sequence)
    Z = np.zeros(M, dtype=int)

    # Sample last state from final forward belief
    Z[M - 1] = np.random.choice(3, p=alphas[M - 1])

    # Go backward
    for n in range(M - 2, -1, -1):
        Xi_n = Xi_sequence[n + 1]   # features at the transition shot
        P = build_transition_matrix(Xi_n, game_id)

        # weights: alpha_{n}[j] * P[j, Z_{n+1}]
        weights = alphas[n] * P[:, Z[n + 1]]
        total = weights.sum()
        if total == 0:
            weights = np.ones(3) / 3.0
        else:
            weights /= total

        Z[n] = np.random.choice(3, p=weights)

    return Z


def sample_all_state_sequences(all_games: list, gammas: list,
                                initial_dist: np.ndarray, iteration: int) -> list:
    """
    Run FFBS for every game. Returns a list of state sequence arrays,
    one per game.

    all_games: list of dicts, each with keys:
        'shot_sequence'  : list of ints
        'Xi_sequence'    : (M_i, NUM_FEATURES) np.ndarray
        'game_id'        : int
    """
    state_sequences = [None] * len(all_games) # state_sequences[i] will be the state sequence for game i.

    state_total_shot_counts = {0: 0, 1: 0, 2: 0}
    state_made_shot_counts = {0: 0, 1: 0, 2: 0}

    games_by_player = defaultdict(list)

    for idx, game in enumerate(all_games):
        games_by_player[game['player_name']].append(idx)

    for player, idxs in games_by_player.items():

        belief_carry = initial_dist.copy()

        for idx in idxs:
            game = all_games[idx]
            game_id = game['game_id']
            shots = game['shot_sequence']
            Xi_seq = game['Xi_sequence']

            alphas = forward_pass(shots, Xi_seq, game_id, gammas, belief_carry, iteration)
            Z = backward_sample(alphas, shots, Xi_seq, game_id)
            state_sequences[idx] = Z

            belief_carry = alphas[-1]  # carry final belief into this player's next game

            for i, state in enumerate(Z): # Count the number of shots and made shots for each state.
                if state == 0:
                    state_total_shot_counts[0] += 1
                    if shots[i] == 0: # If the shot was made, increment the made shot count for the state.
                        state_made_shot_counts[0] += 1
                elif state == 1:
                    state_total_shot_counts[1] += 1
                    if shots[i] == 0:
                        state_made_shot_counts[1] += 1
                elif state == 2:
                    state_total_shot_counts[2] += 1
                    if shots[i] == 0:
                        state_made_shot_counts[2] += 1

        

    return state_sequences, state_total_shot_counts, state_made_shot_counts


# =============================================================================
# LOG-LIKELIHOOD HELPERS
# =============================================================================

def log_likelihood_transition( transition: str,
                               observations: list) -> float:
    """
    Log-likelihood contribution of one transition type (e.g. 'CH') across
    all observed transitions from the source state of that pair.

    observations: list of (Xi_n, did_transition) tuples where
        Xi_n           : (NUM_FEATURES,) feature vector
        did_transition : bool, True if the transition 'CH' actually occurred
    """
    ll = 0.0
    # src = transition[0]   # e.g. 'C' for 'CH'
    # print(f"src: {src}")

    src_idx = {'C': 0, 'N': 1, 'H': 2}[transition[0]]

    for observation in observations:
        # Compute eta for just this transition
        # eta = Xi_n @ beta_vec + b_i_scalar

        Xi_n = observation['Xi_n']
        # src_idx = observation['src_state']
        dst_idx = observation['dst_state']
        game_id = observation['game_id']

        row = build_transition_row(STATE_NAMES[src_idx], Xi_n, game_id) # row: [p_C, p_N, p_H] for the transition from src to dst.

        p = np.log(np.clip(row[dst_idx], 1e-10, 1.0-1e-10))
        # ll += log(p) if did_transition else log(1.0 - p)
        ll += p
    
    """
    The larger ll, the more likely the transition is to occur, and the better beta is at predicting the transition.
    """

    return ll



def log_prior_beta(beta_vec: np.ndarray, transition: str, prior_var: float = 0.05, intercept_prior_var: float = 0.02, momentum_prior_var: float = 0.01) -> float:
    """
    Log prior for one beta vector: N(0, prior_var * I).
    The paper doesn't specify β's prior exactly; N(0,1) per component is standard.
    """
    # prior_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
    # diff = beta_vec - prior_means
    diff = beta_vec - BETA_PRIOR_MEANS[transition]
    var_vec = np.full(NUM_FEATURES, prior_var)
    var_vec[5] = intercept_prior_var # This is how we are reducing the variance of the intercept.
    var_vec[4] = momentum_prior_var # This is how we are reducing the variance of the momentum/how much of an impact the momentum has on the transition.
    return -0.5 * np.sum((diff ** 2) / var_vec)
    """
    Initial pdf: p(x) = (2π sigma^2)^(-1/2) * exp(-x^2) / (2 sigma^2)). Return statement returns how tall the N(0, prior_var) is at the value of beta_vec (I think).
    Taking log of initial pdf gives the statement in return
    This distribution is a normal distribution with mean 0 and variance prior_var.
    Since it is a log prior, pdf densities range from 0 downwards. 
    """


# =============================================================================
# MCMC STEP 2 — METROPOLIS-HASTINGS UPDATE FOR β
# =============================================================================

def collect_transition_observations(transition: str,
                                     all_games: list,
                                     state_sequences: list) -> list:
    """
    From the current sampled state sequences, collect all (Xi_n, did_transition)
    pairs for a given transition type across all games.

    'CH' observations: any shot n where Z_n = C (source = Cold),
    did_transition = True if Z_{n+1} = H.
    """
    src_state = {'C': C_IDX, 'N': N_IDX, 'H': H_IDX}[transition[0]]
    # dst_state = {'C': C_IDX, 'N': N_IDX, 'H': H_IDX}[transition[1]]
    observations = []

    for game, Z in zip(all_games, state_sequences):
        Xi_seq = game['Xi_sequence']
        game_id = game['game_id']
        M = len(Z)
        for n in range(M - 1):
            if Z[n] == src_state:
            #    did = (Z[n + 1] == dst_state)
        
                observations.append({'game_id': game_id, 'Xi_n': Xi_seq[n], 'src_state': Z[n], 'dst_state': Z[n+1]})

    return observations


def mh_update_beta(src_state: str, all_games: list,
                   state_sequences: list, iteration: int, step_size: float = 0.1) -> bool:
    """
    One Metropolis-Hastings step for beta[transition].

    Proposal: beta_proposed = beta_current + N(0, step_size^2 * I)
    Accept/reject via log MH ratio:
        log r = log p(data | proposed) + log prior(proposed)
              - log p(data | current)  - log prior(current)

    Returns:
        accepted: bool, True if the proposal was accepted.
        step_size: float, the step size for the next iteration.
    """
    global beta

    # Map source state to its two off-diagonal targets
    pair_map = {
        'C': ('CN', 'CH'),
        'N': ('NC', 'NH'),
        'H': ('HN', 'HC'),
    }
    t1, t2 = pair_map[src_state]
    curr_t1, curr_t2 = beta[t1].copy(), beta[t2].copy()
    prop_t1 = curr_t1 + np.random.normal(0, step_size, size=NUM_FEATURES)
    prop_t2 = curr_t2 + np.random.normal(0, step_size, size=NUM_FEATURES)

    # Gather observations where this transition's source state was active
    # We pass b_i=0 here because b_i is accounted for separately per game.
    # For a more exact update you'd marginalise over b_i, but fixing at 0
    # when updating β is standard practice in this type of mixed model.
    observations = collect_transition_observations(t1, all_games, state_sequences)

    if len(observations) == 0:
        return False   # No data for this transition; skip

    ll_curr  = log_likelihood_transition(t1, observations)
    lp_curr  = log_prior_beta(curr_t1, t1) + log_prior_beta(curr_t2, t2)

    beta[t1], beta[t2] = prop_t1, prop_t2
    ll_prop = log_likelihood_transition(t1, observations)
    lp_prop = log_prior_beta(prop_t1, t1) + log_prior_beta(prop_t2, t2)


    log_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr)

    # the log of a uniform distribution is between 0 and -infinity. This follows the mh notes I have (I think), 
    # since ratios above 0 will then always be accepted, and ratio below 0 will sometimes be accepted.
    if log(np.random.uniform()) < log_ratio: 
        accepted = True
    else:
        beta[t1], beta[t2] = curr_t1, curr_t2
        accepted = False

    if iteration < 500:
        step_size = np.clip(step_size * (1.05 if accepted else step_size * 0.95), 1e-4, 0.5)

    return accepted, step_size


# =============================================================================
# MCMC STEP 3 — UPDATE b_i (JOINT 6-VECTOR PER GAME)
# =============================================================================

def log_likelihood_b_i(b_i_vec: np.ndarray, game: dict,
                        Z: np.ndarray, gammas: list) -> float:
    """
    Log-likelihood of the observed transitions in game i given b_i_vec.

    For each consecutive state pair (Z_n, Z_{n+1}), evaluate the softmax
    probability of the observed transition under the current beta and b_i_vec.
    """
    gid = game['game_id']
    Xi_seq = game['Xi_sequence']
    M = len(Z)
    ll = 0.0

    # Temporarily install b_i_vec so build_transition_row uses it
    old_b = b_i_store.get(gid, np.zeros(6))
    b_i_store[gid] = b_i_vec

    for n in range(M - 1):
        Xi_n = Xi_seq[n]
        src = STATE_NAMES[Z[n]] 
        dst = Z[n + 1]
        row = build_transition_row(src, Xi_n, gid)
        p = min(max(row[dst], 1e-10), 1.0) # row[dst]: How probable the transition is from src to dst.
        ll += log(p)

    # Restore
    b_i_store[gid] = old_b
    return ll


def log_prior_b_i(b_i_vec: np.ndarray) -> float:
    """
    Log prior for one game's random effect vector: N(0, Sigma_b).
    log p(b_i) = -0.5 * b_i^T Sigma_b^{-1} b_i  + const
    """
    Sigma_b_inv = np.linalg.inv(Sigma_b) # Sigma_b: 6x6 matrix, diagonal values are variances of each transition. Non-diagonal values = covariances between transitions, which means how much the transitions are correlated.
    # Sigma_b_inv: doing same job as 1/prior_var in log_prior_beta. Both are strength-penalty terms.
    return -0.5 * b_i_vec @ Sigma_b_inv @ b_i_vec # Same math as log_prior_beta, just with Sigma_b_inv instead of 1 / prior_var.
    """
    pdf in log form: p(x) = log p(x) = -0.5*log((2π)^k |Σ|)  -  0.5 * x^T Σ^-1 x
    We are dropping the constant term -0.5*log((2π)^k |Σ|) because it is not dependent on x, and it (mainly Sigma_b) doesn't change between when we calculate the log prior for the current and proposed values.
    |Σ|: determinant of the matrix.
    """


def mh_update_b_i(game: dict, Z: np.ndarray,
                   gammas: list, step_size: float = 0.1) -> bool:
    """
    One MH step for the 6-vector b_i for a single game.

    The joint proposal perturbs all 6 random effects simultaneously,
    which respects their correlation structure encoded in Sigma_b.
    """
    gid = game['game_id']
    current = b_i_store.get(gid, np.zeros(6)).copy()

    # Propose from multivariate normal centered on current value
    proposed = current + np.random.multivariate_normal(np.zeros(6), step_size**2 * Sigma_b) # Sigma_b makes the pattern of proposed (ex CN goes up when CH goes up) more likely to be close to Sigma_b's. 

    ll_current  = log_likelihood_b_i(current, game, Z, gammas)
    ll_proposed = log_likelihood_b_i(proposed, game, Z, gammas)
    lp_current  = log_prior_b_i(current)
    lp_proposed = log_prior_b_i(proposed)

    log_ratio = (ll_proposed + lp_proposed) - (ll_current + lp_current)

    if log(np.random.uniform()) < log_ratio:
        b_i_store[gid] = proposed
        return True
    else:
        b_i_store[gid] = current
        return False


# =============================================================================
# MCMC STEP 4 — UPDATE Sigma_b (INVERSE-WISHART CONJUGATE DRAW)
# =============================================================================

def update_Sigma_b(game_ids: list, nu_0: int = 8) -> None:
    """
    Conjugate Inverse-Wishart update for Sigma_b.

    Prior:  Sigma_b ~ IW(nu_0, Psi_0)   where Psi_0 = I (identity)
    Posterior: Sigma_b | {b_i} ~ IW(nu_0 + N, Psi_0 + sum_i b_i b_i^T)

    This is the exact conjugate posterior — no MH needed here.

    nu_0: prior degrees of freedom. Must be > 6 (dimension) for a proper prior.
          8 is a weakly informative default.
    """
    global Sigma_b

    N = len(game_ids)
    Psi_0 = np.eye(6)   # prior scale matrix (identity = weakly informative)

    # Accumulate outer products of b_i vectors
    B_matrix = np.zeros((6, 6))
    for gid in game_ids:
        b = b_i_store.get(gid, np.zeros(6))
        B_matrix += np.outer(b, b)

    nu_post  = nu_0 + N
    Psi_post = Psi_0 + B_matrix

    Sigma_b = invwishart.rvs(df=nu_post, scale=Psi_post)


def update_gammas_constrained(state_totals: dict, state_mades: dict) -> list:
    # Prior hyperparameters (alpha, beta) for (Cold, Neutral, Hot)
    priors = [(31, 46), (20, 26), (25, 25)]
    # priors = [(1, 3), (12, 13), (1, 3)]
    delta = 0.05

    while True:
    
        gammas = [0.0, 0.0, 0.0]
        for s in range(3):
            a = priors[s][0] + state_mades[s]
            b = priors[s][1] + (state_totals[s] - state_mades[s])
            gammas[s] = np.random.beta(a, b)
        

        if gammas[0] + delta < gammas[1] and gammas[1] + delta < gammas[2]:
            return gammas

    
    # Enforce minimum separation gap delta
    gammas.sort()
    gammas[1] = max(gammas[1], gammas[0] + delta)
    gammas[2] = max(gammas[2], gammas[1] + delta)
    
    # Clip to valid probability bounds
    return np.clip(gammas, 0.05, 0.95).tolist()


# =============================================================================
# MAIN MCMC LOOP
# =============================================================================

def run_mcmc(all_games: list,
             gammas: list,
             n_iter: int = 1000,
             warmup: int = 200,
             beta_step: float = 0.1,
             b_i_step: float = 0.1,
             initial_dist: np.ndarray = None,
             verbose: bool = True,
             init_from: dict = None) -> dict:
    """
    Full Gibbs sampler for the BLHMM.

    Each iteration:
      1. FFBS: sample hidden state path Z_i for every game
      2. MH:   update each β^(jk) using the sampled state paths
      3. MH:   update each b_i vector using the sampled state path for game i
      4. IW:   update Sigma_b conjugately from all b_i vectors

    Args:
        all_games    : list of game dicts (see sample_all_state_sequences)
        gammas       : [gamma_C, gamma_N, gamma_H]
        n_iter       : total MCMC iterations (including warmup)
        warmup       : number of burn-in iterations to discard
        beta_step    : MH proposal std for β updates
        b_i_step     : MH proposal scale for b_i updates
        initial_dist : (3,) initial state distribution. Defaults to uniform.
        verbose      : print progress every 100 iterations

    Returns:
        samples: dict with keys 'beta', 'b_i', 'Sigma_b', 'acceptance'
            beta    : list of beta dicts (one per post-warmup iteration)
            b_i     : list of b_i_store snapshots
            Sigma_b : list of Sigma_b arrays
            acceptance: dict tracking MH acceptance rates
    """
    global beta, b_i_store, Sigma_b

    if initial_dist is None:
        initial_dist = np.array([1/3, 1/3, 1/3])

    game_ids = [g['game_id'] for g in all_games]

    can_warm_start = (
        init_from is not None
        and init_from.get('beta')
        and init_from.get('b_i')
        and init_from.get('Sigma_b')
        and init_from.get('model_version') == MODEL_VERSION
    )

    if init_from is not None and not can_warm_start:
        reason = (
            f"model_version mismatch "
            f"(saved={init_from.get('model_version')!r}, current={MODEL_VERSION!r})"
            if init_from.get('beta') else "incomplete saved state"
        )
        if verbose:
            print(f"Cannot start up, {reason}. Falling back to cold start.")
    
    if can_warm_start:
        # --- Warm start from a previous run's final draws ---
        prev_beta = init_from['beta'][-1]
        prev_b_i = init_from['b_i'][-1]
        prev_Sigma_b = init_from['Sigma_b'][-1]

        for t in TRANSITIONS:
            beta[t] = prev_beta[t].copy()
        
        for gid in game_ids:
            # .get() handles any new game_ids not seen in the previous run.
            b_i_store[gid] = prev_b_i.get(gid, np.zeros(6))
        
        Sigma_b = prev_Sigma_b.copy()
        step_sizes = {t: init_from['step_sizes'][t] for t in TRANSITIONS}
        # for t in ['CH', 'HC', 'CN', 'NC', 'HN', 'NH']:
        #     step_sizes[t] = 0.05   # give them room to actually move again after warm start

        if verbose:
            print(f"Warm-starting from previous run's final state "
                  f"({len(init_from['beta'])} prior post-warmup draws, "
                  f"model_version={MODEL_VERSION}). "
                  f"Step sizes: { {t: round(s, 4) for t,s in step_sizes.items()}}")
    else:
        # Cold Start
        for gid in game_ids:
            b_i_store[gid] = np.zeros(6)
        for t in TRANSITIONS:
            beta[t] = np.random.normal(0, 0.01, size=NUM_FEATURES)
            beta[t][5] = -1.5  #######################################################################          <<<<<<<<<<<<<<<<<<<<<<<< FIX?? I don't think it hurts to keep this here, idk though.
        step_sizes = {t: beta_step for t in TRANSITIONS}
        if verbose and init_from is None:
            print(f"Cold-starting MCMC (no init_from provided)")



    recent_accepts = {t: 0 for t in TRANSITIONS}
    target_accept_rate = 0.25
    adapt_interval = 50

    # Tracking
    samples = {'beta': [], 'b_i': [], 'Sigma_b': [], 'gammas': [], 'acceptance': {}}
    trace_keys = [('CH', 0), ('CH', 1), ('CH', 2), ('CH', 3), ('HC', 0), ('HC', 1), ('HC', 2), ('HC', 3), ('CN', 0), ('CN', 1), ('CN', 2), ('CN', 3), ('NC', 0), ('NC', 1), ('NC', 2), ('NC', 3), ('HN', 0), ('HN', 1), ('HN', 2), ('HN', 3), ('NH', 0), ('NH', 1), ('NH', 2), ('NH', 3)]
    trace = {f"{t}_{f}": [] for t, f in trace_keys}
    accept_counts = {t: 0 for t in TRANSITIONS}
    accept_counts['b_i'] = 0
    total_b_i_proposals = 0

    # -------------------------------------------------------------------------
    # Initial state sequences (random draw before any MCMC)
    # -------------------------------------------------------------------------
    state_sequences = [
        np.random.choice(3, size=len(g['shot_sequence'])) for g in all_games
    ]

    for iteration in range(n_iter):

        # --- Step 1: Sample hidden state paths via FFBS ---
        state_sequences, state_total_shot_counts, state_made_shot_counts = sample_all_state_sequences(all_games, gammas, initial_dist, iteration)


        # ================================================================
        # DIAGNOSTIC: Examine sampled transitions by momentum
        # ================================================================
        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
            log_empirical_transitions_by_momentum(
                all_games,
                state_sequences
            )
            log_state_make_rates_by_momentum(
                all_games,
                state_sequences
            )

        gammas = list(update_gammas_constrained(state_total_shot_counts, state_made_shot_counts))

        gamma_c, gamma_n, gamma_h = gammas


        if iteration % 100 == 0:
            print(
                f"Gammas at iteration {iteration}: ",
                f"C={gamma_c:.3f} ",
                f"N={gamma_n:.3f} ",
                f"H={gamma_h:.3f}",
            )

            counts = np.bincount(
                np.concatenate(state_sequences),
                minlength=3
            )

            print(f"Iteration {iteration}")
            print("State counts:", counts)
            print("State proportions:", counts / counts.sum())

            for t in TRANSITIONS:
                print(t, np.round(beta[t], 3))

        # --- Step 2: Update β for each transition pair ---
        # for t in TRANSITIONS:
        pair_map = {
            'C': ('CN', 'CH'),
            'N': ('NC', 'NH'),
            'H': ('HN', 'HC'),
        }
        for src_state in ['C', 'N', 'H']:
            t1, t2 = pair_map[src_state]
            curr_step = max(step_sizes[t1], step_sizes[t2])
            accepted, step_size = mh_update_beta(src_state, all_games, state_sequences, iteration, step_size=curr_step)
            if accepted:
                accept_counts[t1] += 1
                accept_counts[t2] += 1
                recent_accepts[t1] += 1
                recent_accepts[t2] += 1
            
        
        # --- Record trace values for convergence diagnostics ---
        for t, f in trace_keys:
            trace[f"{t}_{f}"].append(beta[t][f])


        # --- Step 3: Update b_i for each game ---
        for game, Z in zip(all_games, state_sequences):
            accepted = mh_update_b_i(game, Z, gammas, step_size=b_i_step)
            if accepted:
                accept_counts['b_i'] += 1
            total_b_i_proposals += 1

        # --- Step 4: Update Sigma_b (conjugate IW draw) ---
        update_Sigma_b(game_ids)

        # --- Collect post-warmup samples ---
        if iteration >= warmup:
            samples['beta'].append({t: beta[t].copy() for t in TRANSITIONS})
            samples['b_i'].append({gid: b_i_store[gid].copy() for gid in game_ids})
            samples['Sigma_b'].append(Sigma_b.copy())
            samples['gammas'].append(list(gammas.copy()))
        if verbose and (iteration + 1) % 100 == 0:
            beta_rates = {t: accept_counts[t] / (iteration + 1) for t in TRANSITIONS}
            b_i_rate = accept_counts['b_i'] / max(total_b_i_proposals, 1)
            print(f"Iter {iteration + 1}/{n_iter} | "
                  f"β accept rates: { {t: f'{r:.2f}' for t, r in beta_rates.items()} } | "
                  f"b_i accept rate: {b_i_rate:.2f}")
        
        if iteration < warmup and (iteration + 1) % adapt_interval == 0:
            for t in TRANSITIONS:
                rate = recent_accepts[t] / adapt_interval
                if rate < target_accept_rate:
                    step_sizes[t] *= 0.85
                else:
                    step_sizes[t] *= 1.15
                step_sizes[t] = np.clip(step_sizes[t], 1e-4, 2.0)
                recent_accepts[t] = 0
            if verbose:
                print(f"Adapted step sizes at iter {iteration + 1} "
                f"{ {t: round(s, 4) for t,s in step_sizes.items()} }")

            

    samples['acceptance'] = {
        t: accept_counts[t] / n_iter for t in TRANSITIONS
    }
    samples['acceptance']['b_i'] = accept_counts['b_i'] / max(total_b_i_proposals, 1)
    samples['step_sizes'] = step_sizes
    samples['model_version'] = MODEL_VERSION # <-- tag every saved run
    samples['trace'] = trace # trace is a dictionary with keys like 'CH_0', 'CH_1', 'CH_2', 'CH_3', 'HC_0', 'HC_1', 'HC_2', 'HC_3', 'CN_0', 'CN_1', 'CN_2', 'CN_3', 'NC_0', 'NC_1', 'NC_2', 'NC_3', 'HN_0', 'HN_1', 'HN_2', 'HN_3', 'NH_0', 'NH_1', 'NH_2', 'NH_3'

    return samples, gammas


def summarize_mcmc(samples: dict) -> None:
    """
    Print posterior means and 95% credible intervals for β and Sigma_b
    from the post-warmup samples.
    """
    print("\n=== MCMC Posterior Summary ===\n")

    beta_samples = samples['beta']   # list of dicts
    if len(beta_samples) == 0:
        print("No post-warmup samples collected.")
        return

    print("β posterior means (95% CI):")
    for t in TRANSITIONS:
        draws = np.array([s[t] for s in beta_samples])   # (n_samples, NUM_FEATURES)
        for f in range(NUM_FEATURES):
            mean = draws[:, f].mean()
            lo, hi = np.percentile(draws[:, f], [2.5, 97.5])
            print(f"  beta[{t}][feature {f}]:  mean={mean:.3f}  95%CI=({lo:.3f}, {hi:.3f})")
    
    print("\nGamma posterior means (95% CI):")
    gamma_draws = np.array(samples['gammas'])  # (n_samples, 3)
    for i, name in enumerate(['C', 'N', 'H']):
        mean = gamma_draws[:, i].mean()
        lo, hi = np.percentile(gamma_draws[:, i], [2.5, 97.5])
        print(f"  gamma_{name}: mean={mean:.3f}  95%CI=({lo:.3f}, {hi:.3f})")

    print("\nSigma_b posterior mean (diagonal — per-transition variance):")
    Sigma_draws = np.array(samples['Sigma_b'])   # (n_samples, 6, 6)
    Sigma_mean = Sigma_draws.mean(axis=0)
    for i, t in enumerate(TRANSITIONS):
        print(f"  Var(b_i^{t}): {Sigma_mean[i, i]:.3f}")

    print("\nMH Acceptance rates:")
    for key, rate in samples['acceptance'].items():
        print(f"  {key}: {rate:.3f}")


def canonicalize_samples(samples: dict) -> dict:
    """
    Fixes label-switching: for each saved MCMC sample, relabel states so
    that state 0 = lowest gamma ("Cold"), state 1 = middle ("Neutral"),
    state 2 = highest ("Hot"), consistently across beta, b_i, and gammas.
    Without this, averaging samples['beta'] or samples['b_i'] across
    iterations mixes together contradictory labelings and washes out
    the signal.
    """
    canon_beta = []
    canon_b_i = []
    canon_gammas = []

    for beta_sample, b_i_sample, gammas_sample in zip(
        samples['beta'], samples['b_i'], samples['gammas']
    ):
        # order[canonical_rank] = which original state index holds that rank
        order = np.argsort(gammas_sample)
        # rank[original_index] = canonical_rank it maps to
        rank = np.empty(3, dtype=int)
        for canonical_rank, orig_idx in enumerate(order):
            rank[orig_idx] = canonical_rank

        canon_gammas.append([gammas_sample[i] for i in order])

        new_beta = {}
        for t in TRANSITIONS:
            orig_src = STATE_NAMES.index(t[0])
            orig_dst = STATE_NAMES.index(t[1])
            new_t = STATE_NAMES[rank[orig_src]] + STATE_NAMES[rank[orig_dst]]
            new_beta[new_t] = beta_sample[t]
        canon_beta.append(new_beta)

        new_b_i = {}
        for gid, vec in b_i_sample.items():
            new_vec = np.zeros(6)
            for t in TRANSITIONS:
                orig_src = STATE_NAMES.index(t[0])
                orig_dst = STATE_NAMES.index(t[1])
                new_t = STATE_NAMES[rank[orig_src]] + STATE_NAMES[rank[orig_dst]]
                new_vec[T_IDX[new_t]] = vec[T_IDX[t]]
            new_b_i[gid] = new_vec
        canon_b_i.append(new_b_i)

    canonicalized = dict(samples)  # shallow copy; keeps acceptance/trace/etc as-is
    canonicalized['beta'] = canon_beta
    canonicalized['b_i'] = canon_b_i
    canonicalized['gammas'] = canon_gammas
    return canonicalized


def apply_mcmc_samples_to_globals(samples: dict, method: str = 'last') -> bool:
    """
    Copy saved MCMC chains into module-level beta, b_i_store, and Sigma_b.

    samples['beta']    : list of {transition: vector} dicts (one per saved iteration)
    samples['b_i']     : list of {game_id: vector} dicts
    samples['Sigma_b'] : list of 6x6 matrices

    method='mean' uses posterior means; method='last' uses the final saved draw.
    """
    global beta, b_i_store, Sigma_b

    beta_samples = samples.get('beta', [])
    b_i_samples = samples.get('b_i', [])
    sigma_samples = samples.get('Sigma_b', [])

    if not beta_samples or not b_i_samples or not sigma_samples:
        print(
            "Cannot apply MCMC samples: no post-warmup draws saved "
            f"(beta={len(beta_samples)}, b_i={len(b_i_samples)}, "
            f"Sigma_b={len(sigma_samples)}). "
            "Check that n_iter > warmup."
        )
        return False

    if method == 'last':
        beta_draw = beta_samples[-1]
        b_i_draw = b_i_samples[-1]
        for t in TRANSITIONS:
            beta[t] = beta_draw[t].copy()
        b_i_store = {gid: vec.copy() for gid, vec in b_i_draw.items()}
        Sigma_b = sigma_samples[-1].copy()
    else:
        for t in TRANSITIONS:
            beta[t] = np.mean([s[t] for s in beta_samples], axis=0)
        all_gids = set().union(*(snap.keys() for snap in b_i_samples))
        b_i_store = {
            gid: np.mean([snap[gid] for snap in b_i_samples if gid in snap], axis=0)
            for gid in all_gids
        }
        Sigma_b = np.mean(sigma_samples, axis=0)

    print(f"Applied MCMC samples to globals ({method}, {len(beta_samples)} draws).")
    return True


# =============================================================================
# UTILITY: BUILD all_games STRUCTURE FROM RAW SEQUENCES
# =============================================================================

def build_game_dict(game_id: int, player_name: str, shot_sequence: list,
                    clock_sequence: list, is_home_sequence: list, opp_def_3pt_pct_avg: list, three_point_game_num: list, three_point_momentum: list, three_point_intercept: list) -> dict:
    """
    Pack raw sequences for one game into the dict format expected by MCMC.

    Xi_sequence columns: [clock_time, is_home]
    """

    # Example Feature Standardization before building Xi_seq:
    clock_std = (clock_sequence - np.mean(clock_sequence)) / (np.std(clock_sequence) + 1e-8)
    opp_def_std = (opp_def_3pt_pct_avg - np.mean(opp_def_3pt_pct_avg)) / (np.std(opp_def_3pt_pct_avg) + 1e-8)
    game_num_std = (three_point_game_num - np.mean(three_point_game_num)) / (np.std(three_point_game_num) + 1e-8)
    three_point_momentum_std = (three_point_momentum - np.mean(three_point_momentum)) / (np.std(three_point_momentum) + 1e-8)
    is_home_sequence_std = (is_home_sequence - np.mean(is_home_sequence)) / (np.std(is_home_sequence) + 1e-8)

    # print(f"clock_std: {clock_std}")
    # print(f"opp_def_std: {opp_def_std}")
    # print(f"game_num_std: {game_num_std}")
    # print(f"three_point_intercept: {three_point_intercept}")

    # erdgf

    M = len(shot_sequence)
    Xi_seq = np.column_stack([
        np.array(clock_std[:M], dtype=float),
        np.array(is_home_sequence_std[:M], dtype=float),
        np.array(opp_def_std[:M], dtype=float),
        np.array(game_num_std[:M], dtype=float),
        np.array(three_point_momentum_std[:M], dtype=float),
        np.array(three_point_intercept[:M], dtype=float)
    ])
    return {
        'game_id': game_id,
        'player_name': player_name,
        'shot_sequence': shot_sequence,
        'Xi_sequence': Xi_seq,
    }


def plot_trace(samples: dict, warmup: int = None) -> None:
    """
    Plot the per-iteration trace of the tracked beta components.
    A converged chain should look like it's oscillating around a stable
    level by the end of the run, not still trending in one direction.

    warmup: if given, draws a vertical line marking where warmup ended,
            so you can see whether the chain looks stable *after* that point.
    """
    trace = samples.get('trace', {})
    if not trace:
        print("No trace data found — did you add trace tracking to run_mcmc()?")
        return

    n_keys = len(trace)
    fig, axes = plt.subplots(n_keys, 1, figsize=(10, 2.5 * n_keys), sharex=True)
    if n_keys == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, trace.items()):
        ax.plot(values, linewidth=0.7)
        ax.set_ylabel(key)
        ax.axhline(sum(values[-500:]) / len(values[-500:]), color='red',
                   linestyle='--', linewidth=0.8, label='mean of last 500')
        if warmup is not None:
            ax.axvline(warmup, color='gray', linestyle=':', linewidth=0.8, label='warmup end')
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Iteration')
    fig.suptitle('MCMC Trace Plots — checking for convergence')
    plt.tight_layout()
    plt.savefig('mcmc_trace2.png', dpi=120)
    print("Saved trace plot to mcmc_trace2.png")
    plt.close()



# =============================================================================
# MAIN
# =============================================================================

def main():
    from indv_player_metrics import IndvPlayerMetrics
    # print(f"Entered main function")
    global GAMMAS

    in_prod = True # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<------------------------ CHANGE THIS TO TRUE WHEN RUNNING IN PRODUCTION

    indv_player_metrics = IndvPlayerMetrics()
    ncaa_data_fetcher = NCAADataFetcher()


    appended_game_stats = []
    extended_game_stats = {}
    all_games_for_mcmc = []

    samples = None


    # If the file all_games_for_mcmc.pkl exists, load it and skip running the MCMC
    if os.path.exists('stats_and_samples/mcmc_samples.pkl') and os.path.exists('stats_and_samples/extended_game_stats.pkl') and os.path.exists('stats_and_samples/appended_game_stats.pkl'): # and os.path.exists('stats_and_samples/all_games_for_mcmc.pkl'):
        with open('stats_and_samples/mcmc_samples.pkl', 'rb') as f:
            loaded_samples = pickle.load(f)
        print(f"Loaded MCMC samples with {len(loaded_samples.get('beta', []))} post-warmup draws"
              f"(Model_version={loaded_samples.get('model_version', 'unknown')}, "
              f"current={MODEL_VERSION})")
        with open('stats_and_samples/extended_game_stats.pkl', 'rb') as f:
            extended_game_stats = pickle.load(f)
        print(f"Loaded {len(extended_game_stats)} extended game stats from extended_game_stats.pkl")
        with open('stats_and_samples/appended_game_stats.pkl', 'rb') as f:
            appended_game_stats = pickle.load(f)
        print(f"Loaded {len(appended_game_stats)} appended game stats from appended_game_stats.pkl")
        with open('stats_and_samples/all_games_for_mcmc.pkl', 'rb') as f:
            all_games_for_mcmc = pickle.load(f)
        print(f"Loaded {len(all_games_for_mcmc)} all games for MCMC from all_games_for_mcmc.pkl")
    

        # samples['beta'] = [samples['beta'][0]]
        # samples['b_i'] = [samples['b_i'][0]]
        # samples['Sigma_b'] = [samples['Sigma_b'][0]]

        '''with open('stats_and_samples/mcmc_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)'''

        
        # We are saving the all_games_for_mcmc list to a file for now, so we don't need to run the MCMC again to test the model.

        # print(f"Running MCMC when all_games_for_mcmc.pkl exists")
        if not in_prod:
            samples, gammas = run_mcmc(
                all_games=all_games_for_mcmc,
                gammas=GAMMAS,
                n_iter=1000,
                warmup=600,
                beta_step=0.5,
                b_i_step=0.5,
                verbose=True,
                init_from=loaded_samples
            )
        else:
            samples = loaded_samples
            with open('stats_and_samples/gammas.pkl', 'rb') as f:
                gammas = pickle.load(f)
                GAMMAS = gammas
        print(f"Loaded gammas from gammas.pkl")
        samples = canonicalize_samples(samples)

        summarize_mcmc(samples)
        # plot_trace(samples, warmup=5)
        apply_mcmc_samples_to_globals(samples, method='mean')

        if not in_prod:
            # Save so the *next* dev-loop run can warm-start from this one
            with open('stats_and_samples/mcmc_samples.pkl', 'wb') as f:
                pickle.dump(samples, f)
            print(f"Saved samples to mcmc_samples.pkl")
            
            # Save gammas to a file
            with open('stats_and_samples/gammas.pkl', 'wb') as f:
                pickle.dump(gammas, f)
            print(f"Saved gammas to gammas.pkl")

    else:
        # If the files don't exist, run the MCMC
        print("No stats_and_samples/mcmc_samples.pkl, stats_and_samples/extended_game_stats.pkl, or stats_and_samples/appended_game_stats.pkl file found. Running MCMC.")

        pbp_data, boxscore_data = ncaa_data_fetcher.get_pbp_data()
        # print(f"Finished getting play-by-play data for {len(pbp_data)} games.")

        # print(f"Inspecting the first game's play-by-play data:")
        # print(f"pbp_data[0]['periods'][0]['playbyplayStats'] has "
        #     f"{len(pbp_data[0]['periods'][0]['playbyplayStats'])} events, with keys "
        #     f"{pbp_data[0]['periods'][0]['playbyplayStats'][23]['homeText']}\n")

        for pbp_data, boxscore_data in zip(pbp_data, boxscore_data):
            game_stats = ncaa_data_fetcher.get_all_player_stats_of_game(pbp_data, boxscore_data)
            # re9fjigdknm
            appended_game_stats.append(game_stats)

            for player, stats in game_stats.items():
                if player not in extended_game_stats:
                    extended_game_stats[player] = copy.deepcopy(stats)
                    extended_game_stats[player]['team_name'] = stats['team_name']
                    
                    continue

                # FIX
                extended_game_stats[player]['two_point_sequence'].extend(stats['two_point_sequence'])
                extended_game_stats[player]['three_point_sequence'].extend(stats['three_point_sequence'])
                extended_game_stats[player]['clock_time_sequence_two_point'].extend(stats['clock_time_sequence_two_point'])
                extended_game_stats[player]['clock_time_sequence_three_point'].extend(stats['clock_time_sequence_three_point'])
                extended_game_stats[player]['is_home_sequence_two_point'].extend(stats['is_home_sequence_two_point'])
                extended_game_stats[player]['is_home_sequence_three_point'].extend(stats['is_home_sequence_three_point'])
                extended_game_stats[player]['opp_def_3pt_pct_avg'].extend(stats['opp_def_3pt_pct_avg'])
                # extended_game_stats[player]['three_point_game_num'].extend(stats['three_point_game_num'])
                extended_game_stats[player]['three_point_intercept'].extend(stats['three_point_intercept'])
        
        # For each player, we need to create a list of game numbers, with each game number representing the game that player was in at that time.
        # Stats['three_point_game_num'] or stats['three_point_game_num] does not exist, so we need to create it.
        # We will use the game_id to create the list of game numbers.

        # ------------------------------------------------------------------
        # Build all_games list for MCMC
        # Each entry is one player-game observation unit.
        # Here we run across all games for the test player, treating each
        # game as a separate observation unit (as the paper intends).
        # ------------------------------------------------------------------

        all_games_for_mcmc = []
        raw_game_nums_by_player = {}
        skip_players = []

        for game_id, game_stats in enumerate(appended_game_stats):
            
            for player, player_data in game_stats.items():
                # temp = player_data['opp_def_3pt_pct_avg'][0]
                shots = player_data['three_point_sequence']
                clocks = player_data['clock_time_sequence_three_point']
                home = player_data['is_home_sequence_three_point']
                opp_def_3pt_pct_avg = player_data['opp_def_3pt_pct_avg']
                three_point_intercept = player_data['three_point_intercept']

                if len(shots) < 1:
                    skip_players.append(player)
                    continue   # need at least 2 shots for a transition
   

                raw_game_nums_by_player.setdefault(player, [])
                game_num = len(raw_game_nums_by_player[player]) + 1
                raw_game_nums_by_player[player].append([game_num, game_id])

                extended_game_stats[player]['three_point_game_num'].extend([game_num] * len(shots))
        


        def compute_ewma_momentum(shot_sequence: list, alpha: float = 0.4):
            momentum = []
            current = 0.35 # Initialize at prior mean (don't have one at the moment)
            for shot in shot_sequence:
                momentum.append(round(current, 3))
                val = 1.0 if shot == 0 else 0.0
                current = alpha * val + (1 - alpha) * current
            return momentum
                
        
        for player in extended_game_stats:
            if len(extended_game_stats[player]['three_point_game_num']) == 0:
                skip_players.append(player)
                continue
            # re-normalize the three_point_game_num to be between 0 and 1
            max_three_point_game_num = max(extended_game_stats[player]['three_point_game_num'])
            extended_game_stats[player]['three_point_game_num'] = [x / max_three_point_game_num for x in extended_game_stats[player]['three_point_game_num']]

            three_point_sequence = extended_game_stats[player]['three_point_sequence']
            three_point_momentum = compute_ewma_momentum(three_point_sequence)  # Compute the Exponentially Weighted Moving Average (EWMA) momentum
            extended_game_stats[player]['three_point_momentum'] = three_point_momentum

        temp_player_momentum_counter = dict.fromkeys(extended_game_stats.keys(), 0)

        
        for game_id, game_stats in enumerate(appended_game_stats):
            for player, player_data in game_stats.items():

                # add three_point_game_num to appended_game_stats
                shots = player_data['three_point_sequence']
                if len(shots) < 1:
                    continue
                raw_game_num = [shot_list[0] for shot_list in raw_game_nums_by_player[player] if shot_list[1] == game_id][0]
                max_raw_game_num = max([game_list[0] for game_list in raw_game_nums_by_player[player]])
                normalized_game_num = raw_game_num / max_raw_game_num
                # normalized_game_num_list = [num for num in extended_game_stats[player]['three_point_game_num'] if num == normalized_game_num]
                normalized_game_num_list = [normalized_game_num] * len(shots)

                appended_game_stats[game_id][player]['three_point_game_num'] = normalized_game_num_list

                full_three_point_momentum = extended_game_stats[player]['three_point_momentum']

                seq = []
                try:
                    seq = full_three_point_momentum[temp_player_momentum_counter[player]:temp_player_momentum_counter[player] + len(shots)]
                except:
                    seq = full_three_point_momentum[temp_player_momentum_counter[player]:]


                appended_game_stats[game_id][player]['three_point_momentum'] = seq

                appended_game_stats[game_id][player]['three_point_momentum'] = seq

                '''print(f"\nthree_point_momentum: {seq}")
                print(f"player_data['three_point_momentum']: {player_data['three_point_momentum']}")
                print(f"shots: {shots}")
                print(f"len of shots: {len(shots)}, len of full_three_point_momentum: {len(full_three_point_momentum)}, len of seq: {len(seq)}, temp_player_momentum_counter[player]: {temp_player_momentum_counter[player]}, len of appended_game_stats[game_id][player]['three_point_sequence']: {len(appended_game_stats[game_id][player]['three_point_sequence'])}")
                print(f"player: {player}")
                print(f"appended_game_stats: {appended_game_stats[game_id][player]}")'''

                # shots = player_data['three_point_sequence']
                clocks = player_data['clock_time_sequence_three_point']
                home = player_data['is_home_sequence_three_point']
                opp_def_3pt_pct_avg = player_data['opp_def_3pt_pct_avg']
                three_point_game_num = player_data['three_point_game_num']
                three_point_intercept = player_data['three_point_intercept']
                three_point_momentum = player_data['three_point_momentum']
                game_dict = build_game_dict(game_id, player, shots, clocks, home, opp_def_3pt_pct_avg, three_point_game_num, three_point_momentum, three_point_intercept)
                all_games_for_mcmc.append(game_dict)
                temp_player_momentum_counter[player] += len(shots)

                # erdihsfjn
                # continue

        print(f"\nRunning MCMC on {len(all_games_for_mcmc)} games for all players.")
        '''
        if len(all_games_for_mcmc) == 0:
            print("No games found with enough shots. Running single-game diagnostic instead.")
            # Fall back to the single-game forward filter for smoke testing
            test_player = extended_game_stats[5][test_player_name]
            process(
                test_player['three_point_sequence'],
                test_player['clock_time_sequence_three_point'],
                test_player['is_home_sequence_three_point']
            )
            return'''

        samples, gammas = run_mcmc(
            all_games=all_games_for_mcmc,
            gammas=GAMMAS,
            n_iter=1000,
            warmup=600,
            beta_step=0.1,
            b_i_step=0.1,
            verbose=True
        )
        samples = canonicalize_samples(samples)
        GAMMAS = gammas


        print(f"Saved {len(samples.get('beta', []))} post-warmup MCMC draws")

        # print(f"beta: {beta}")

       # print(f"samples[acceptance]: {samples['acceptance']}")

        summarize_mcmc(samples)
        plot_trace(samples, warmup=30000)
        apply_mcmc_samples_to_globals(samples, method='mean')



        # Save the all_games_for_mcmc list to a file
        '''with open('stats_and_samples/all_games_for_mcmc.pkl', 'wb') as f:
            pickle.dump(all_games_for_mcmc, f)'''

        # Save values needed for indv_player_metrics in a file
        with open('stats_and_samples/mcmc_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
        with open('stats_and_samples/extended_game_stats.pkl', 'wb') as f:
            pickle.dump(extended_game_stats, f)
        with open('stats_and_samples/appended_game_stats.pkl', 'wb') as f:
            pickle.dump(appended_game_stats, f)
        with open('stats_and_samples/all_games_for_mcmc.pkl', 'wb') as f:
            pickle.dump(all_games_for_mcmc, f)
        


    while True:
        # user_input = input("Enter the player name: ")
        user_input = "Tre White"
        all_player_names = list(extended_game_stats.keys())
        if user_input in all_player_names:
            test_player_name = user_input
        else:
            print("Player not found. Please try again.")
            continue

        # Run forward filter on the test player (example: Milan Momcilovic)
        print(f"\nRunning forward filter on the test player ({test_player_name})... #########################################################################################################################################")
        test_player = extended_game_stats[test_player_name]
        # Get amount of shots the player took each game
        game_shots = [] # holds the number of shots the player took in each game
        game_ids = [] # holds the game ids
        for game_id, game_stats in enumerate(appended_game_stats):

            for player, stats in game_stats.items():
                if player == test_player_name:
                    game_shots.append(len(stats['three_point_sequence']))
                    game_ids.append(game_id)
                    # print(f"appending game {game_id} shots: {len(stats['three_point_sequence'])}, {stats['three_point_sequence']}")
        
        # game_shots[0] = game_shots[0] - sum(game_shots[1:])

        process_params = [test_player['three_point_sequence'], test_player['clock_time_sequence_three_point'], test_player['is_home_sequence_three_point'], test_player['opp_def_3pt_pct_avg'],test_player['three_point_game_num'], test_player['three_point_momentum'], test_player['three_point_intercept'], game_shots, game_ids, test_player_name, appended_game_stats, extended_game_stats]

        gammas_mean = np.mean(samples['gammas'], axis=0).tolist()
        # print(f"Initializing dash app")

        all_player_stats = {}
        counter = 0
        for player in all_player_names:
            temp_test_player = extended_game_stats[player]
            if len(temp_test_player.get('three_point_sequence', [])) < 1:
                print(f"Skipping {player}: no three-point shots")
                continue
            temp_game_shots = []
            temp_game_ids = []
            for game_id, game_stats in enumerate(appended_game_stats):
                if player in game_stats:
                    temp_game_shots.append(len(game_stats[player]['three_point_sequence']))
                    temp_game_ids.append(game_id)
            temp_process_params = [temp_test_player['three_point_sequence'], temp_test_player['clock_time_sequence_three_point'], temp_test_player['is_home_sequence_three_point'], temp_test_player['opp_def_3pt_pct_avg'], temp_test_player['three_point_game_num'], temp_test_player['three_point_momentum'], temp_test_player['three_point_intercept'], temp_game_shots, temp_game_ids, player, appended_game_stats, extended_game_stats]
            app, temp_player_stats = indv_player_metrics.init_dash_app(player, temp_process_params, all_player_names, gammas_mean)
            temp_player_stats['team_name'] = extended_game_stats[player]['team_name']
            counter += 1
            if counter > 100:
                break
            all_player_stats[player] = temp_player_stats
        
        # Store all player stats in a json file
        def _json_default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')

        with open('all_player_hothand_stats.json', 'w') as f:
            json.dump(all_player_stats, f, default=_json_default)
        
        app, player_stats = indv_player_metrics.init_dash_app(test_player_name, process_params, all_player_names, gammas_mean, verbose=True)
        # print(f"Dash app initialized")
        
        # app.run(debug=True)

        summarize_mcmc(samples)
        # return

        return app

        # break


# if __name__ == "__main__":
#     main()