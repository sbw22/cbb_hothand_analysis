import http.client
import json
from datetime import datetime, timedelta
from math import exp, log
import numpy as np
from scipy.stats import invwishart
from fetch_ncaa_data import NCAADataFetcher
import pickle
import os
from dash import Dash, html, dcc
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

# State index constants
C_IDX, N_IDX, H_IDX = 0, 1, 2
STATE_NAMES = ['C', 'N', 'H']
TRANSITIONS = ['CH', 'HC', 'CN', 'NC', 'NH', 'HN']
T_IDX = {t: i for i, t in enumerate(TRANSITIONS)}   # e.g. 'CH' -> 0

NUM_FEATURES = 3   # must match len(Xi_n): [clock_time, is_home]

# =============================================================================
# GLOBAL MODEL PARAMETERS  (updated in-place during MCMC)
# =============================================================================

# β: global fixed-effect coefficients, one vector per transition pair.
# Initialized to zeros → all transition probabilities start at 1/3 each.
beta = {t: np.zeros(NUM_FEATURES) for t in TRANSITIONS}

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
                 initial_dist: np.ndarray) -> list:
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
                                initial_dist: np.ndarray) -> list:
    """
    Run FFBS for every game. Returns a list of state sequence arrays,
    one per game.

    all_games: list of dicts, each with keys:
        'shot_sequence'  : list of ints
        'Xi_sequence'    : (M_i, NUM_FEATURES) np.ndarray
        'game_id'        : int
    """
    state_sequences = []
    for game in all_games:
        gid = game['game_id']
        shots = game['shot_sequence']
        Xi_seq = game['Xi_sequence']

        alphas = forward_pass(shots, Xi_seq, gid, gammas, initial_dist)
        Z = backward_sample(alphas, shots, Xi_seq, gid)
        state_sequences.append(Z)

    return state_sequences


# =============================================================================
# LOG-LIKELIHOOD HELPERS
# =============================================================================

def log_likelihood_transition(beta_vec: np.ndarray, b_i_scalar: float,
                               transition: str,
                               observations: list) -> float:
    """
    Log-likelihood contribution of one transition type (e.g. 'CH') across
    all observed transitions from the source state of that pair.

    observations: list of (Xi_n, did_transition) tuples where
        Xi_n           : (NUM_FEATURES,) feature vector
        did_transition : bool, True if the transition 'CH' actually occurred
    """
    ll = 0.0
    src = transition[0]   # e.g. 'C' for 'CH'

    for Xi_n, did_transition in observations:
        # Compute eta for just this transition
        eta = Xi_n @ beta_vec + b_i_scalar

        # The softmax denominator requires the other off-diagonal eta too,
        # but since we don't have it here we use a marginal binary logit:
        #   p(this transition | from src) vs p(not this transition | from src)
        # This is an approximation that's standard when updating one β at a time.
        # For a full joint update you'd need all etas simultaneously.
        p = exp(eta) / (1.0 + exp(eta)) # Probability of the transition occurring
        p = min(max(p, 1e-10), 1 - 1e-10)   # numerical clip
        ll += log(p) if did_transition else log(1.0 - p)
    
    """
    The larger ll, the more likely the transition is to occur, and the better beta is at predicting the transition.
    """

    return ll



def log_prior_beta(beta_vec: np.ndarray, prior_var: float = 1.0) -> float:
    """
    Log prior for one beta vector: N(0, prior_var * I).
    The paper doesn't specify β's prior exactly; N(0,1) per component is standard.
    """
    return -0.5 * np.sum(beta_vec ** 2) / prior_var
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
    dst_state = {'C': C_IDX, 'N': N_IDX, 'H': H_IDX}[transition[1]]
    observations = []

    for game, Z in zip(all_games, state_sequences):
        Xi_seq = game['Xi_sequence']
        M = len(Z)
        for n in range(M - 1):
            if Z[n] == src_state:
                did = (Z[n + 1] == dst_state)
                observations.append((Xi_seq[n], did))

    return observations


def mh_update_beta(transition: str, all_games: list,
                   state_sequences: list, step_size: float = 0.1) -> bool:
    """
    One Metropolis-Hastings step for beta[transition].

    Proposal: beta_proposed = beta_current + N(0, step_size^2 * I)
    Accept/reject via log MH ratio:
        log r = log p(data | proposed) + log prior(proposed)
              - log p(data | current)  - log prior(current)

    Returns True if the proposal was accepted.
    """
    global beta

    current = beta[transition].copy()
    proposed = current + np.random.normal(0, step_size, size=NUM_FEATURES)

    # Gather observations where this transition's source state was active
    # We pass b_i=0 here because b_i is accounted for separately per game.
    # For a more exact update you'd marginalise over b_i, but fixing at 0
    # when updating β is standard practice in this type of mixed model.
    observations = collect_transition_observations(transition, all_games, state_sequences)

    if len(observations) == 0:
        return False   # No data for this transition; skip

    ll_current  = log_likelihood_transition(current,  0.0, transition, observations)
    ll_proposed = log_likelihood_transition(proposed, 0.0, transition, observations)
    lp_current  = log_prior_beta(current)
    lp_proposed = log_prior_beta(proposed)

    log_ratio = (ll_proposed + lp_proposed) - (ll_current + lp_current)

    # the log of a uniform distribution is between 0 and -infinity. This follows the mh notes I have (I think), 
    # since ratios above 0 will then always be accepted, and ratio below 0 will sometimes be accepted.
    if log(np.random.uniform()) < log_ratio: 
        beta[transition] = proposed
        return True
    return False


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
             verbose: bool = True) -> dict:
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

    # Initialize b_i to zero vectors for all games
    for gid in game_ids:
        b_i_store[gid] = np.zeros(6)

    # Initialize beta to small random values to break symmetry
    for t in TRANSITIONS:
        beta[t] = np.random.normal(0, 0.1, size=NUM_FEATURES)

    # Tracking
    samples = {'beta': [], 'b_i': [], 'Sigma_b': [], 'acceptance': {}}
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
        state_sequences = sample_all_state_sequences(all_games, gammas, initial_dist)

        # --- Step 2: Update β for each transition pair ---
        for t in TRANSITIONS:
            accepted = mh_update_beta(t, all_games, state_sequences, step_size=beta_step)
            if accepted:
                accept_counts[t] += 1

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

        if verbose and (iteration + 1) % 100 == 0:
            beta_rates = {t: accept_counts[t] / (iteration + 1) for t in TRANSITIONS}
            b_i_rate = accept_counts['b_i'] / max(total_b_i_proposals, 1)
            print(f"Iter {iteration + 1}/{n_iter} | "
                  f"β accept rates: { {t: f'{r:.2f}' for t, r in beta_rates.items()} } | "
                  f"b_i accept rate: {b_i_rate:.2f}")

    samples['acceptance'] = {
        t: accept_counts[t] / n_iter for t in TRANSITIONS
    }
    samples['acceptance']['b_i'] = accept_counts['b_i'] / max(total_b_i_proposals, 1)

    return samples


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

    print("\nSigma_b posterior mean (diagonal — per-transition variance):")
    Sigma_draws = np.array(samples['Sigma_b'])   # (n_samples, 6, 6)
    Sigma_mean = Sigma_draws.mean(axis=0)
    for i, t in enumerate(TRANSITIONS):
        print(f"  Var(b_i^{t}): {Sigma_mean[i, i]:.3f}")

    '''print("\nMH Acceptance rates:")
    for key, rate in samples['acceptance'].items():
        print(f"  {key}: {rate:.3f}")'''


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

def build_game_dict(game_id: int, shot_sequence: list,
                    clock_sequence: list, is_home_sequence: list, opp_def_3pt_pct_avg: list) -> dict:
    """
    Pack raw sequences for one game into the dict format expected by MCMC.

    Xi_sequence columns: [clock_time, is_home]
    """
    M = len(shot_sequence)
    Xi_seq = np.column_stack([
        np.array(clock_sequence[:M], dtype=float),
        np.array(is_home_sequence[:M], dtype=float),
        np.array(opp_def_3pt_pct_avg[:M], dtype=float)
    ])
    return {
        'game_id': game_id,
        'shot_sequence': shot_sequence,
        'Xi_sequence': Xi_seq,
    }




# =============================================================================
# MAIN
# =============================================================================

def main():
    from indv_player_metrics import IndvPlayerMetrics

    indv_player_metrics = IndvPlayerMetrics()
    ncaa_data_fetcher = NCAADataFetcher()


    appended_game_stats = []
    extended_game_stats = {}

    

    gammas = [0.20, 0.32, 0.48]   # gamma_C, gamma_N, gamma_H
    all_games_for_mcmc = []
    samples = []


    # If the file all_games_for_mcmc.pkl exists, load it and skip running the MCMC
    if os.path.exists('stats_and_samples/mcmc_samples.pkl') and os.path.exists('stats_and_samples/extended_game_stats.pkl') and os.path.exists('stats_and_samples/appended_game_stats.pkl'): # and os.path.exists('stats_and_samples/all_games_for_mcmc.pkl'):
        '''with open('stats_and_samples/all_games_for_mcmc.pkl', 'rb') as f:
            all_games_for_mcmc = pickle.load(f)
        print(f"Loaded {len(all_games_for_mcmc)} games from all_games_for_mcmc.pkl")'''
        with open('stats_and_samples/mcmc_samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        print(f"Loaded MCMC samples with {len(samples.get('beta', []))} post-warmup draws")
        with open('stats_and_samples/extended_game_stats.pkl', 'rb') as f:
            extended_game_stats = pickle.load(f)
        print(f"Loaded {len(extended_game_stats)} extended game stats from extended_game_stats.pkl")
        with open('stats_and_samples/appended_game_stats.pkl', 'rb') as f:
            appended_game_stats = pickle.load(f)
        print(f"Loaded {len(appended_game_stats)} appended game stats from appended_game_stats.pkl")

        samples['beta'] = [samples['beta'][0]]
        samples['b_i'] = [samples['b_i'][0]]
        samples['Sigma_b'] = [samples['Sigma_b'][0]]

        '''with open('stats_and_samples/mcmc_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)'''

        apply_mcmc_samples_to_globals(samples, method='last')

    else:
        # If the files don't exist, run the MCMC
        print("No stats_and_samples/mcmc_samples.pkl, stats_and_samples/extended_game_stats.pkl, or stats_and_samples/appended_game_stats.pkl file found. Running MCMC.")

        pbp_data, boxscore_data = ncaa_data_fetcher.get_pbp_data()
        print(f"Finished getting play-by-play data for {len(pbp_data)} games.")

        print(f"Inspecting the first game's play-by-play data:")
        print(f"pbp_data[0]['periods'][0]['playbyplayStats'] has "
            f"{len(pbp_data[0]['periods'][0]['playbyplayStats'])} events, with keys "
            f"{pbp_data[0]['periods'][0]['playbyplayStats'][23]['homeText']}\n")

        for pbp_data, boxscore_data in zip(pbp_data, boxscore_data):
            game_stats = ncaa_data_fetcher.get_all_player_stats_of_game(pbp_data, boxscore_data)
            # re9fjigdknm
            appended_game_stats.append(game_stats)

            for player, stats in game_stats.items():
                if player not in extended_game_stats:
                    extended_game_stats[player] = stats
                    
                    continue

                # FIX
                extended_game_stats[player]['two_point_sequence'].extend(stats['two_point_sequence'])
                extended_game_stats[player]['three_point_sequence'].extend(stats['three_point_sequence'])
                extended_game_stats[player]['clock_time_sequence_two_point'].extend(stats['clock_time_sequence_two_point'])
                extended_game_stats[player]['clock_time_sequence_three_point'].extend(stats['clock_time_sequence_three_point'])
                extended_game_stats[player]['is_home_sequence_two_point'].extend(stats['is_home_sequence_two_point'])
                extended_game_stats[player]['is_home_sequence_three_point'].extend(stats['is_home_sequence_three_point'])
                extended_game_stats[player]['opp_def_3pt_pct_avg'].extend(stats['opp_def_3pt_pct_avg'])
                
                

        print("Player stats for the first game:")
        for player, stats in appended_game_stats[0].items():
            total = len(stats['two_point_sequence']) + len(stats['three_point_sequence'])
            print(f"  {player}: {total} total actions "
                f"(2pt: {len(stats['two_point_sequence'])}, "
                f"3pt: {len(stats['three_point_sequence'])})")

        # ------------------------------------------------------------------
        # Build all_games list for MCMC
        # Each entry is one player-game observation unit.
        # Here we run across all games for the test player, treating each
        # game as a separate observation unit (as the paper intends).
        # ------------------------------------------------------------------

        all_games_for_mcmc = []

        for game_id, game_stats in enumerate(appended_game_stats):
            '''if test_player_name not in game_stats:
                continue'''
            # print(game_stats)
            for player, stats in game_stats.items():
                player_data = stats
                shots = player_data['three_point_sequence']
                clocks = player_data['clock_time_sequence_three_point']
                home = player_data['is_home_sequence_three_point']
                opp_def_3pt_pct_avg = player_data['opp_def_3pt_pct_avg']

                if len(shots) < 2:
                    continue   # need at least 2 shots for a transition

                game_dict = build_game_dict(game_id, shots, clocks, home, opp_def_3pt_pct_avg)
                all_games_for_mcmc.append(game_dict)

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

        # gammas = [0.20, 0.32, 0.48]   # gamma_C, gamma_N, gamma_H

        samples = run_mcmc(
            all_games=all_games_for_mcmc,
            gammas=gammas,
            n_iter=60000,
            warmup=30000,
            beta_step=0.1,
            b_i_step=0.1,
            verbose=True
        )



        print(f"Saved {len(samples.get('beta', []))} post-warmup MCMC draws")

        # print(f"beta: {beta}")

       # print(f"samples[acceptance]: {samples['acceptance']}")

        summarize_mcmc(samples)
        apply_mcmc_samples_to_globals(samples, method='last')



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
        print(f"\nRunning forward filter on the test player ({test_player_name})...")
        test_player = extended_game_stats[test_player_name]
        # Get amount of shots the player took each game
        game_shots = [] # holds the number of shots the player took in each game
        for game_id, game_stats in enumerate(appended_game_stats):
            for player, stats in game_stats.items():
                if player == test_player_name:
                    game_shots.append(len(stats['three_point_sequence']))
                    print(f"appending game {game_id} shots: {len(stats['three_point_sequence'])}, {stats['three_point_sequence']}")
        
        game_shots[0] = game_shots[0] - sum(game_shots[1:])
        print(f"game_shots: {game_shots}")

        process_params = [test_player['three_point_sequence'], test_player['clock_time_sequence_three_point'], test_player['is_home_sequence_three_point'], test_player['opp_def_3pt_pct_avg'], game_shots, test_player_name, appended_game_stats, extended_game_stats]

        
        '''all_beliefs = indv_player_metrics.process(
            test_player['three_point_sequence'],
            test_player['clock_time_sequence_three_point'],
            test_player['is_home_sequence_three_point'],
            test_player['opp_def_3pt_pct_avg'],
            game_shots,
            test_player_name
        )'''


        app = indv_player_metrics.init_dash_app(test_player_name, process_params, all_player_names)
        
        # app.run(debug=True)

        return app

        # break


# if __name__ == "__main__":
#     main()