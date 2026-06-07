from math import exp


# Keep the state names in one place so the rest of the model can refer to
# them without repeating string literals everywhere.
STATE_NAMES = ("cold", "neutral", "hot")


# This file is for the probability side of the hot-hand model.
# The goal is to keep the math helpers separate from the play-by-play scraping
# code in hothand.py so the model can be developed more cleanly.
#
# Function guide:
# - validate_probability(p)
#     Make sure one number is a valid probability.
# - normalize(weights)
#     Turn a list of nonnegative weights into probabilities that sum to 1.
# - softmax(logits)
#     Turn unconstrained scores into a valid categorical probability vector.
# - bernoulli_likelihood(made_shot, make_probability)
#     Return the probability of a make or miss under one Bernoulli model.
# - beta_posterior(alpha, beta, makes, misses)
#     Update a Beta prior after seeing makes and misses.
# - beta_posterior_mean(alpha, beta, makes, misses)
#     Get the posterior mean for one make probability.
# - dirichlet_posterior(alpha, counts)
#     Update a Dirichlet prior after seeing categorical counts.
# - dirichlet_posterior_mean(alpha, counts)
#     Get the posterior mean for an initial-state vector or transition row.
# - predict_next_state_distribution(belief, transition_matrix)
#     Push the current state belief forward one step.
# - update_state_belief(...)
#     Do one full HMM update: predict, apply shot likelihoods, renormalize.
#
# Distribution choices for this project:
# - Hidden state: Categorical over {cold, neutral, hot}
# - Initial state distribution: Dirichlet prior
# - Transition rows: Dirichlet prior if fixed, or softmax if feature-based
# - Shot outcome given state: Bernoulli
# - Unknown make probability inside one state: Beta prior


def validate_probability(p):
    # This helper catches values that should be probabilities but are not in
    # the valid range [0, 1].
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Expected probability in [0, 1], got {p}.")
    return p


def normalize(weights):
    # Many model updates create unnormalized weights first. This function turns
    # those weights into a proper probability distribution.
    if not weights:
        raise ValueError("Cannot normalize an empty list.")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Weights must be nonnegative, got {weights}.")

    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Weight sum must be positive, got {weights}.")

    return [weight / total for weight in weights]


def softmax(logits):
    # Softmax is useful when we want to start from arbitrary real-valued scores
    # and convert them into probabilities that sum to 1.
    if not logits:
        raise ValueError("Softmax needs at least one logit.")

    max_logit = max(logits)
    shifted = [exp(logit - max_logit) for logit in logits]
    return normalize(shifted)


def bernoulli_likelihood(made_shot, make_probability):
    # A shot is a binary event, so a Bernoulli model is the natural choice.
    # If the shot was made, return p. If it was missed, return 1 - p.
    p = validate_probability(make_probability)
    if made_shot:
        return p
    else:
        return 1.0 - p


def beta_posterior(alpha, beta, makes, misses):
    # Use this when one unknown quantity is a single make probability.
    # Beta is a good prior here because it updates cleanly after Bernoulli data.
    #
    # Prior: theta ~ Beta(alpha, beta)
    # Data: makes, misses
    # Posterior: theta | data ~ Beta(alpha + makes, beta + misses)
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta prior parameters must be positive.")
    if makes < 0 or misses < 0:
        raise ValueError("Make and miss counts must be nonnegative.")

    return alpha + makes, beta + misses


def beta_posterior_mean(alpha, beta, makes, misses):
    # This is a quick summary of the updated Beta distribution.
    post_alpha, post_beta = beta_posterior(alpha, beta, makes, misses)
    return post_alpha / (post_alpha + post_beta)


def dirichlet_posterior(alpha, counts):
    # Use this when the unknown quantity is a whole probability vector, like
    # an initial-state distribution or one row of a transition matrix.
    #
    # Prior: pi ~ Dirichlet(alpha)
    # Data: counts of visits/transitions in each category
    # Posterior: pi | data ~ Dirichlet(alpha + counts)
    if len(alpha) != len(counts):
        raise ValueError("alpha and counts must have the same length.")
    if not alpha:
        raise ValueError("Dirichlet parameters cannot be empty.")
    if any(a <= 0 for a in alpha):
        raise ValueError("Dirichlet prior parameters must be positive.")
    if any(count < 0 for count in counts):
        raise ValueError("Counts must be nonnegative.")

    return [a + count for a, count in zip(alpha, counts)]


def dirichlet_posterior_mean(alpha, counts):
    # This gives one simple "best guess" vector after the Dirichlet update.
    posterior = dirichlet_posterior(alpha, counts)
    total = sum(posterior)
    return [value / total for value in posterior]


def predict_next_state_distribution(belief, transition_matrix):
    # This is the prediction step of the HMM. Start from the current belief
    # over states and move it forward one shot using the transition matrix.
    if len(belief) != len(transition_matrix):
        raise ValueError("Belief length must match transition matrix size.")

    for row in transition_matrix:
        if len(row) != len(belief):
            raise ValueError("Transition matrix must be square.")

    next_belief = [0.0 for _ in belief]
    for current_state, current_prob in enumerate(belief):
        for next_state, transition_prob in enumerate(transition_matrix[current_state]):
            next_belief[next_state] += current_prob * transition_prob

    return next_belief


def update_state_belief(belief, transition_matrix, emission_make_probabilities, made_shot):
                                                                                                                                                                                                                                                                                                # This is one full HMM update step.
    #
    # 1. Predict the next hidden-state distribution with the transition matrix.
    # 2. Weight that prediction by the Bernoulli likelihood in each state.
    # 3. Renormalize to get the posterior belief over states.
    if len(emission_make_probabilities) != len(belief):
        raise ValueError("Emission probabilities must match belief length.")

    predicted = predict_next_state_distribution(belief, transition_matrix)
    weighted = []
    for state_index, predicted_prob in enumerate(predicted):
        likelihood = bernoulli_likelihood(
            made_shot=made_shot,
            make_probability=emission_make_probabilities[state_index],
        )
        weighted.append(predicted_prob * likelihood)

    return normalize(weighted)
