import numpy as np

def log_empirical_transitions_by_momentum(all_games, state_sequences):
    """
    Examine the sampled hidden-state transitions and report how they
    behave at different momentum levels.

    This is diagnostic only. It does not modify the model.
    """

    # Momentum bins
    bins = [
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.01),
    ]

    state_names = ['C', 'N', 'H']

    # counts[(bin_index, source, destination)]
    counts = {}

    for bin_idx in range(len(bins)):
        for src in range(3):
            for dst in range(3):
                counts[(bin_idx, src, dst)] = 0

    # ------------------------------------------------------------------
    # Go through every game and every transition
    # ------------------------------------------------------------------
    for game, Z in zip(all_games, state_sequences):

        # We need momentum for each shot
        # momentums = game.get('momentum')
        # print(f"game keys: {game.keys()}")
        Xi_seq = game['Xi_sequence']
        momentums = [Xi_n[4] for Xi_n in Xi_seq]

        if momentums is None:
            continue

        # Need one momentum value for each state
        if len(momentums) != len(Z):
            continue

        for n in range(len(Z) - 1):

            src = int(Z[n])
            dst = int(Z[n + 1])
            momentum = float(momentums[n])

            # Find momentum bin
            for bin_idx, (low, high) in enumerate(bins):
                if low <= momentum < high:
                    counts[(bin_idx, src, dst)] += 1
                    break

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 75)
    print("EMPIRICAL SAMPLED-STATE TRANSITIONS BY MOMENTUM")
    print("=" * 75)

    for bin_idx, (low, high) in enumerate(bins):

        print(f"\nMomentum {low:.2f} - {high:.2f}")

        for src in range(3):

            row_counts = np.array([
                counts[(bin_idx, src, dst)]
                for dst in range(3)
            ])

            total = row_counts.sum()

            if total == 0:
                print(
                    f"  From {state_names[src]}: "
                    "NO OBSERVATIONS"
                )
                continue

            probabilities = row_counts / total

            print(
                f"  From {state_names[src]} "
                f"(n={total}): "
                f"C={probabilities[0]:.3f}  "
                f"N={probabilities[1]:.3f}  "
                f"H={probabilities[2]:.3f}"
            )

    print("=" * 75)



def log_state_make_rates_by_momentum(all_games, state_sequences):

    bins = [
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.01),
    ]

    state_names = ['C', 'N', 'H']

    made_counts = {}
    total_counts = {}

    for bin_idx in range(len(bins)):
        for state in range(3):
            made_counts[(bin_idx, state)] = 0
            total_counts[(bin_idx, state)] = 0

    for game, Z in zip(all_games, state_sequences):

        # momentums = game.get('momentum')

        Xi_seq = game['Xi_sequence']
        momentums = [Xi_n[4] for Xi_n in Xi_seq]

        if momentums is None:
            continue

        shots = game['shot_sequence']

        if len(momentums) != len(Z):
            continue

        for n in range(len(Z)):

            momentum = float(momentums[n])
            state = int(Z[n])

            # Find bin
            bin_idx = None

            for i, (low, high) in enumerate(bins):
                if low <= momentum < high:
                    bin_idx = i
                    break

            if bin_idx is None:
                continue

            total_counts[(bin_idx, state)] += 1

            # Adjust this if your shot representation isn't 1/0
            if shots[n] == 1:
                made_counts[(bin_idx, state)] += 1

    print("\n")
    print("=" * 75)
    print("SAMPLED STATE MAKE RATES BY MOMENTUM")
    print("=" * 75)

    for bin_idx, (low, high) in enumerate(bins):

        print(f"\nMomentum {low:.2f} - {high:.2f}")

        for state in range(3):

            total = total_counts[(bin_idx, state)]
            made = made_counts[(bin_idx, state)]

            if total == 0:
                print(f"  {state_names[state]}: NO OBSERVATIONS")
                continue

            rate = made / total

            print(
                f"  {state_names[state]}: "
                f"{rate:.3f} "
                f"({made}/{total})"
            )

    print("=" * 75)