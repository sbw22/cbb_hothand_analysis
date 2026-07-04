import numpy as np
from hothand_v2 import build_transition_matrix
class IndvPlayerMetrics:
    def __init__(self):
        STATE_NAMES = ['C', 'N', 'H']
        self.STATE_NAMES = STATE_NAMES
        self.build_transition_matrix = build_transition_matrix

    def create_graph_data(self, all_beliefs: list, shot_sequence: list, game_shots: list, test_player_name: str):
        """
        Populate a plotly plot with the data from all beliefs. Include on each shot if the player made or missed the shot, ideally with a closed circle if the shot was made and an open circle if the shot was missed.
        Add lines that seperate each game.
        """
        import plotly.graph_objects as go
        beliefs = np.array(all_beliefs)   # shape (num_shots, 3)
        x = list(range(1, len(beliefs) + 1))   # display shots as 1, 2, 3, ...
        made = [shot == 0 for shot in shot_sequence]
        game_start_indices = [1 + sum(game_shots[:i]) for i in range(len(game_shots))]
        y_max = float(beliefs.max())
        marker_y = 0   # sit on the x-axis

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 0], mode='lines', name='P(C)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 1], mode='lines', name='P(N)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 2], mode='lines', name='P(H)', line=dict(color='red')))
        made_x = [x[i] for i, is_made in enumerate(made) if is_made]
        missed_x = [x[i] for i, is_made in enumerate(made) if not is_made]
        if made_x:
            fig.add_trace(go.Scatter(
                x=made_x, y=[marker_y] * len(made_x), mode='markers', name='Made',
                marker=dict(color='green', size=10, symbol='circle'),
            ))
        if missed_x:
            fig.add_trace(go.Scatter(
                x=missed_x, y=[marker_y] * len(missed_x), mode='markers', name='Missed',
                marker=dict(color='red', size=10, symbol='circle-open', line=dict(color='red', width=2)),
            ))
        for i in range(len(game_start_indices)):
            print(f"Game start index: {game_start_indices[i]}")
            fig.add_vline(x=game_start_indices[i]-0.5, line=dict(color='black', width=2 ))
        fig.update_layout(title=f'Beliefs over Time- {test_player_name}', xaxis_title='Shot Number', yaxis_title='Belief (%)')
        fig.update_yaxes(range=[0, y_max+0.1])
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.show()
        return

    # =============================================================================
    # LEGACY FORWARD FILTER (kept for single-game diagnostic use)
    # =============================================================================

    def process(self, shot_sequence, clock_sequence, is_home_sequence, game_shots, test_player_name,
                initial_distribution=None,
                gamma_C=0.2, gamma_N=0.32, gamma_H=0.48):
        """
        Single-game forward filter (diagnostic / visualisation only).
        Prints shot-by-shot belief updates and transition statistics.
        Does NOT update β or b_i — use run_mcmc() for fitting.
        """
        if initial_distribution is None:
            initial_distribution = np.array([1/3, 1/3, 1/3])

        gammas = [gamma_C, gamma_N, gamma_H]
        belief = np.array(initial_distribution, dtype=float)

        M = len(shot_sequence)
        Xi_seq = np.column_stack([
            np.array(clock_sequence[:M], dtype=float),
            np.array(is_home_sequence[:M], dtype=float)
        ])

        p = np.zeros((3, 3))

        all_beliefs = []

        for n, shot in enumerate(shot_sequence):
            made = (shot == 0)
            Xi_n = Xi_seq[n]

            P = self.build_transition_matrix(Xi_n, game_id=0) # P is using the build_transition_matrix function from hothand_v2.py, and the updated global beta and b_i values
            predicted = belief @ P

            likelihoods = np.array([gammas[s] if made else (1 - gammas[s]) for s in range(3)])
            updated = likelihoods * predicted
            total = updated.sum()
            belief = updated / total if total > 0 else np.ones(3) / 3.0

            p = P

            all_beliefs.append(belief)

            print(f"Shot: {'make' if made else 'miss'} | "
                f"P(C)={belief[0]:.3f}  P(N)={belief[1]:.3f}  P(H)={belief[2]:.3f}")

        print(f"\nTransition matrix (last shot):")
        for j, row in enumerate(p):
            print(f"  {self.STATE_NAMES[j]}: {row}")

        self.t_step_transition_probabilities(p, 10)
        self.occupancy_times(p, n=M - 1, initial_state=0)
        self.sojourn_times(p, t=1)
        self.create_graph_data(all_beliefs, shot_sequence, game_shots, test_player_name)

        


    # =============================================================================
    # ANALYSIS FUNCTIONS
    # =============================================================================

    def t_step_transition_probabilities(self, p, t=2):
        np_p = np.array(p)
        p_t = np.linalg.matrix_power(np_p, t)
        print(f"\nTransition probabilities in {t} step(s):")
        for row in p_t:
            print(f"  {row}")


    def sojourn_times(self, p, t=1):
        """
        P(sojourn = t) = p_jj^(t-1) * (1 - p_jj)
        Expected sojourn time = 1 / (1 - p_jj)
        """
        p = np.array(p)
        sojourn = [0.0, 0.0, 0.0]
        for j in range(3):
            sojourn[j] = (p[j, j] ** (t - 1)) * (1 - p[j, j]) if t >= 1 else 0.0

        expected = [1.0 / (1 - p[j, j]) if p[j, j] < 1 else float('inf') for j in range(3)]

        print(f"\nSojourn time distribution P(sojourn={t}):")
        for j in range(3):
            print(f"  {self.STATE_NAMES[j]}: P={sojourn[j]:.3f}  |  E[sojourn]={expected[j]:.2f} shots")

        return sojourn


    def occupancy_times(self, p, n, initial_state=None):
        """
        Occupancy time matrix M where M[j, k] = expected visits to state k
        starting from state j over n transitions (n+1 shots).

        Computed as M = sum_{t=0}^{n} P^t (matrix geometric series).
        """
        np_p = np.array(p, dtype=float)
        M_occ = np.zeros((3, 3))
        P_power = np.eye(3)
        for t in range(n + 1):
            M_occ += P_power
            P_power = P_power @ np_p

        print(f"\nOccupancy time matrix over {n} transitions ({n+1} shots):")
        print(f"  {'':6} {'->C':>8} {'->N':>8} {'->H':>8}")
        for j in range(3):
            row_str = "  ".join(f"{M_occ[j, k]:8.3f}" for k in range(3))
            print(f"  {self.STATE_NAMES[j]}: {row_str}  (sum={M_occ[j].sum():.1f})")

        if initial_state is not None:
            print(f"\nStarting from {self.STATE_NAMES[initial_state]}:")
            for k in range(3):
                print(f"  E[visits to {self.STATE_NAMES[k]}]: {M_occ[initial_state, k]:.3f}")
            return M_occ[initial_state]

        return M_occ