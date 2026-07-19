import numpy as np
from hothand_v2 import build_transition_matrix
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
class IndvPlayerMetrics:
    def __init__(self):
        self.STATE_NAMES = ['C', 'N', 'H']
        self.STATE_NAMES_DISPLAY = ['Cold', 'Neutral', 'Hot']
        self.STATE_COLORS = {'Cold': 'blue', 'Neutral': 'green', 'Hot': 'red'}
        self.build_transition_matrix = build_transition_matrix

    def init_dash_app(self, test_player_name: str, process_params: list, all_player_names: list):

        def create_occupancy_fig(occupancy_times: list):
            occupancy_fig = px.bar(x=self.STATE_NAMES_DISPLAY, y=occupancy_times, title='Expected Number of Shots in Each State -- Next Ten Shots', color=self.STATE_NAMES_DISPLAY, color_discrete_map=self.STATE_COLORS, text_auto=True)
            occupancy_fig.update_layout(
                title='<b>Expected Number of Shots in Each State -- Next Ten Shots</b>',
                autosize=True,
                width=None,
                height=None,
                margin=dict(l=40, r=20, t=50, b=40),
                showlegend=False,
            )
            occupancy_fig.update_xaxes(title='State', ticktext=self.STATE_NAMES_DISPLAY)
            occupancy_fig.update_yaxes(title=' # of Expected Shots in Each State')


            return occupancy_fig

        def create_sojourn_fig(sojourn_times: list):
            sojourn_fig = px.bar(x=self.STATE_NAMES_DISPLAY, y=sojourn_times, title='Expected Number of Shots to Leave State', color=self.STATE_NAMES_DISPLAY, color_discrete_map=self.STATE_COLORS, text_auto=True)
            sojourn_fig.update_layout(
                title='<b>Expected Number of Shots to Leave State</b>',
                autosize=True,
                width=None,
                height=None,
                margin=dict(l=40, r=20, t=50, b=40),
                showlegend=False,
            )
            sojourn_fig.update_xaxes(title='State', ticktext=self.STATE_NAMES_DISPLAY)
            sojourn_fig.update_yaxes(title='# of Shots to Leave State')


            return sojourn_fig

        def create_belief_occupancy_fig(beliefs_percentages: list):
            # We are creating a horizonal multi-colored bar, with the colors inside the bar representing the probabilities of the player being in each state.
            # 2. Build the stacked horizontal bar
            fig = go.Figure()
            bar_1 = beliefs_percentages[0]
            bar_2 = beliefs_percentages[1]
            bar_3 = beliefs_percentages[2]

            fig.add_trace(go.Bar(
                x=[bar_1],
                # y=[self.STATE_NAMES_DISPLAY[0]],
                orientation='h',
                marker_color=self.STATE_COLORS[self.STATE_NAMES_DISPLAY[0]],
                name=self.STATE_NAMES_DISPLAY[0],
                text=[f"{bar_1:.2f}%"],
                textposition='auto',
                insidetextanchor='middle',
            ))

            fig.add_trace(go.Bar(
                x=[bar_2],
                # y=[self.STATE_NAMES_DISPLAY[1]],
                orientation='h',
                marker_color=self.STATE_COLORS[self.STATE_NAMES_DISPLAY[1]],
                name=self.STATE_NAMES_DISPLAY[1],
                text=[f"{bar_2:.2f}%"],
                textposition='auto',
                insidetextanchor='middle',
            ))

            fig.add_trace(go.Bar(
                x=[bar_3],
                # y=[self.STATE_NAMES_DISPLAY[2]],
                orientation='h',
                marker_color=self.STATE_COLORS[self.STATE_NAMES_DISPLAY[2]],
                name=self.STATE_NAMES_DISPLAY[2],
                text=[f"{bar_3:.2f}%"],
                textposition='auto',
                insidetextanchor='middle',
            ))

            # 3. Force the layout to stack the bars next to each other
            fig.update_layout(
                title="Amount of Time as Most Likely State",
                title_x=0.5,
                title_y=0.9,
                barmode='stack',
                xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=20, r=20, t=40, b=20),
                height=100,
                showlegend=False,
                
            )

            return fig

        
        app = Dash(__name__)
        self.appended_game_stats = process_params[5]
        self.extended_game_stats = process_params[6]
        process_stats = self.process(*process_params[:5])
        beliefs_fig, beliefs_percentages = self.create_beliefs_fig(*process_stats[:4])

        belief_occupancy_fig = create_belief_occupancy_fig(beliefs_percentages)

        occupancy_times, last_state = process_stats[5] # Sampling the next 10 shots, so last_state is the initial state
        occupancy_fig = create_occupancy_fig(occupancy_times)

        sojourn_stats = self.sojourn_times(process_stats[4], 1)
        sojourn_fig = create_sojourn_fig(sojourn_stats[1])


        app.layout = html.Div(children=[
            html.H1(f'Hot Hand Probability Analysis', style={'text-align': 'center'}),

            html.Pre(children=[
                "\tHow do you know if a player is on a hot streak? The short answer is that you can't. Coaches even at the highest levels of the game have had trouble identifying players with a hot hand. This has led to inefficiencies in a coach's ability to use a player's hot hand to their advantage. However, while it is difficult to identify a player's hot hand, coaches use many different tools and resources to help them identify players with a hot hand, including their own intuition and observations of the player's performance.", 
                html.Br(), 
                html.Br(), 
                "\tThis project aims to provide another tool for coaches, analysts, and other basketball enthusiasts to help them identify players with a hot hand. This model uses a Bayesian approach to calculate the player's statistical probabilities of being in each state (Cold, Neutral, Hot) over time.",
            ],
            
            style={'font-size': '14px', 'color': 'black', 'text-align': 'left', 'width': '80%', 'margin': '0 auto', 'margin-bottom': '20px', 'white-space': 'pre-wrap'}),

            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': player, 'value': player} for player in all_player_names],
                value = test_player_name,
                clearable=False,
                style={'font-size': '20px', 'color': 'black', 'width': '20%', 'margin': '0 auto', 'text-align': 'center'}
            ),
            # This plot shows the player's probabilities of being in each state <br>over time. Each line represents the probability of the player being in that <br>specific state. The shots that were made and missed are marked with a <br>green and red circle respectively.",
            
            html.Div(children=[
                
                dcc.Graph(
                    id='beliefs-graph', 
                    figure=beliefs_fig,
                ),
                dcc.Graph(
                    id='belief-occupancy-graph', 
                    figure=belief_occupancy_fig,
                    config={'displayModeBar': False},
                    style={'width': '70%', 'margin': '0 auto', 'text-align': 'center'}
                ),
                html.Div(
                    html.P("This plot shows the player's probabilities of being in each state over time. Each line represents the probability of the player being in that specific state over time. The shots that were made and missed are marked with a green and red circle respectively."),
                    style={'font-size': '20px', 'color': 'black', 'text-align': 'left', 'width': '80%', 'margin': '0 auto', 'height': '60px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'width': '100%', 'gap': '0px', 'margin': '0 auto', 'height': '100%', 'border': '2px solid black', 'margin-top': '10px', 'padding-bottom': '25px'}),

            html.Div(children=[
                html.Div( children=[
                
                    html.Div(
                        dcc.Graph(id='sojourn-graph', figure=sojourn_fig, style={'width': '100%', 'height': '400px'}, responsive=True),
                    ),

                    html.Div(
                        html.P("This plot shows the expected number of shots the player is expected to be in each state before leaving that state."),
                        style={'font-size': '20px', 'color': 'black', 'text-align': 'left', 'width': '80%', 'margin': '0 auto', 'height': '50px'}
                    ),
                ], style={'flex': '1', 'minWidth': 0, 'border': '2px solid black', 'padding': '10px'}),


                html.Div( children=[
                
                    html.Div(
                        dcc.Graph(id='occupancy-graph', figure=occupancy_fig, style={'width': '100%', 'height': '400px'}, responsive=True),
                    ),

                    html.Div(
                        html.P("This plot shows the expected number of shots the player is expected to take in each state over the next 10 shots."),
                        style={'font-size': '20px', 'color': 'black', 'text-align': 'left', 'width': '80%', 'margin': '0 auto', 'height': '50px'}
                    ),

                ], style={'flex': '1', 'minWidth': 0, 'border': '2px solid black', 'padding': '10px'}),

            ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'gap': '0px', 'margin': '0 auto'}),

        ])


        @callback(
            Output('beliefs-graph', 'figure'),
            Output('belief-occupancy-graph', 'figure'),
            Output('sojourn-graph', 'figure'),
            Output('occupancy-graph', 'figure'),
            # Output('player-name-header', 'children'),
            Input('player-dropdown', 'value'),
        )


        def update_beliefs_and_sojourn_graphs(player_name: str):

            try: 
                updated_process_params = update_process_params(player_name)
                process_stats = self.process(*updated_process_params[:5])
                beliefs_fig, beliefs_percentages = self.create_beliefs_fig(*process_stats[:4])

                sojourn_stats = self.sojourn_times(process_stats[4], 1)
                sojourn_fig = create_sojourn_fig(sojourn_stats[1])

                belief_occupancy_fig = create_belief_occupancy_fig(beliefs_percentages)
        
                occupancy_times, last_state = process_stats[5] # Sampling the next 10 shots, so last_state is the initial state
                occupancy_fig = create_occupancy_fig(occupancy_times)

                return beliefs_fig, belief_occupancy_fig, sojourn_fig, occupancy_fig

            except Exception as e:
                print(f"Error creating beliefs figure: {e}")
                return None, None, None, None



        def update_process_params(player_name: str):
            test_player = self.extended_game_stats[player_name]
            game_shots = [] # holds the number of shots the player took in each game
            for game_id, game_stats in enumerate(self.appended_game_stats):
                for player, stats in game_stats.items():
                    if player == player_name:
                        game_shots.append(len(stats['three_point_sequence']))
                        print(f"appending game {game_id} shots: {len(stats['three_point_sequence'])}, {stats['three_point_sequence']}")
            
            game_shots[0] = game_shots[0] - sum(game_shots[1:])
            return [test_player['three_point_sequence'], test_player['clock_time_sequence_three_point'], test_player['is_home_sequence_three_point'], game_shots, player_name, self.appended_game_stats, self.extended_game_stats]

        
        return app

    def create_beliefs_fig(self, all_beliefs: list, shot_sequence: list, game_shots: list, test_player_name: str):
        """
        Populate a plotly plot with the data from all beliefs. Include on each shot if the player made or missed the shot, ideally with a closed circle if the shot was made and an open circle if the shot was missed.
        Add lines that seperate each game.
        """
        beliefs = np.array(all_beliefs)   # shape (num_shots, 3)
        top_beliefs = [beliefs[i, :].argmax() for i in range(len(beliefs))]
        top_beliefs_counts = [top_beliefs.count(i) for i in range(3)]
        top_beliefs_percentages = [top_beliefs_counts[i] / len(beliefs) * 100 for i in range(3)]
        print(f"\n\ntop_beliefs: {top_beliefs}\n\ntop_beliefs_counts: {top_beliefs_counts}\n\n")
        print(f"\n\ntop_beliefs_percentages: {top_beliefs_percentages}\n\n")

        x = list(range(1, len(beliefs) + 1))   # display shots as 1, 2, 3, ...
        made = [shot == 0 for shot in shot_sequence]
        game_start_indices = [1 + sum(game_shots[:i]) for i in range(len(game_shots))]
        y_max = float(beliefs.max())
        marker_y = 0   # sit on the x-axis

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 0], mode='lines', name='Probability of Being Cold', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 1], mode='lines', name='Probability of Being Neutral', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x, y=beliefs[:, 2], mode='lines', name='Probability of Being Hot', line=dict(color='red')))
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
        for i in range(1, len(game_start_indices)):
            print(f"Game start index: {game_start_indices[i]}")
            fig.add_vline(x=game_start_indices[i]-0.5, line=dict(color='grey', width=2 ), name="Game Seperator Line")
            fig.update_layout(title=f'<b>Beliefs over Time</b> ', xaxis_title='Shot Number', yaxis_title='Belief Probability (%)')
            # 2. Add an annotation right below the title but above the data

        # legend-only stub
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Game Separator Line', line=dict(color='grey', width=2)))
        
        fig.update_yaxes(range=[0, y_max+0.1])
        fig.update_layout(
            # margin=dict(t=150),
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1, ))

        return fig, top_beliefs_percentages

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

        # initial_state = np.where(all_beliefs[0] == max(all_beliefs[0]))[0][0]
        last_state = np.where(all_beliefs[-1] == max(all_beliefs[-1]))[0][0]

        self.t_step_transition_probabilities(p, 10)
        # self.occupancy_times(p, n= M - 1, initial_state=initial_state)
        occupancy_times = self.occupancy_times(p, n= 9, initial_state=last_state) # Sampling the next 10 shots, so last_state is the initial state
        self.sojourn_times(p, t=1)
        # self.create_beliefs_fig(all_beliefs, shot_sequence, game_shots, test_player_name)

        return [all_beliefs, shot_sequence, game_shots, test_player_name, p, [occupancy_times, last_state]]

        


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


        # Round to 3 decimal places
        sojourn = [round(sojourn[j], 3) for j in range(3)]
        expected = [round(expected[j], 3) for j in range(3)]

        print(f"\nSojourn time distribution P(sojourn={t}):")
        for j in range(3):
            print(f"  {self.STATE_NAMES[j]}: P={sojourn[j]:.3f}  |  E[sojourn]={expected[j]:.2f} shots")

        return [sojourn, expected]


    def occupancy_times(self, p, n, initial_state=None):
        """
        Occupancy time matrix M where M[j, k] = expected visits to state k
        starting from state j over n transitions (n+1 shots).

        Computed as M = sum_{t=0}^{n} P^t (matrix geometric series).
        """
        M_occ_init_state = [] # Occupancy times from initial state

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
                M_occ_init_state.append(round(float(M_occ[initial_state, k]), 3))
            return M_occ_init_state

        return M_occ