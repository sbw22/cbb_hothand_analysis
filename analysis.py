import json
from indv_player_metrics import IndvPlayerMetrics
import pickle
class Analysis:
    def __init__(self):
        pass

    def import_player_stats(self):

        indv_player_metrics = IndvPlayerMetrics()
        '''player_stats = {}
        json_path = 'all_player_hothand_stats.json'
        with open(json_path, 'r') as f:
            player_stats = json.load(f)'''
        
        pkl_path = 'stats_and_samples/extended_game_stats.pkl'
        with open(pkl_path, 'rb') as f:
            extended_game_stats = pickle.load(f)
        print(f"Loaded {len(extended_game_stats)} extended game stats from extended_game_stats.pkl")

        pkl_path = 'stats_and_samples/appended_game_stats.pkl'
        with open(pkl_path, 'rb') as f:
            appended_game_stats = pickle.load(f)
        print(f"Loaded {len(appended_game_stats)} appended game stats from appended_game_stats.pkl")

        pkl_path = 'stats_and_samples/gammas.pkl'
        with open(pkl_path, 'rb') as f:
            gammas_mean = pickle.load(f)
        print(f"Loaded gammas_mean from gammas.pkl")
        all_player_names = list(extended_game_stats.keys())


        all_player_stats = {}
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

            all_player_stats[player] = temp_player_stats
        
        # print(player_stats['Tre White'])
        print(f"Loaded {len(all_player_stats)} player stats from extended_game_stats.json and appended_game_stats.json")
        return all_player_stats, all_player_names

    def analyze_player_stats(self):
        player_stats, all_player_names = self.import_player_stats()
        print(player_stats['Tre White'].keys())
        print(player_stats['Tre White']['beliefs_percentages'])
        # print(f"type of player_stats['Tre White']['game_shots'][0]: {type(player_stats['Tre White']['game_shots'][0])}")
        sorted_by_hot_belief = sorted(player_stats.items(), key=lambda x: x[1]['beliefs_percentages'][0], reverse=True)
        counter = 0
        for player, stats in sorted_by_hot_belief:
            
            counter += 1
            print(f"{counter}. {player}: {round(stats['beliefs_percentages'][0], 2)} - {stats['team_name']}")

            if counter > 100:
                break

    def analyze_all_player_stats(self):
        player_stats = self.import_player_stats()

        for player, stats in player_stats.items():
            print(player, stats)

if __name__ == "__main__":
    analysis = Analysis()
    analysis.analyze_player_stats()