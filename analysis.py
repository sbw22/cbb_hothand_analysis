import json

class Analysis:
    def __init__(self):
        pass

    def import_player_stats(self):
        player_stats = {}
        json_path = 'all_player_hothand_stats.json'
        with open(json_path, 'r') as f:
            player_stats = json.load(f)
        
        # print(player_stats['Tre White'])
        return player_stats

    def analyze_player_stats(self):
        player_stats = self.import_player_stats()
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