import http.client
import json
from datetime import datetime, timedelta


def get_game_ids():

    conn = http.client.HTTPSConnection("ncaa-api.henrygd.me")

    # Define a specific date (e.g., 2024-12-31)
    specific_date = datetime(year=2025, month=11, day=3) # First day of the 2025-26 season
    # Grab game dates for all games in the 2025-26 season
    list_of_game_ids = []
    counter = 0
    while specific_date < datetime(year=2026, month=4, day=15):
        if counter == 1: # Only check the first 2 days of the season to avoid making too many API calls while testing
            break
        counter += 1
        if counter % 10 == 0:
            print(f"Counter: {counter}, specific_date: {specific_date.strftime('%Y-%m-%d')}")

        date_str = specific_date.strftime("%Y/%m/%d")
        conn.request("GET", f"/scoreboard/basketball-men/d1/{date_str}/all-conf")
        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")
        try:
            json_data = json.loads(decoded_data)
        except json.JSONDecodeError:
            print("Response was not valid JSON")
            print(decoded_data)
            return
        
        # print(f"json_data: {json.dumps(json_data, indent=2)}")
        # print(f"json_data[0]: {json.dumps(json_data[0], indent=2)}")

        # print(f"type of json_data: {type(json_data)}")
        # print(f"json_data keys: {json_data.keys()}")

        # These are the four keys in the json_data dict: 'inputMD5Sum', 'instanceId', 'updated_at', and 'games'
        # print(f"json_data['inputMD5Sum']: {json_data['inputMD5Sum']}") Like a key or identifier value or smth. Its a string of number and letters. 
        # print(f"json_data['instanceId']: {json_data['instanceId']}")  # Like a key or identifier value or smth. Its a string of number and letters.
        # print(f"json_data['updated_at']: {json_data['updated_at']}")  # Timestamp of when the data was last updated. Format: "2024-11-03 T00:00:00Z"
        # print(f"json_data['games']: {json.dumps(json_data['games'], indent=2)}")  # List of games on that date. Each game is a dictionary with details about the game.
        
        # print(f"type of json_data['games']: {type(json_data['games'])}")  # Should be a list

        # print(f"\n\njson_data['games'][0]: {json.dumps(json_data['games'][0], indent=2)}")  # Details of the first game in the list -- dict has one value: ['game']
        # print(f"\n\njson_data['games'][0]['game']: {json.dumps(json_data['games'][0]['game'], indent=2)}")  # Details of the first game in the list -- dict has one value: ['game']
        
        for game in json_data['games']:
            game_id = game['game']['gameID']
            list_of_game_ids.append(game_id)

        specific_date += timedelta(days=1)
        # break
    
    conn.close()
    
    return list_of_game_ids


def get_pbp_data():

    conn = http.client.HTTPSConnection("ncaa-api.henrygd.me")

    list_of_game_ids = get_game_ids()

    print(f"Total number of games in the 2025-26 season: {len(list_of_game_ids)}")
    print(f"game_ids for the first 10 games: {list_of_game_ids[:10]}\n")

    # Creates a list of play-by-play data for all the games in the season. Each element in the list is a dict with the play-by-play data for one game.
    list_of_pbp_data = []
    for id in list_of_game_ids:
        conn.request("GET", f"/game/{id}/play-by_play")
        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")
        try:
            pbp_json_data = json.loads(decoded_data)
            list_of_pbp_data.append(pbp_json_data)
        except json.JSONDecodeError:
            print(f"Response for game ID {id} was not valid JSON")
            print(decoded_data)
            continue

    
    '''
    print(f"pbp_json_data keys: {pbp_json_data.keys()}\n\n\n") 

    print(f"pbp_json_data['__typename']: {pbp_json_data['__typename']}\n")  # Should be the same as list_of_game_ids[0]
    print(f"pbp_json_data['contestId']: {pbp_json_data['contestId']}\n")  # Should be the same as list_of_game_ids[0]
    print(f"pbp_json_data['title']: {pbp_json_data['title']}\n") 
    print(f"pbp_json_data['description']: {pbp_json_data['description']}\n") 
    print(f"pbp_json_data['divisionName']: {pbp_json_data['divisionName']}\n")
    print(f"pbp_json_data['status']: {pbp_json_data['status']}\n") 
    print(f"pbp_json_data['period']: {pbp_json_data['period']}\n") 
    print(f"pbp_json_data['minutes']: {pbp_json_data['minutes']}\n") 
    print(f"pbp_json_data['seconds']: {pbp_json_data['seconds']}\n") 
    print(f"pbp_json_data['teams']: {pbp_json_data['teams']}\n") 
    print(f"pbp_json_data['periods']: {pbp_json_data['periods']}\n")
    '''

    conn.close()

    print(f"keys of list_of_pbp_data[0]: {list_of_pbp_data[0].keys()}\n")  # Should include 'periods' and 'teams'

    return list_of_pbp_data


def get_all_player_stats_of_game(pbp_data):
    # This function takes play-by-play data for one game, and returns a dict (for now). Keys of the dict are names of players in the game, 
    # and values are lists of actions the player has in the game. Each item in the list should contain the type of action (e.g., "shot", 
    # "pass", "rebound"), the time of the action, and any other relevant details (e.g., for a shot, whether it was made or missed, the 
    # distance of the shot, etc.).

    player_stats = {}

    print(f"pbp_data keys: {pbp_data.keys()}\n")  # Should include 'periods' and 'teams'

    print(f"pbp_data['periods'] length: {len(pbp_data['periods'])}\n")
    dfg
    return
    for period in pbp_data['periods']:
        for play in period['plays']:
            for team in pbp_data['teams']:
                for player in team['players']:
                    if player['id'] == play['playerId']:
                        player_name = player['name']
                        action_type = play['type']
                        action_time = play['time']
                        # Add any other relevant details from the play data as needed
                        action_details = {
                            'type': action_type,
                            'time': action_time,
                            # Add other details here
                        }
                        if player_name not in player_stats:
                            player_stats[player_name] = []
                        player_stats[player_name].append(action_details)
    return player_stats



def main():
    pbp_data = get_pbp_data()
    print(f"Total number of games with play-by-play data: {len(pbp_data)}\n")
    game_stats = get_all_player_stats_of_game(pbp_data[0])
    print(f"Total number of players in the first game: {len(game_stats)}")

    print(f"Player stats for the first game:")
    for player, actions in game_stats.items():
        print(f"  {player}: {len(actions)} actions")

if __name__ == "__main__":
    main()