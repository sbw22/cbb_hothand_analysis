import http.client
import json
from datetime import datetime, timedelta


def find_game_id(scoreboard_json, team1, team2):
    for game in scoreboard_json['games']:
        if team1 in game['home']['names']['short'] and team2 in game['away']['names']['short']:
            return game['game_id']

def main():

    conn = http.client.HTTPSConnection("ncaa-api.henrygd.me")

    # Define a specific date (e.g., 2024-12-31)
    specific_date = datetime(year=2025, month=11, day=3) # First day of the 2025-26 season
    # Grab game dates for all games in the 2025-26 season
    list_of_game_ids = []
    counter = 0
    while specific_date < datetime(year=2026, month=4, day=15):
        counter += 1
        if counter == 5: # Only check the first 5 days of the season to avoid making too many API calls while testing
            break
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


if __name__ == "__main__":
    main()