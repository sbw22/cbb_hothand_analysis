import http.client
import json
from datetime import datetime, timedelta
from math import exp
import numpy as np

class NCAADataFetcher:

    def __init__(self):
        self.API_HOST = "ncaa-api.henrygd.me"
        self.SEASON_START = datetime(year=2025, month=11, day=3)
        self.SEASON_START = datetime(year=2026, month=1, day=1) # switched to Jan 1 for testing so we get conference games in the sample
        self.SEASON_END = datetime(year=2026, month=4, day=15)
        self.MAX_TEST_GAMES = 50

    # Centralize the API host and season boundaries so the rest of the script can
    # refer to a single source of truth instead of repeating literal values.



    def get_game_ids(self):

        # Open a reusable HTTPS connection to the NCAA API host.
        conn = http.client.HTTPSConnection(self.API_HOST)

        # Start from the first day of the season and collect game IDs one date at a time.
        specific_date = self.SEASON_START
        # This list accumulates every game ID discovered from the scoreboard endpoint.
        list_of_game_ids = []
        # Temporary guard that keeps the script from iterating too far while testing.
        counter = 0
        while specific_date < self.SEASON_END:
            # Only inspect the first date in this reduced testing mode.
            '''if counter > 25: #####################################################################################################################
                break'''
            # Format the date to match the API path shape YYYY/MM/DD.
            date_str = specific_date.strftime("%Y/%m/%d")
            # Request the daily scoreboard for Division I men's basketball.
            conn.request("GET", f"/scoreboard/basketball-men/d1/{date_str}/all-conf") # switched all-conf to big-12
            # Read the HTTP response body so it can be parsed as JSON.
            res = conn.getresponse()
            data = res.read()
            decoded_data = data.decode("utf-8")
            try:
                # Convert the raw response text into a Python dictionary.
                json_data = json.loads(decoded_data)
            except json.JSONDecodeError:
                # If the API returns HTML or any non-JSON payload, stop and surface it.
                print("Response was not valid JSON")
                print(decoded_data)
                return

            # The scoreboard response contains a `games` array; each item wraps a game object.

            for game in json_data['games']:
                # Pull out the unique game identifier that will be used for play-by-play calls.
                game_id = game['game']['gameID']
                home_team = game['game']['home']
                away_team = game['game']['away']
                home_team_conference = home_team['conferences'][0]['conferenceSeo']
                away_team_conference = away_team['conferences'][0]['conferenceSeo']
                if away_team_conference != "big-12" and home_team_conference != "big-12":
                    continue
                counter += 1 # Only trigger counter if we find a game with a big 12 team
                '''if counter > 25: ######################################################################################################################
                    break'''
                list_of_game_ids.append(game_id)
                print(f"len of list_of_game_ids: {len(list_of_game_ids)}, counter value: {counter}")
                # print(f"keys in game object: {game['game'].keys()}")
                # print(f"home team: {game['game']['home']}")#  , visitor team: {game['game']['away']}")
                # print(f"big 12 team found")

            # Move forward one day and continue scanning the season schedule.
            specific_date += timedelta(days=1)
            if len(json_data['games']) == 0:
                print(f"No games found for {date_str}")
        
        # Close the connection before returning the accumulated game IDs.
        conn.close()
        
        return list_of_game_ids


    def get_pbp_data(self):

        # Open a second connection for the play-by-play requests.
        conn = http.client.HTTPSConnection(self.API_HOST)

        # Use only a small subset of games while debugging so the API load stays low.
        list_of_game_ids = self.get_game_ids()[:self.MAX_TEST_GAMES]
        list_of_game_ids = self.get_game_ids()

        # Give a quick progress summary before making the per-game requests.
        # print(f"Found {len(list_of_game_ids)} game IDs for testing.")

        # Store one parsed play-by-play payload per game.
        list_of_pbp_data = []
        for game_id in list_of_game_ids:
            # Request the play-by-play endpoint for one specific contest.
            conn.request("GET", f"/game/{game_id}/play-by-play")
            # Read and decode the response so JSON parsing can happen safely.
            res = conn.getresponse()
            data = res.read()
            decoded_data = data.decode("utf-8")
            try:
                # Parse the JSON payload into a Python dictionary.
                pbp_json_data = json.loads(decoded_data)
                list_of_pbp_data.append(pbp_json_data)
            except json.JSONDecodeError:
                # If a game request fails, print the raw response and keep going.
                print(f"Response for game ID {game_id} was not valid JSON")
                print(decoded_data)
                continue

        
        # If every request failed, return an empty list instead of indexing into nothing.
        if not list_of_pbp_data:
            print("No play-by-play data was returned for any game.")
            conn.close()
            return []

        # Close the connection after the batch is complete.
        conn.close()

        return list_of_pbp_data


    def get_all_player_stats_of_game(self, pbp_data):
        # This dictionary is the output of the function: each key is a player-like label,
        # and each value is the chronological list of events we were able to associate
        # with that label.
        player_stats = {}

        # The play-by-play payload is organized by periods, so if that top-level field
        # is missing there is nothing meaningful to walk and the function can exit early.
        if 'periods' not in pbp_data:
            print(f"'periods' key not found in pbp_data\n")
            return {}

        
        # For some reason, each play is duplicated in the play-by-play data, so for now I am just going to skip the next event when we record an event.
        skip_counter = False

        # Walk every period, then each play within the period, and attach the play to the
        # best actor label we can infer from the payload.
        for period in pbp_data['periods']:

            # My code starts here
            counter = 0
            for play in period['playbyplayStats']:
                # print(f"")
                
                if skip_counter:
                    skip_counter = False
                    continue

                play_list = play['eventDescription'].split()
                is_home_bool = bool(play['isHome'])
                is_home = 0 if is_home_bool else 1 # 0 if at home, 1 if away
                # clock_time_str = play['clock']
                clock_time_list = play['clock'].split(":")
                clock_time = (int(clock_time_list[0]) * 60) + int(clock_time_list[1]) # The total amount of seconds left on the clock
                clock_time /= 1200 # Normalizing the clock value to be between 0 and 1
                # Sometimes, the player's last name is concatenated with 'makes' or 'misses' in the event description, 
                # so we need to split those apart to get the player's name and the shot result as separate tokens.
                if play_list[1][-6:] == "misses":
                    # split apart players last name and the 'misses' string
                    play_list[1:2] = [play_list[1][:-6], 'misses']
                elif play_list[1][-5:] == "makes":
                    # split apart players last name and the 'makes' string
                    play_list[1:2] = [play_list[1][:-5], 'makes']

                player_name = play_list[0] + " " + play_list[1]


                # Check if the event description contains the keywords that indicate a shot event, and print it if found.
                # If not, skip to the next event. This is a quick way to filter for shot events without needing to parse 
                # every single event description in detail.
                if not (set(['three', 'two', 'point', 'shot']) & (set(play_list))):
                    continue
                if (set(['blocks', 'layup', 'free', 'throw']) & (set(play_list))):
                    continue
                # If the player name is one of the following, skip it because those are not actual players and will just 
                # add noise to our player-level stats.
                if player_name in ["TV timeout", "Subbing out", "Subbing in", "End of", "Team defensive", "Team offensive"]:
                    continue

                '''if counter == 10:
                    print(f"play: {play}")
                    wefds'''
                if player_name not in player_stats:
                    player_stats[player_name] = {'two_point_sequence': [], 'three_point_sequence': [], 'clock_time_sequence_two_point': [], 'clock_time_sequence_three_point': [], 'is_home_sequence_two_point': [], 'is_home_sequence_three_point': []} # 0 = make, 1 = miss

                
                # Record whether the shot was a make or miss (0 or 1) in the appropriate sequence list for the player. This is a simple way to 
                # track the player's shooting performance over time, and we can analyze these sequences later to identify hot hand patterns.
                if 'makes' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                    # print(f"Found a made 3pt shot: {play}")
                    player_stats[player_name]['three_point_sequence'].append(0)
                    player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
                elif 'misses' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                    # print(f"Found a missed 3pt shot: {play}")
                    player_stats[player_name]['three_point_sequence'].append(1)
                    player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
                elif 'makes' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                    # print(f"Found a made 2pt shot: {play}")
                    player_stats[player_name]['two_point_sequence'].append(0)
                    player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
                elif 'misses' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                    # print(f"Found a missed 2pt shot: {play}")
                    player_stats[player_name]['two_point_sequence'].append(1)
                    player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
                else:
                    continue
                
                
                skip_counter = True # Set the skip counter to True so that the next event will be skipped, 
                # which should prevent us from processing the duplicate event in the play-by-play data.
                counter += 1

        return player_stats