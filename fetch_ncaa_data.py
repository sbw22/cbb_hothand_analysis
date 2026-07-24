import http.client
import json
from datetime import datetime, timedelta
from math import exp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests  # session-based, pools connections + handles gzip automatically
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class NCAADataFetcher:

    def __init__(self):
        self.API_HOST = "ncaa-api.henrygd.me"
        self.SEASON_START = datetime(year=2025, month=11, day=3)
        self.SEASON_START = datetime(year=2026, month=1, day=1) # switched to Jan 1 for testing so we get conference games in the sample
        self.SEASON_END = datetime(year=2026, month=4, day=15)
        self.MAX_TEST_GAMES = 50
        self.team_def_3pt_pct_bank = {}

    # Centralize the API host and season boundaries so the rest of the script can
    # refer to a single source of truth instead of repeating literal values.



    '''def get_game_ids(self):

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
            if counter > 500: #####################################################################################################################
                break
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
                if counter > 500: ######################################################################################################################
                    break
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
        
        return list_of_game_ids'''

    def _build_session(self):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,       # waits 1.5s, 3s, 6s, 12s, 24s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)"})
        return session

    def get_game_ids(self):
        session = self._build_session()
        base_url = f"https://{self.API_HOST}"

        dates = []
        d = self.SEASON_START
        while d < self.SEASON_END:
            dates.append(d.strftime("%Y/%m/%d"))
            d += timedelta(days=1)

        def fetch_day(date_str):
            try:
                resp = session.get(f"{base_url}/scoreboard/basketball-men/d1/{date_str}/all-conf", timeout=20)
                resp.raise_for_status()
                return date_str, resp.json()
            except Exception as e:
                print(f"Failed on {date_str}: {e}")
                return date_str, None

        list_of_game_ids = []
        failed_dates = []
        with ThreadPoolExecutor(max_workers=5) as executor:   # <-- much gentler
            futures = {executor.submit(fetch_day, d): d for d in dates}
            for future in as_completed(futures):
                date_str, json_data = future.result()
                if not json_data:
                    failed_dates.append(date_str)
                    continue
                for game in json_data['games']:
                    list_of_game_ids.append(game['game']['gameID'])

        # Second pass, serial, for anything that still failed after retries
        if failed_dates:
            print(f"Retrying {len(failed_dates)} failed dates serially...")
            for date_str in failed_dates:
                _, json_data = fetch_day(date_str)
                if json_data:
                    for game in json_data['games']:
                        list_of_game_ids.append(game['game']['gameID'])

        return list_of_game_ids


    '''def get_pbp_data(self):

        # Open a second connection for the play-by-play requests.
        conn = http.client.HTTPSConnection(self.API_HOST)

        # Use only a small subset of games while debugging so the API load stays low.
        # list_of_game_ids = self.get_game_ids()[:self.MAX_TEST_GAMES]
        list_of_game_ids = self.get_game_ids()

        # Give a quick progress summary before making the per-game requests.
        # print(f"Found {len(list_of_game_ids)} game IDs for testing.")

        # Store one parsed play-by-play payload per game.
        list_of_pbp_data = []
        list_of_boxscore_data = []
        for game_id in list_of_game_ids:
            # Request the play-by-play endpoint for one specific contest.
            conn.request("GET", f"/game/{game_id}/play-by-play")

            # Read and decode the response so JSON parsing can happen safely.
            res_pbp = conn.getresponse()
            data_pbp = res_pbp.read()
            decoded_data_pbp = data_pbp.decode("utf-8")
            try:
                # Parse the JSON payload into a Python dictionary.
                pbp_json_data = json.loads(decoded_data_pbp)
                list_of_pbp_data.append(pbp_json_data)
            except json.JSONDecodeError:
                # If a game request fails, print the raw response and keep going.
                print(f"Response for game ID {game_id} was not valid JSON")
                print(decoded_data_pbp)
                continue

            conn.request("GET", f"/game/{game_id}/boxscore")
            res_boxscore = conn.getresponse()
            data_boxscore = res_boxscore.read()
            decoded_data_boxscore = data_boxscore.decode("utf-8")
            try:
                boxscore_json_data = json.loads(decoded_data_boxscore)
                list_of_boxscore_data.append(boxscore_json_data)
            except json.JSONDecodeError:
                print(f"Response for game ID {game_id} was not valid JSON")
                print(decoded_data_boxscore)
                continue
        
        # If every request failed, return an empty list instead of indexing into nothing.
        if not list_of_pbp_data or not list_of_boxscore_data:
            print("No play-by-play data was returned for any game.")
            conn.close()
            return [], []
        

        # Close the connection after the batch is complete.
        conn.close()

        return list_of_pbp_data, list_of_boxscore_data'''
    
    def get_pbp_data(self):
        list_of_game_ids = self.get_game_ids()
        session = self._build_session()
        base_url = f"https://{self.API_HOST}"

        def fetch_game(game_id):
            try:
                pbp_resp = session.get(f"{base_url}/game/{game_id}/play-by-play", timeout=20)
                pbp_resp.raise_for_status()
                pbp_json = pbp_resp.json()
                box_resp = session.get(f"{base_url}/game/{game_id}/boxscore", timeout=20)
                box_resp.raise_for_status()
                box_json = box_resp.json()
                return game_id, pbp_json, box_json
            except Exception as e:
                print(f"Failed on game {game_id}: {e}")
                return game_id, None, None

        results = {}
        failed_ids = []
        with ThreadPoolExecutor(max_workers=8) as executor:   # <-- also gentler
            futures = {executor.submit(fetch_game, gid): gid for gid in list_of_game_ids}
            for future in as_completed(futures):
                game_id, pbp_json, box_json = future.result()
                if pbp_json is not None and box_json is not None:
                    results[game_id] = (pbp_json, box_json)
                else:
                    failed_ids.append(game_id)

        if failed_ids:
            print(f"Retrying {len(failed_ids)} failed games serially...")
            for gid in failed_ids:
                _, pbp_json, box_json = fetch_game(gid)
                if pbp_json is not None and box_json is not None:
                    results[gid] = (pbp_json, box_json)

        list_of_pbp_data = [results[gid][0] for gid in list_of_game_ids if gid in results]
        list_of_boxscore_data = [results[gid][1] for gid in list_of_game_ids if gid in results]

        return list_of_pbp_data, list_of_boxscore_data
    

    def add_team_def_3pt_pct_to_bank(self, boxscore_data):
        # print(f"boxscore_data['teamBoxscore'][0]['teamStats']: {boxscore_data['teamBoxscore'][0]['teamStats']}"
        #team1_3pt_pct = float(boxscore_data['teamBoxscore'][0]['teamStats']['threePointPercentage'][:-1])/100 if boxscore_data['teamBoxscore'][0]['teamStats']['threePointPercentage'] != '' else 0.0 # Assume the team has a 0% 3pt percentage (didn't shoot any 3pt shots)
        # team2_3pt_pct = float(boxscore_data['teamBoxscore'][1]['teamStats']['threePointPercentage'][:-1])/100 if boxscore_data['teamBoxscore'][1]['teamStats']['threePointPercentage'] != '' else 0.0 # Assume the team has a 0% 3pt percentage (didn't shoot any 3pt shots)
        try: 
            team1_3pt_pct = float(boxscore_data['teamBoxscore'][0]['teamStats']['threePointPercentage'][:-1])/100
        except:
            print(f"Error: boxscore_data['teamBoxscore'][0]['teamStats']['threePointPercentage'] is not a valid float")
            print(f"boxscore_data['teamBoxscore'][0]['teamStats']: {boxscore_data['teamBoxscore'][0]['teamStats']}")
            print(f"self.team_def_3pt_pct_bank: {self.team_def_3pt_pct_bank}")
            team1_3pt_pct = 0.0 # Assume the team has a 0% 3pt percentage (didn't shoot any 3pt shots)
        try:
            team2_3pt_pct = float(boxscore_data['teamBoxscore'][1]['teamStats']['threePointPercentage'][:-1])/100
        except:
            print(f"Error: boxscore_data['teamBoxscore'][1]['teamStats']['threePointPercentage'] is not a valid float")
            print(f"boxscore_data['teamBoxscore'][1]['teamStats']: {boxscore_data['teamBoxscore'][1]['teamStats']}")
            print(f"self.team_def_3pt_pct_bank: {self.team_def_3pt_pct_bank}")
            team2_3pt_pct = 0.0 # Assume the team has a 0% 3pt percentage (didn't shoot any 3pt shots)
        # print(f"team1_3pt_pct: {team1_3pt_pct}, team2_3pt_pct: {team2_3pt_pct}")
        team1_id = boxscore_data['teams'][0]['teamId']
        team2_id = boxscore_data['teams'][1]['teamId']
        if team1_id not in self.team_def_3pt_pct_bank:
            self.team_def_3pt_pct_bank[team1_id] = []
        if team2_id not in self.team_def_3pt_pct_bank:
            self.team_def_3pt_pct_bank[team2_id] = []

        box_id1 = boxscore_data['teamBoxscore'][0]['teamId']
        box_id2 = boxscore_data['teamBoxscore'][1]['teamId']
        self.team_def_3pt_pct_bank[team1_id].append(team2_3pt_pct if box_id1 == team1_id else team1_3pt_pct) if team2_3pt_pct != 0.0 else None # Append the opponent's 3pt percentage
        self.team_def_3pt_pct_bank[team2_id].append(team1_3pt_pct if box_id2 == team2_id else team2_3pt_pct) if team1_3pt_pct != 0.0 else None
        # print(f"team_def_3pt_pct_bank: {self.team_def_3pt_pct_bank}")

        return team1_id, team2_id


    def get_all_player_stats_of_game(self, pbp_data, boxscore_data):
        # print(f"boxscore_data: {boxscore_data}")

        team1_id, team2_id = self.add_team_def_3pt_pct_to_bank(boxscore_data) # Get the team IDs and add the opponent's 3pt percentage to the bank
        
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
                # print(f"play: {play}")
                # print(f"play_list: {play_list}")
                team_id = str(play['teamId'])
                if team_id not in self.team_def_3pt_pct_bank.keys():
                    print(f"keys in team_def_3pt_pct_bank: {self.team_def_3pt_pct_bank.keys()}")
                    print(f"team_id: {team_id} not found in team_def_3pt_pct_bank")
                    continue
                opp_def_3pt_pct_avg = sum(self.team_def_3pt_pct_bank[team1_id])/len(self.team_def_3pt_pct_bank[team1_id]) if team_id == team2_id else sum(self.team_def_3pt_pct_bank[team2_id])/len(self.team_def_3pt_pct_bank[team2_id]) # Calculate the average 3pt percentage for the team with team1_id if the team_id is team2_id, otherwise calculate the average 3pt percentage for the team with team2_id
                # print(f"opp_def_3pt_pct_avg: {opp_def_3pt_pct_avg}")
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
                    player_stats[player_name] = {'two_point_sequence': [], 'three_point_sequence': [], 'clock_time_sequence_two_point': [], 'clock_time_sequence_three_point': [], 'is_home_sequence_two_point': [], 'is_home_sequence_three_point': [], 'opp_def_3pt_pct_avg': []} # 0 = make, 1 = miss

                
                # Record whether the shot was a make or miss (0 or 1) in the appropriate sequence list for the player. This is a simple way to 
                # track the player's shooting performance over time, and we can analyze these sequences later to identify hot hand patterns.
                if 'makes' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                    # print(f"Found a made 3pt shot: {play}")
                    player_stats[player_name]['three_point_sequence'].append(0)
                    player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
                    player_stats[player_name]['opp_def_3pt_pct_avg'].append(opp_def_3pt_pct_avg)
                elif 'misses' in play_list and 'three' in play_list and 'point' in play_list and 'shot' in play_list:
                    # print(f"Found a missed 3pt shot: {play}")
                    player_stats[player_name]['three_point_sequence'].append(1)
                    player_stats[player_name]['clock_time_sequence_three_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_three_point'].append(is_home)
                    player_stats[player_name]['opp_def_3pt_pct_avg'].append(opp_def_3pt_pct_avg)
                elif 'makes' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                    # print(f"Found a made 2pt shot: {play}")
                    player_stats[player_name]['two_point_sequence'].append(0)
                    player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
                    player_stats[player_name]['opp_def_3pt_pct_avg'].append(opp_def_3pt_pct_avg) # CHANGE THIS LATER IF WE START USING 2PT SHOTS
                elif 'misses' in play_list and 'two' in play_list and 'point' in play_list and 'jump' in play_list and 'shot' in play_list:
                    # print(f"Found a missed 2pt shot: {play}")
                    player_stats[player_name]['two_point_sequence'].append(1)
                    player_stats[player_name]['clock_time_sequence_two_point'].append(clock_time)
                    player_stats[player_name]['is_home_sequence_two_point'].append(is_home)
                    player_stats[player_name]['opp_def_3pt_pct_avg'].append(opp_def_3pt_pct_avg) # CHANGE THIS LATER IF WE START USING 2PT SHOTS
                else:
                    continue
                
                
                skip_counter = True # Set the skip counter to True so that the next event will be skipped, 
                # which should prevent us from processing the duplicate event in the play-by-play data.
                counter += 1

        return player_stats