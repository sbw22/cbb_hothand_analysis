import http.client
import json
from datetime import datetime, timedelta


# Centralize the API host and season boundaries so the rest of the script can
# refer to a single source of truth instead of repeating literal values.
API_HOST = "ncaa-api.henrygd.me"
# The first day we want to inspect for the 2025-26 college basketball season.
SEASON_START = datetime(year=2025, month=11, day=3)
# The cutoff date for the season loop; the code walks day by day until this date.
SEASON_END = datetime(year=2026, month=4, day=15)
# Cap the number of games we fetch during testing so the script stays fast.
MAX_TEST_GAMES = 10


def get_game_ids():

    # Open a reusable HTTPS connection to the NCAA API host.
    conn = http.client.HTTPSConnection(API_HOST)

    # Start from the first day of the season and collect game IDs one date at a time.
    specific_date = SEASON_START
    # This list accumulates every game ID discovered from the scoreboard endpoint.
    list_of_game_ids = []
    # Temporary guard that keeps the script from iterating too far while testing.
    counter = 0
    while specific_date < SEASON_END:
        # Only inspect the first date in this reduced testing mode.
        if counter == 1:
            break
        counter += 1
        # Format the date to match the API path shape YYYY/MM/DD.
        date_str = specific_date.strftime("%Y/%m/%d")
        # Request the daily scoreboard for Division I men's basketball.
        conn.request("GET", f"/scoreboard/basketball-men/d1/{date_str}/all-conf")
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
            list_of_game_ids.append(game_id)

        # Move forward one day and continue scanning the season schedule.
        specific_date += timedelta(days=1)
    
    # Close the connection before returning the accumulated game IDs.
    conn.close()
    
    return list_of_game_ids


def get_pbp_data():

    # Open a second connection for the play-by-play requests.
    conn = http.client.HTTPSConnection(API_HOST)

    # Use only a small subset of games while debugging so the API load stays low.
    list_of_game_ids = get_game_ids()[:MAX_TEST_GAMES]

    # Give a quick progress summary before making the per-game requests.
    print(f"Found {len(list_of_game_ids)} game IDs for testing.")

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


def get_all_player_stats_of_game(pbp_data):
    # This dictionary is the output of the function: each key is a player-like label,
    # and each value is the chronological list of events we were able to associate
    # with that label.
    player_stats = {}

    # The play-by-play payload is organized by periods, so if that top-level field
    # is missing there is nothing meaningful to walk and the function can exit early.
    if 'periods' not in pbp_data:
        print(f"'periods' key not found in pbp_data\n")
        return {}

    def extract_actor_name(play):
        # Prefer the explicit first/last name fields because they are the most
        # structured signal in the payload and avoid string parsing when present.
        first_name = (play.get('firstName') or '').strip()
        last_name = (play.get('lastName') or '').strip()
        if first_name or last_name:
            # Combine the two fields into one display name. Some non-player events use
            # team names in these fields, so filter out obvious team-level labels.
            full_name = f"{first_name} {last_name}".strip()
            if full_name.lower() not in {'winthrop winthrop', 'queens (nc) queens (nc)'}:
                return full_name

        # Fall back to the human-readable event description when the structured name
        # fields are empty. This is less reliable, but it still gives us a usable key.
        event_description = (play.get('eventDescription') or '').strip()
        # Skip events that are not useful for player-level grouping, such as timeouts
        # and end-of-period markers.
        if not event_description or "time out" in event_description.lower() or event_description.lower().startswith("end of"):
            return None

        # Substitution events store the player name after a hyphen, for example
        # "Subbing in for Winthrop-Kareem Rozier".
        if event_description.startswith("Subbing in for ") or event_description.startswith("Subbing out for "):
            return event_description.split("-", 1)[-1].strip() if "-" in event_description else "Team"

        # Many scoring, rebound, steal, and foul descriptions use the "team's player"
        # format, so split on the possessive marker and keep the trailing name portion.
        if "'s " in event_description:
            candidate_name = event_description.rsplit("'s ", 1)[-1]
        elif "'s" in event_description:
            candidate_name = event_description.rsplit("'s", 1)[-1]
        else:
            # If the description does not mention a person at all, group it under a
            # generic Team bucket so the event is still counted instead of dropped.
            return "Team"

        # Strip trailing context such as "(draws the foul)" so the label stays stable.
        return candidate_name.split(" (", 1)[0].strip()

    # Walk every period, then each play within the period, and attach the play to the
    # best actor label we can infer from the payload.
    for period in pbp_data['periods']:
        for play in period.get('playbyplayStats', []):
            # Pull the raw event text once so we do not repeatedly look up the same field.
            event_description = (play.get('eventDescription') or '').strip()
            if not event_description:
                continue

            # Derive the grouping key for this event. The helper prefers structured
            # names first and only falls back to parsing text when necessary.
            player_name = extract_actor_name(play)
            if player_name is None:
                continue
            # Keep the original description plus a few useful metadata fields so later
            # analysis can inspect the raw event without querying the API again.
            action_details = {
                'description': event_description,
                'time': play.get('clock'),
                'teamId': play.get('teamId'),
                'score': play.get('score'),
            }

            # Create the player's list on first use, then append each event in the order
            # the API returned them, which is already chronological within each period.
            if player_name not in player_stats:
                player_stats[player_name] = []
            player_stats[player_name].append(action_details)
    return player_stats



def main():
    # Fetch the season sample, then summarize the first game as a quick smoke test.
    pbp_data = get_pbp_data()
    print(f"Finished getting play-by-play data for {len(pbp_data)} games.")
    # Analyze the first returned contest because the script is still in exploratory mode.
    game_stats = get_all_player_stats_of_game(pbp_data[0])
    print(f"Total number of players in the first game: {len(game_stats)}")

    print("Player stats for the first game:")
    for player, actions in game_stats.items():
        # Show how many recorded events were attached to each player-like key.
        print(f"  {player}: {len(actions)} actions")

if __name__ == "__main__":
    # Run the script only when executed directly, not when imported as a module.
    main()