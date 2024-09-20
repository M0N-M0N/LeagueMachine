import sys
import time
from urllib.parse import urlencode
import pprint
import json

import requests

sys.path.append('ignoreme')

from ignoreme import keys

#Limit Api Requests
def rate_limited_request(max_requests, interval_seconds):
    current_time = time.time()

    if "last_request_time" not in rate_limited_request.__dict__:
        rate_limited_request.last_request_time = current_time
        rate_limited_request.request_count = 1
        return True
    elif current_time - rate_limited_request.last_request_time >= interval_seconds:
        rate_limited_request.last_request_time = current_time
        rate_limited_request.request_count = 1
        return True
    elif rate_limited_request.request_count < max_requests:
        rate_limited_request.request_count += 1
        return True
    else:
        return False

def get_from_riot(api_url):
    params = {
        'api_key': keys.API_KEY

    }

    if rate_limited_request(max_requests=99, interval_seconds=120):
        try:
            response = requests.get(api_url, params=urlencode(params))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"{e}")
            time.sleep(3)
    else:
        print("Rate limit exceeded. Please wait before making another request.")
        time.sleep(15)

        return None

def get_details_from_riot( riot_id = None, riot_tag = "EUW", region=keys.DEFAULT_REGION ):
    if not riot_id:
        riot_id = "Davemon130"
        riot_tag = "EUW"
    api_url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{riot_id}/{riot_tag}"

    return api_url

def get_matches_riot(puuid=None, region=keys.DEFAULT_REGION):
    if not puuid:
        puuid = "9trpv63KcBiFqvOteBeWGbwhjVMvpeVTD0_qmbHnglEMx40RHENewljMMj8y7wt1as2poRMz8EKP7w"

    api_url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count=5"

    return api_url

#needed to get champ details
def get_match_details(matchId=None, region=keys.DEFAULT_REGION):
    if not matchId:
        matchId = "EUW1_6993770668"

    api_url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{matchId}"

    return api_url

#get timeline
def get_match_timeline(matchId=None, region=keys.DEFAULT_REGION):
    if not matchId:
        matchId = "EUW1_6993770668"

    api_url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline"

    return api_url

#save to json
def json_match(summary, timeline):
    save_as_json("tempmatch", timeline)
    save_as_json("tempmatchsum", summary)

    return True

#do it all in one
def process_match( matchId, region = None ):
    match_sum = get_from_riot(get_match_details(matchId))
    match_info = get_from_riot(get_match_timeline(matchId))
    json_match(match_sum, match_info)

    return match_sum, match_info

def get_puuid(summonerid, region=keys.DEFAULT_REGION_CODE):
    api_url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summonerid}"
    return api_url


def get_bestinLeague(region=keys.DEFAULT_REGION_CODE):
    api_url = f"https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    return api_url

def save_as_json(filename, df):
    with open(f"{filename}.json", 'w') as json_file:
        json.dump(df, json_file)

    return True


if __name__ == '__main__':
    # riot_info = get_details_from_riot()
    # matches_info = get_from_riot(get_matches_riot())
    # match_sum = get_from_riot(get_match_details())
    # match_info = get_from_riot(get_match_timeline())
    challenger_list = get_from_riot(get_bestinLeague())
    save_as_json("challenger_list", challenger_list)
    # pprint.pprint(match_sum)

