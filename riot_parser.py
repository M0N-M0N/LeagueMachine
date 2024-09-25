import pprint

import pandas as pd
import numpy as np
import time
import json
import RiotApi as Riot
import os

import cv2

def read_tempmatches():
    df_timeline = pd.read_json('tempmatch.json')
    df_summary = pd.read_json('tempmatchsum.json')

    return df_summary, df_timeline

def time_converter(timestring):
    total_seconds = timestring // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    time_format =  f"{minutes:02}:{seconds:02}"

    return total_seconds, time_format

#get the per min stat of the players/participants
def snapshot_per_min(df, frame=20):
    investigate_time = df["info"]["frames"][frame]["timestamp"]
    investigate_events = df["info"]["frames"][frame]["participantFrames"]

    return investigate_time, investigate_events

def format_timeline_data(df_timelines,df_summary):
    match_players = {"matchId":df_summary["metadata"]["matchId"]}
    # match_players2 = {"matchId":df_summary["metadata"]["matchId"]}
    match_players3 = {f"matchId_{df_summary["metadata"]["matchId"]}":{}}
    k,d,a = 0,0,0
    # pprint.pprint()
    if df_summary["info"]["gameMode"] == "CLASSIC":
        for participant in df_summary["info"]["participants"]:
            match_players.update({participant["participantId"]:{
                "puid": participant["puuid"],
                "championName": participant["championName"],
                "championId": participant["championId"],
                "role": participant["role"],
                "playerSubteamId": participant["playerSubteamId"],
                "subteamPlacement": participant["subteamPlacement"],
                "win": participant["win"],
                "teamId": participant["teamId"],
                "teamPosition": participant["teamPosition"],
                "kda": f"{participant["kills"]}/{participant["deaths"]}/{participant["assists"]}"
            }})
            # match_players2.update({f"participant_{participant["participantId"]}":{
            #     "puid": participant["puuid"],
            #     "championName": participant["championName"],
            #     "championId": participant["championId"],
            #     "role": participant["role"],
            #     "playerSubteamId": participant["playerSubteamId"],
            #     "subteamPlacement": participant["subteamPlacement"],
            #     "win": participant["win"],
            #     "teamId": participant["teamId"],
            #     "teamPosition": participant["teamPosition"],
            #     "kda": f"{participant["kills"]}/{participant["deaths"]}/{participant["assists"]}"
            # }})
            # print(get_position(participant["teamPosition"]))
            match_players3[f"matchId_{df_summary["metadata"]["matchId"]}"].update({
                f"participant_{participant["participantId"]}": {
                    f"role_{participant["role"]}": {f"teamId_{participant["teamId"]}": {
                        f"position_{get_position(participant["teamPosition"])}": {}}}}})

        # pprint.pprint(match_players3)


        for frame, frame_details in enumerate(df_timelines["info"]["frames"]):
            # break
            # print(frame_details)
            # pprint.pprint(frame_details.keys())

            for key, value in frame_details["participantFrames"].items():
                if frame != 0:
                    # pprint.pprint(match_players[value["participantId"]])
                    k = match_players[value["participantId"]][f"{frame-1}"]["kills"]
                    d = match_players[value["participantId"]][f"{frame-1}"]["deaths"]
                    a = match_players[value["participantId"]][f"{frame-1}"]["assists"]
                # print(value["participantId"])
                # break
                role = match_players[value["participantId"]]["role"]
                teamId = match_players[value["participantId"]]["teamId"]
                teamPosition = get_position(match_players[value["participantId"]]["teamPosition"])
                kda = f"{k}/{d}/{a}"

                # print(f"{key}: {value}")
                # pprint.pprint(value)
                frame_dict = {
                    "currentGold": value["currentGold"],
                    "level": value["level"],
                    "cs": value["minionsKilled"],
                    "totalGold": value["totalGold"],
                    "jungleMinionsKilled": value["jungleMinionsKilled"],
                    "kills": k,
                    "deaths": d,
                    "assists": a,
                    "game-time": f"{frame} minutes"
                }

                frame_dict2 = {
                    f"{frame}" : f"Current-Gold : {value["currentGold"]}, "
                                             f"Level : {value["level"]}, "
                                             f"CS : {value["minionsKilled"]}, "
                                             f"Total-Gold : {value["totalGold"]}, "
                                             f"JungleMinionsKilled : {value["jungleMinionsKilled"]}, "
                                             f"Kills : {k}, "
                                             f"Deaths : {d}, "
                                             f"Assists : {a}, "
                                             f"game-time: {frame} minutes"

                }
                match_players[value["participantId"]].update({f"{frame}": frame_dict})
                temp = match_players3[f"matchId_{df_summary["metadata"]["matchId"]}"][f"participant_{value["participantId"]}"][f"role_{role}"][f"teamId_{teamId}"][f"position_{teamPosition}"]
                # print(match_players2[value["participantId"]][)
                # if "minutes" in temp:
                #     # match_players2[f"participant_{value["participantId"]}"]["minutes"].update(frame_dict2)
                #     temp["minutes"].update(frame_dict2)
                # else:
                #     # match_players2[f"participant_{value["participantId"]}"].update({"minutes":frame_dict2})
                #     temp.update({"minutes":frame_dict2})
                temp.update(frame_dict2)
                # pprint.pprint(match_players3)
                # quit()

            for event in frame_details["events"]:
                # print(event)
                if event["type"] == "CHAMPION_KILL":
                    # pprint.pprint(event)
                    #add to the assist counter of a player
                    if "assistingParticipantIds" in event:
                        for assist in event["assistingParticipantIds"]:
                            match_players[assist][f"{frame}"].update({"assists" : match_players[assist][f"{frame}"]["assists"] +1 })
                    #add to kill count
                    if event["killerId"] != 0:
                        match_players[event["killerId"]][f"{frame}"].update({"kills" : match_players[event["killerId"]][f"{frame}"]["kills"] +1 })
                    #add to death count
                    match_players[event["victimId"]][f"{frame}"].update({"deaths" : match_players[event["victimId"]][f"{frame}"]["deaths"] +1 })
                    # break
    else:
        print(f"A game of {df_summary["info"]["gameMode"]} is not accepted")
        return None

    # print("match_players2")
    # pprint.pprint(match_players2)
    return match_players3

#get the common names for team positions
def get_position(position):
    match position:
        case 'BOTTOM':
            return "adc"
        case 'UTILITY':
            return "support"
        case 'MIDDLE':
            return "midlane"
        case 'JUNGLE':
            return "jungler"
        case 'TOP':
            return "top"
        case _:
            return position

#get the puuids from a list of challenger info. summoner id does not give match details
def get_challenger_puuids():
    df_matches = pd.read_json('challenger_list.json')
    best_players_sumId = []
    best_players_puuId = []
    # pprint.pprint(df_matches.iloc[:, :3])
    #sort challenger by their ELO
    df_sorted_matches = df_matches.sort_values(by="entries", key=lambda k: k.str["leaguePoints"], ascending=False)

    #get top 200 players
    for entry in df_sorted_matches.iloc[:200]["entries"]:
        if entry["inactive"] == False:
            # print(entry)
            best_players_sumId.append(entry["summonerId"])


    # pprint.pprint(best_player[1])
    # get_from_riot(get_matches_riot())

    #get the puuid from summoner id
    for player in best_players_sumId:
        success = False
        while not success:
            puuid = Riot.get_from_riot(Riot.get_puuid(player))
            best_players_puuId.append(puuid["puuid"])
            print(puuid["puuid"])
            success = True



    Riot.save_as_json("challenger_puuids", best_players_puuId)

    return True

def get_best_matches():
    puuids = pd.read_json('challenger_puuids.json')
    best_player_matches = []
    details = []
    # pprint.pprint(puuids.iloc[:5,0])
    for puuid in puuids.iloc[:,0]:
        # print(puuid)
        success = False
        while not success:
            details = Riot.get_from_riot(Riot.get_matches_riot(puuid))
            if details:
                for detail in details:
                    if detail not in best_player_matches:
                        best_player_matches.append(detail)
                        pprint.pprint(detail)
                success = True


    pprint.pprint(len(best_player_matches))
    Riot.save_as_json("best_player_matches", best_player_matches)
    return best_player_matches

def save_match2Json(matchId):
    match_sum, match_info = Riot.process_match(matchId)
    ret_value = False
    ret_Text = ""

    if match_sum and match_info:
        match_players = format_timeline_data(match_info, match_sum)
        if match_players:
            matchid = next(iter(match_players))[8:]
            tojson_success = Riot.save_as_json(f"matchesjson/{matchid}",match_players)
            if tojson_success:
                ret_value, ret_Text = True, f"Match saved to {matchid}"
        else:
            ret_value, ret_Text = True, f"Match not saved"
    else:
        ret_value, ret_Text = False, "Missing match summary or info"

    return ret_value, ret_Text

def get_matches_details_batch(): #process_match
    matchids = pd.read_json('best_player_matches.json')
    for matchid in matchids.iloc[:,0]:
        # print(matchid)
        success = False
        match_players = {}
        while not success:
            # print(puuid)
            print(f"Processing matchId: {matchid}")
            is_processed, save_msg = save_match2Json(matchid)
            print(save_msg)
            if is_processed:
                success = True

    # pprint.pprint(best_player_matches)
    return True

#make the json flat just for easier llm context feeding
def flatten_json(json_obj, parent_key='', sep='_'):
    items = {}
    for k, v in json_obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep))
        else:
            items[new_key] = v
    return items

#json processor for flattening
def process_json_file(json_file_path, output_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            flattened_dict = flatten_json(json_data)

            with open(output_file_path, 'w') as output_file:
                json.dump(flattened_dict, output_file, indent=4)
            print(f"Flattened data saved to {output_file_path}")

    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {json_file_path}")

def flatten_mathesjson():
    # Specify the folder path containing your JSON files
    folder_path = 'matchesjson/temp'

    for filename in os.listdir(folder_path):
        # output_name = os.path.splitext(filename)[0]
        if filename.endswith('.json'):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(folder_path, f"flat_{filename}")
            # print(input_file_path, output_file_path)
            process_json_file(input_file_path, output_file_path)












# print(df["info"]["participants"])
# investigate = df["info"]["frames"][0]["participantFrames"]["1"]
# investigate_time = df["info"]["frames"][20]["timestamp"]
# investigate_events = df["info"]["frames"][5]["participantFrames"]
# investigate_players = df["info"]["frames"][5]["events"]
# print(investigate)
# print(time_converter(investigate_time))
# pprint.pprint(investigate_players)
# pprint.pprint(investigate_events)
# df_edit = format_timeline_data(df_timeline,df_summary)
# pprint.pprint(df_edit)
# with open(f"matchesjson/{df_edit["matchId"]}.json", 'w') as json_file:
# json.dump(df_edit, json_file)
# pprint.pprint(df_summary["info"]["participants"][0].keys())


# get_challenger_puuids()
# get_best_matches()
# save_match2Json("EUW1_6999974885")
get_matches_details_batch()
# flatten_mathesjson()


# temp = pd.read_json("matchesjson/temp/flat_EUW1_6994400145.json", typ='dictionary')
# pprint.pprint(temp)
# # print(temp)

# print(cv2.getBuildInformation())
# print(cv2.__file__)
