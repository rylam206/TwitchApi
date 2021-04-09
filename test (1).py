"""
Ryan Lam and Shi Ming Kuang
CSE 163
Final Project

This is a module meant for testing code from twitch.py and main.py
"""

import json
import requests
import twitch
import pandas as pd
import random


def test_twitch_authentication():
    """
    This is a test function for Twitch authentication of client credentials
    which is tested on an example endpoint with query 'a_seagull'.
    Printing successful string representations of json objects with
    the example query and endpoint allow for further implementations
    of the Twitch API; treat this function as a mini/simpler version
    of the twitch.py module.
    """
    url = 'https://api.twitch.tv/helix/'
    client_id = 'bkhohw0wmcr9r4g0dhe60wandg3dgr'
    client_secret = 'juhmt6ld36xdjpx0r12p6w5aokh9dc'
    request_accesss = 'https://id.twitch.tv/oauth2/token?client_id='
    request_accesss += client_id
    request_accesss += '&client_secret='
    request_accesss += client_secret
    request_accesss += '&grant_type=client_credentials'
    access_code = requests.post(request_accesss)
    access_token = json.loads(access_code.text)
    print(access_token)
    access_token = access_token['access_token']
    print('Access Token:', access_token)
    print()
    headers = {
        'client-id': client_id, 'Authorization': 'Bearer ' + str(access_token)
    }
    endpoint = 'search/channels?query='
    query = 'a_seagull'
    endpoint += query
    response = requests.get(url + endpoint, headers=headers)
    response_json = response.json()  # json object
    print(response_json)
    print()
    # convert Python object to json string
    print_response = json.dumps(response_json, indent=4)
    print('json string', print_response)


def test_get_game_query():
    """
    This is a test function for the get_games_query function
    in twitch.py
    """
    game_name = 'valorant'
    endpt = twitch.get_games_query(game_name)
    response = twitch.get_response(endpt)
    print(twitch.response_to_string(response))
    print(game_name, 'id:', twitch.get_game_id(response))


def test_male_vs_female(data):
    count = 0
    total = 1000
    male_id = [19571641, 60056333, 37402112, 39276140, 71092938]
    female_id = [44445592, 125387632, 70357283, 41245072, 31106024]
    data = pd.read_csv('streamers/streamers.csv')
    for i in range(0, total):
        male_rand = random.choice(male_id)
        female_rand = random.choice(female_id)
        male = data[(data['gender'] == 'male') &
                    (data['id'] == male_rand)]
        female = data[(data['gender'] == 'female') &
                      (data['id'] == female_rand)]
        index = male['view_count'].idxmax()
        male_view = male.loc[index, 'view_count']
        index = female['view_count'].idxmax()
        female_view = female.loc[index, 'view_count']
        if(male_view > female_view):
            count += 1
    return count / total


def main():
    test_twitch_authentication()
    test_get_game_query()
    print(test_male_vs_female('streamers/streamers.csv'))


if __name__ == '__main__':
    main()
