"""
Ryan Lam and Shi Ming Kuang
CSE 163
Final Project

This program implements the Twitch API to scrape streamer data
from the platform. Most of the functions that are implemented
return specific endpoints for the Twitch API to work with and
scrape the desired data.
"""

import requests
import json
import pandas as pd

URL = 'https://api.twitch.tv/helix/'
CLIENT_ID = 'bkhohw0wmcr9r4g0dhe60wandg3dgr'
CLIENT_SECRET = 'juhmt6ld36xdjpx0r12p6w5aokh9dc'
AUTHORIZATION = 'https://id.twitch.tv/oauth2/token?client_id='
AUTHORIZATION += CLIENT_ID + '&client_secret='
AUTHORIZATION += CLIENT_SECRET + '&grant_type=client_credentials'
ACCESS_CODE = requests.post(AUTHORIZATION)
ACCESS_TOKEN = json.loads(ACCESS_CODE.text)
TOKEN = ACCESS_TOKEN['access_token']
HEADERS = {
    'client-id': CLIENT_ID, 'Authorization': 'Bearer ' + str(TOKEN)
}


def save_json_to_csv(response, name):
    """
    Takes in a GET request, converts the response to a json object,
    and returns a saved CSV file
    """
    response = response.json()
    x = []
    for item in response['data']:
        df = pd.DataFrame.from_dict([item])
        x.append(df)
    new = pd.concat(x, ignore_index=True)
    return new.to_csv(name + '.csv')


def get_response(query):
    """
    This function takes in a query (a specified endpoint)
    that is concatenated with overall API URL and return
    a GET request for the specified URL.
    """
    url = URL + query
    response = requests.get(url, headers=HEADERS)
    return response


def response_to_string(response):
    """
    This function takes in a GET request, converts the request object
    to json, and prints out
    the response of that request as a json object.
    """
    response_json = response.json()
    string = json.dumps(response_json, indent=2)
    return string


def get_user_query(user_login):
    """
    Takes in user_login of streamer and
    return the endpoint for getting information of
    the specified Twitch user.
    """
    return 'users?login={0}'.format(user_login)


def get_user_videos_query(user_id):
    """
    Takes in the user_id of a streamer and
    returns the endpoint for getting the streamer's
    VODs(videos on demand).
    """
    result = 'videos?user_id='
    result += str(user_id)
    result += '&first=100&type=archive'
    return result


def get_game_vids(game_id):
    """
    Takes in game_id and returns endpoint for accessing VODs
    for the game.
    """
    game_id_to_str = str(game_id)
    result = 'videos?game_id=' + game_id_to_str
    result += '&first=100&sort=views&type=archive'
    return result


def get_top_games_query():
    """
    Returns the specified endpoint for the top games currently
    being played on Twitch.
    """
    return 'games/top'


def get_game_id(get):
    """
    This function must take in the GET request of a game
    that has its game_id
    and returns that game_id as a string.
    """
    request_to_js = json.loads(get.text)
    return request_to_js['data'][0]['id']


def get_games_query(game_name):
    """
    Returns endpoint for the specified game played
    on Twitch.
    """
    result = 'games?name=' + game_name
    return result
