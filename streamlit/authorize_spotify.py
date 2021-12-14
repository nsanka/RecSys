from bottle import route, run, request
import spotipy
from spotipy import oauth2

import os, sys
cwd = os.getcwd()
sys.path.insert(1, cwd)
import config

PORT_NUMBER = 8080
SPOTIPY_CLIENT_ID = config.SPOTIPY_CLIENT_ID
SPOTIPY_CLIENT_SECRET = config.SPOTIPY_CLIENT_SECRET
SPOTIPY_REDIRECT_URI = config.SPOTIPY_REDIRECT_URI
SCOPE = 'user-library-read'
CACHE = '.spotipyoauthcache'

sp_oauth = oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, scope=SCOPE, cache_path=CACHE)

@route('/')
def index():
    access_token = ""
    token_info = sp_oauth.get_cached_token()
    if token_info:
        print("Found cached token!")
        access_token = token_info['access_token']
    else:
        url = request.url
        code = sp_oauth.parse_response_code(url)
        if 'localhost' not in code:
            print("Found Spotify auth code in Request URL! Trying to get valid access token...")
            #token_info = sp_oauth.get_access_token(code, as_dict=True)
            #access_token = token_info['access_token']
            #print(token_info)
            access_token = sp_oauth.get_access_token(code, as_dict=False)

    if access_token:
        print("Access token available! Trying to get user information...")
        sp = spotipy.Spotify(access_token)
        results = sp.current_user()
        return results
    else:
        return htmlForLoginButton()

def htmlForLoginButton():
    auth_url = getSPOauthURI()
    htmlLoginButton = "<a href='" + auth_url + "'target='_blank'>Login to Spotify</a>"
    return htmlLoginButton

def getSPOauthURI():
    auth_url = sp_oauth.get_authorize_url()
    return auth_url

print(sp_oauth.get_authorize_url())
run(host='', port=8080)