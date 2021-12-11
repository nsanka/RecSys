import re
import base64
import datetime
from numpy.core.arrayprint import format_float_positional
import requests
import spotipy
import json
import time
import random
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from urllib.parse import urlencode
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from scipy.spatial.distance import cdist

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_num_tracks_fig(filename, opt='total', rows=200):
    with open(filename) as log_file:
        playlist_files = []
        new_tracks = []
        req_tracks = []
        for line in log_file:
            line = line.strip()
            if 'File: data/mpd.slice' in line:
                playlist_files.append(line.split('.')[-2])
            if 'Created new track_ids' in line:
                new_tracks.append(int(line.split(':')[1].rstrip()))
            if opt == 'total':
                mode = 'overlay'
                if 'Total tracks/ratings in this file' in line:
                    req_tracks.append(int(line.split(':')[1].rstrip()))
            else:
                mode = 'stack'
                if 'Tracks already exist' in line:
                    req_tracks.append(int(line.split(':')[1].rstrip()))

        req_tracks_df = pd.DataFrame(zip(playlist_files, req_tracks), columns=['files', 'num of tracks'])
        req_tracks_df['tracks'] = opt
        new_tracks_df = pd.DataFrame(zip(playlist_files, new_tracks), columns=['files', 'num of tracks'])
        new_tracks_df['tracks'] = 'new'
        tracks_df = pd.concat([req_tracks_df.iloc[:rows], new_tracks_df.iloc[:rows]])
        
        fig = px.bar(tracks_df,
                    x='files',
                    y='num of tracks',
                    color='tracks',
                    barmode=mode)
        return fig

class SPR_ML_Model():
    def __init__(self, model_path, tsne_path, scaler_path, playlists_path, train_data_scaled_path):
        """
        Inits class with hard coded values for the Spotify instance and gets the paths for all the models and data
        """
        # Model loading
        self.model = pickle.load(open(model_path, 'rb'))
        self.tsne_transformer = pickle.load(open(tsne_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))

        # Data loading
        self.playlists = json.load(open(playlists_path, "r"))
        self.train_scaled_data = np.loadtxt(train_data_scaled_path, delimiter=',')
        self.train_data_scaled_feats_df = pd.DataFrame(self.train_scaled_data)
        self.train_data_scaled_feats_df['cluster'] = pd.Categorical(self.model.labels_)
        
class SpotifyRecommendations():
    """
    This Class will provide music recommendations in a form of Playlists
    Attributes:
        - model_path (str): Path to where the model is saved, should be pretrained.
        - tsne_path (str): Path to where the TSNE transformer is saved, should be pretrained.
        - scaler_path (str): Path to where the Standar Scaler transformer is saved, should be pretrained.
        - playlists_path (str): Path to where the Playlists file is saved, will represent the pool provide recommendations.
        - scaled_data_path (str): Path to where the Scaled Data is saved, all playlists used for training should be present.
    
        This function will compute the most similar or disimilar playlists given a target vector 'y' which represents the mean
        features of the user's favorite songs. Similarity is calculated based on metrics such as Cosine, Manhattan, Euclidean, etc.
        Parameters:
            - model: Trained clustering model.
            - train_data_scaled_feats_df (dataframe): Dataframe with scaled data for all the training data
            - playlists (dictionary): Dictionary with all the playlists from the .json files
            - y (np.array): user's favorite songs scaled vector
            - n (int): top n playlists to retrieve
            - metric (str): metric to use, recommended 'cityblock', 'euclidean', 'cosine'.
            - similar (bool): whether to calculate most similar or most disimilar 
            - printing (bool): whether to print the results or not
        Output:
            - indices (np.array): indices of the top n playlists based on the train_data_scaled_feats_df dataframe

    """
    def __init__(self, playlist_uri=None, sp_user=None, len_of_favs='all_time'):
        """
        Inits class with hard coded values for the Spotify instance and gets the paths for all the models and data
        """
        self.feat_cols_user = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

        self.playlist_uri = playlist_uri
        self.len_of_favs = len_of_favs
        
        if self.playlist_uri is not None:
            self.sp = spotipy.Spotify(client_credentials_manager = SpotifyClientCredentials())
        else:
            # Hardcoded init variables
            # Defining scope to read user playlist and write playlist to user
            #self.scope = 'user-library-read user-follow-read playlist-modify-private playlist-modify'
            self.scope = "user-library-read"
            token = spotipy.util.prompt_for_user_token(sp_user, self.scope)
            self.sp = spotipy.Spotify(auth=token)
            #print(self.sp.me())

    def set_ml_model(self, model, tsne_transformer, scaler, playlists, train_data_scaled_feats_df):
        # Model loading
        self.model = model
        self.tsne_transformer = tsne_transformer
        self.scaler = scaler

        # Data loading
        self.playlists = playlists
        self.train_data_scaled_feats_df = train_data_scaled_feats_df

    def get_current_user_fav_tracks(self):
        "Get all favorite tracks from current user and return them in a dataframe"
        results = self.sp.current_user_saved_tracks()
        tracks = results['items']
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])

        self.fav_songs_df = pd.json_normalize(tracks, record_path=['track', 'artists'], meta=[['added_at'], ['track', 'id'], ['track', 'name']])
        self.fav_songs_df = self.fav_songs_df.drop_duplicates(subset='track.id', keep="first")
        self.fav_songs_df['added_at'] = pd.to_datetime(self.fav_songs_df['added_at'])
        self.fav_songs_df = self.fav_songs_df.sort_values(by='added_at', ascending=True).set_index('added_at')
        self.fav_songs_df = self.fav_songs_df[['name', 'id', 'track.id', 'track.name']]
        self.fav_songs_df.rename(columns={'track.id':'uri', 'track.name': 'song', 'name': 'artist', 'id': 'artist_uri'}, inplace=True)
        #self.artist_uri = self.fav_songs_df['artist_uri'].tolist()
        return self.fav_songs_df

    def get_current_user_audio_features(self):
        "Extract audio features from each track from the user's favorite tracks and return a dataframe"
        try:
            self.fav_songs_df # Checks if it exist else runs the function to get the variable
        except:
            self.get_current_user_fav_tracks()
        
        audio_feats = []
        if self.len_of_favs == 'last_month':
            fav_songs_df = self.fav_songs_df.last('1M').copy()
        elif self.len_of_favs == '6_months':
            fav_songs_df = self.fav_songs_df.last('6M').copy()
        else:
            fav_songs_df = self.fav_songs_df.copy()

        all_uris = fav_songs_df['uri'].tolist()
        chunks_uris = [all_uris[i:i + 100] for i in range(0, len(all_uris), 100)]
        for chunk in  chunks_uris:
            for _ in range(5):
                try:
                    chunk_audio_feats = self.sp.audio_features(chunk)
                    audio_feats.append(chunk_audio_feats)
                except Exception as e: 
                    print(e)
                    print('chunk: {}'.format(chunk))
                else:
                    break
            else:
                print('Everything failed')
        
        audio_feats_df = pd.DataFrame([item for sublist in audio_feats for item in sublist if item])
        self.fav_songs_feats_df = fav_songs_df.merge(audio_feats_df, how='right', left_on="uri", right_on='id')
        return self.fav_songs_feats_df

    def get_user_raw_y(self):
        "Get user 'y' vector without scaling"
        try:
            self.fav_songs_feats_df # Checks if it exist else runs the function to get the variable
        except:
            self.get_current_user_audio_features()

        self.raw_y = self.fav_songs_feats_df[self.feat_cols_user].mean()
        return self.raw_y

    def get_user_scaled_y_vector(self):
        "Get user 'y' vector after scaling in a numpy array with shape of (1,n)"
        try:
            self.raw_y # Checks if it exist else runs the function to get the variable
        except:
            self.get_user_raw_y()
            
        self.scaled_y = self.scaler.transform(np.array(self.raw_y).reshape(1,-1))
        return self.scaled_y

    def get_top_n_playlists(self, n=10, metric='cityblock', similar=True, printing=False):
        """
        This function will compute the most similar or disimilar playlists given a target vector 'y' which represents the mean
        features of the user's favorite songs. Similarity is calculated based on metrics such as Cosine, Manhattan, Euclidean, etc.
        Parameters:
            - model: Trained clustering model.
            - train_data_scaled_feats_df (dataframe): Dataframe with scaled data for all the training data
            - playlists (dictionary): Dictionary with all the playlists from the .json files
            - scaled_y (np.array): user's favorite songs scaled vector
            - n (int): top n playlists to retrieve
            - metric (str): metric to use, recommended 'cityblock', 'euclidean', 'cosine'.
            - similar (bool): whether to calculate most similar or most disimilar 
            - printing (bool): whether to print the results or not
        Output:
            - top_playlists (np.array): indices of the top n playlists based on the train_data_scaled_feats_df dataframe
        
        """
        try:
            self.scaled_y
        except:
            self.get_user_scaled_y_vector()

        # Get labels from model and predict user cluster
        self.user_cluster = self.model.predict(self.scaled_y)
        
        # Slice df for the predicted cluster and get Playlist IDs (PIDs)
        df_slice = self.train_data_scaled_feats_df[self.train_data_scaled_feats_df['cluster']==self.user_cluster[0]]
        df_slice = df_slice.drop(['cluster'], axis=1)
        indices = self.train_data_scaled_feats_df[self.train_data_scaled_feats_df['cluster']==self.user_cluster[0]].reset_index()['index'].to_numpy() # PIDs for the cluster
        
        # Convert df slice to numpy, compute similarities and grab the top n PIDs
        sliced_data_array = df_slice.to_numpy()
        if similar:
            simi = cdist(sliced_data_array, self.scaled_y, metric=metric).argsort(axis=None)[:n]
        else:
            simi = cdist(sliced_data_array, self.scaled_y, metric=metric).argsort(axis=None)[-n:]
        self.top_playlists = indices[simi]
        
        if printing:
            for idx in simi:
                print('Playlist: {}\tpid:{}'.format(self.playlists[idx]['name'], self.playlists[idx]['pid']))
                for song in self.playlists[idx]['tracks'][0:3]:
                    print('Artist: {}\t Song:{}'.format(song['artist_name'], song['track_name']))
                print('\n')
        
        return self.top_playlists

    def get_songs_recommendations(self, n=30, printing=False):
        """
        This function computes the variance, of each song in the given playlists, to the user's favorite songs (scaled_y)
        Parameters:
            - n (int): number of songs to recommend, default to 30.
            - printing (bool): Flag to print or not the song recommendations, default to False.
        """

        try:
            self.top_playlists
        except:
            self.get_top_n_playlists()

        playlist_audio_features = []
        for playlist in self.top_playlists:
            for song in self.playlists[playlist]['tracks']:
                playlist_audio_features.append(self.sp.audio_features(song['track_uri'].split("k:")[1])[0])

        playlist_audio_features_df = pd.DataFrame(playlist_audio_features)
        array_audio_feats = playlist_audio_features_df[self.feat_cols_user].to_numpy()
        
        y_vector = np.array(self.raw_y).reshape(1,-1)
        low_variance_indices = np.sum(np.square((y_vector-array_audio_feats)),axis=1).argsort(axis=None)
        self.song_uris = playlist_audio_features_df.loc[low_variance_indices]['id']
        self.song_uris.drop_duplicates(inplace=True)
        self.song_uris = self.song_uris[:n]

        if printing:
            for uri in self.song_uris:
                print('Song: {}'.format(self.sp.track(uri)['name']))
                print('Artist: {}\n'.format(self.sp.track(uri)['artists'][0]['name']))

        return self.song_uris

    def build_spotify_playlist(self, playlist_name='Machine Learning Playlist', 
                               description='Hell yeah, this is a Machine Learning Playlist generated on {}'.format(datetime.date.today().strftime("%B %d, %Y"))):
        """
        Build and Publish Spotify Playlist
        Parameters:
            - playlist_name (str): Name of playlist.
            - decription (str): Description of playlist.
            - target (str): 'user' or 'playlist', user will use user's favorite tracks and playlist will 
        """
        if self.playlist_uri is None:
            try:
                self.song_uris
            except:
                self.get_songs_recommendations()

            items = self.song_uris.to_list()
        else:
            try:
                self.plylst_song_uris
            except:
                self.get_playlist_songs_recommendations()

            items = self.plylst_song_uris.to_list()
        #user_id = self.sp.current_user()['id']
        #new_playlist = self.sp.user_playlist_create(user_id, playlist_name, description=description)
        #self.sp.playlist_add_items(new_playlist['id'],items=items)
        return items
            
    def get_playlist_tracks(self):
        plylst = self.sp.playlist(self.playlist_uri)
        song = []
        artist = []
        uri = []
        self.artist_uri = []

        for i in plylst['tracks']['items']:
            song.append(i['track']['name'])
            artist.append(i['track']['artists'][0]['name'])
            uri_id = i['track']['uri']
            uri.append(re.search('(^spotify:track:([^\s]+))', uri_id).groups()[1])
            self.artist_uri.append(i['track']['artists'][0]['id'])

        self.plylst_songs_df = pd.DataFrame(
            {'song': song,
            'artist': artist,
            'uri': uri,
            'artist_uri':self.artist_uri
            })
        return self.plylst_songs_df

    def get_playlist_audio_features(self):
        "Extract audio features from each track from the target playlist tracks and return a dataframe"
        try:
            self.plylst_songs_df # Checks if it exist else runs the function to get the variable
        except:
            self.get_playlist_tracks()
        
        appended_df = []
        for i in self.plylst_songs_df['uri']:
            appended_df.append(pd.DataFrame(self.sp.audio_features(i)[0], index=[i,]))
        uri_df = pd.concat(appended_df)
        uri_df = uri_df.drop(['uri'], axis=1)
        uri_df = uri_df.reset_index()
        uri_df = uri_df.rename(columns={"index": "uri"})

        self.plylst_feats_df = pd.merge(self.plylst_songs_df, uri_df, on="uri")
        return self.plylst_feats_df

    def get_playlist_raw_y(self):
        "Get user 'y' vector without scaling"
        try:
            self.plylst_feats_df # Checks if it exist else runs the function to get the variable
        except:
            self.get_playlist_audio_features()

        self.plylst_raw_y = self.plylst_feats_df[self.feat_cols_user].mean()
        return self.plylst_raw_y

    def get_playlist_scaled_y_vector(self):
        "Get user 'y' vector after scaling in a numpy array with shape of (1,n)"
        try:
            self.plylst_raw_y # Checks if it exist else runs the function to get the variable
        except:
            self.get_playlist_raw_y()
            
        self.plylst_scaled_y = self.scaler.transform(np.array(self.plylst_raw_y).reshape(1,-1))
        return self.plylst_scaled_y

    def get_playlist_top_n_playlists(self, n=10, metric='cityblock', similar=True, printing=False):
        """
        This function will compute the most similar or disimilar playlists given a target vector 'y' which represents the mean
        features of the user's favorite songs. Similarity is calculated based on metrics such as Cosine, Manhattan, Euclidean, etc.
        Parameters:
            - model: Trained clustering model.
            - train_data_scaled_feats_df (dataframe): Dataframe with scaled data for all the training data
            - playlists (dictionary): Dictionary with all the playlists from the .json files
            - plylst_scaled_y (np.array): user's favorite songs scaled vector
            - n (int): top n playlists to retrieve
            - metric (str): metric to use, recommended 'cityblock', 'euclidean', 'cosine'.
            - similar (bool): whether to calculate most similar or most disimilar 
            - printing (bool): whether to print the results or not
        Output:
            - plylst_top_playlists (np.array): indices of the top n playlists based on the train_data_scaled_feats_df dataframe
        
        """
        try:
            self.plylst_scaled_y
        except:
            self.get_playlist_scaled_y_vector()


        # Get labels from model and predict user cluster
        self.train_data_scaled_feats_df['cluster'] = pd.Categorical(self.model.labels_)
        self.plylst_cluster = self.model.predict(self.plylst_scaled_y)
        
        # Slice df for the predicted cluster and get Playlist IDs (PIDs)
        df_slice = self.train_data_scaled_feats_df[self.train_data_scaled_feats_df['cluster']==self.plylst_cluster[0]]
        df_slice = df_slice.drop(['cluster'], axis=1)
        indices = self.train_data_scaled_feats_df[self.train_data_scaled_feats_df['cluster']==self.plylst_cluster[0]].reset_index()['index'].to_numpy() # PIDs for the cluster
        
        # Convert df slice to numpy, compute similarities and grab the top n PIDs
        sliced_data_array = df_slice.to_numpy()
        if similar:
            simi = cdist(sliced_data_array, self.plylst_scaled_y, metric=metric).argsort(axis=None)[:n]
        else:
            simi = cdist(sliced_data_array, self.plylst_scaled_y, metric=metric).argsort(axis=None)[-n:]
        self.plylst_top_playlists = indices[simi]
        
        if printing:
            for idx in simi:
                print('Playlist: {}\tpid:{}'.format(self.playlists[idx]['name'], self.playlists[idx]['pid']))
                for song in self.playlists[idx]['tracks'][0:3]:
                    print('Artist: {}\t Song:{}'.format(song['artist_name'], song['track_name']))
                print('\n')
        
        return self.plylst_top_playlists

    def get_playlist_songs_recommendations(self, n=30, printing=False):
        """
        This function computes the variance, of each song in the given playlists, to the user's favorite songs (plylst_scaled_y)
        Parameters:
            - n (int): number of songs to recommend, default to 30.
            - printing (bool): Flag to print or not the song recommendations, default to False.
        """

        try:
            self.plylst_top_playlists
        except:
            self.get_playlist_top_n_playlists()

        playlist_audio_features = []
        for playlist in self.plylst_top_playlists:
            for song in self.playlists[playlist]['tracks']:
                playlist_audio_features.append(self.sp.audio_features(song['track_uri'].split("k:")[1])[0])

        playlist_audio_features_df = pd.DataFrame(playlist_audio_features)
        array_audio_feats = playlist_audio_features_df[self.feat_cols_user].to_numpy()
        
        y_vector = np.array(self.plylst_raw_y).reshape(1,-1)
        low_variance_indices = np.sum(np.square((y_vector-array_audio_feats)),axis=1).argsort(axis=None)
        self.plylst_song_uris = playlist_audio_features_df.loc[low_variance_indices]['id']
        self.plylst_song_uris.drop_duplicates(inplace=True)
        self.plylst_song_uris = self.plylst_song_uris[:n]

        if printing:
            for uri in self.plylst_song_uris:
                print('Song: {}'.format(self.sp.track(uri)['name']))
                print('Artist: {}\n'.format(self.sp.track(uri)['artists'][0]['name']))

        return self.plylst_song_uris

    def get_spotify_wrapped(self):
        "Get Spotify Wrapped for current user"
        try:
            self.artist_uri()
        except:
            if self.playlist_uri is None:
                self.get_current_user_fav_tracks()
            else:
                self.get_playlist_tracks()

        if self.playlist_uri is None:
            user = self.sp.current_user()['display_name']
            followers = self.sp.current_user()['followers']['total']
            print("Hello {}!".format(user))
            print("We are happy that you are using our product. Let's see some of your personal Spotify stats.\n")
            time.sleep(5)
            if followers >= 1:
                print("At this moment you have a total of {} followers, that's not bad at all!\nThey know you have an amazing music taste.\n".format(followers))
            else:
                print("Ouch, at this moment you don't have any followers, let me know if you want me to follow you. I'll be happy to see what type of music you listen to.\n")
            time.sleep(6)

            top_artists = []
            genres = []
            try:
                for artist in self.sp.current_user_top_artists(time_range='long_term')['items']:
                    top_artists.append(artist['name'])
                    genres.append(artist['genres'])
                print("These are your top artist of all time:")
                for i in top_artists[:5]:
                    print(i)
                print("\n")
            except:
                print("Ooops, it seems that you don't have top artist at the moment.\n")

            time.sleep(6)
            top_tracks = []
            try:
                print("And these are your top tracks of all time:")
                for i in self.sp.current_user_top_tracks(time_range='long_term')['items'][:5]:
                    print("{} - {}".format(i['name'], i['artists'][0]['name']))
            except:
                print("Ooops, it seems that you don't have top tracks at the moment.\n")

        genres = []
        for artist in self.artist_uri:
            genres.append(self.sp.artist(artist)['genres'])

        text = [item for sublist in genres for item in sublist]
        text = ' '.join(text)
        text

        sequential =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 
                    'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        color = random.choice(sequential)

        wc = WordCloud(background_color ='white',relative_scaling=0, width=500, height=500, colormap=color).generate(text)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title('These are the genres\nyou listen to the most.\n', size=20)

    def __str__(self):
        return 'Spotify Recommender System with model: {} on {} playlists.'.format(self.model, len(self.train_data_scaled_feats_df))

# Examples on how to use it
# Need to call SpotifyRecommendations with the given paths for the models and data
#x = SpotifyRecommendations(model_path, tsne_path, scaler_path, playlists_path, train_data_scaled_path)

# This will build a playlist based on the current user logged in
#x.build_spotify_playlist()

# This will build a playlist based on a playlist
#x.build_spotify_playlist(playlist='71vjvXmodX7GgWNV7oOb64')

# Fine tune the recommendations
# n: number of similar playlists
# metric: type of metric, you can try 'euclidean', 'cosine', 'cityblock'
# similar: True for similar False for longest distance but still within the same cluster

#x.get_top_n_playlists(n=10, metric='cityblock', similar=True, printing=False) # Fine tune for current user
#x.get_playlist_top_n_playlists(n=10, metric='cityblock', similar=True, printing=False) # Fine tune for playlist

# After tuning, run againn build_spotify_playlist() or  build_spotify_playlist()

# Examples:
#x.build_spotify_playlist(playlist='71vjvXmodX7GgWNV7oOb64') # From a previously generated playlist
#x.build_spotify_playlist(playlist_name = 'On User') # On current user but giving a playlist name
#x.build_spotify_playlist(playlist_name = 'Metal Essentials', playlist='37i9dQZF1DWWOaP4H0w5b0') # Based on a Metal essentials playlist
#x.build_spotify_playlist(playlist_name = 'Classical Essentials', playlist='37i9dQZF1DWWEJlAGA9gs0') # Based on a Classical essentials playlist
#x.build_spotify_playlist() # On current user with default values
#x.get_spotify_wrapped()

class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"

    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        #Returns a base64 encoded string
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()

    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
        "Authorization": f"Basic {client_creds_b64}" 
        }

    def get_token_data(self):
        return {
        "grant_type": "client_credentials"
        }

    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299): 
            raise Exception("Could not authenticate client")
            #return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires 
        self.access_token_did_expire = expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token()
        return token

    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }      
        return headers

    def base_search(self, query_params):
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        print(lookup_url)
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        print(r.json())
        return r.json()

    def search(self, query=None, operator=None, operator_query=None, search_type='playlist'):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)
   
