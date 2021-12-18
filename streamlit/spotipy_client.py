import os
import re
import base64
import datetime
import platform
from numpy.core.arrayprint import format_float_positional
import requests
import spotipy
import time
import random
import pickle
import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import plotly.express as px
from urllib.parse import urlencode
from urllib.request import urlopen
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from scipy.spatial.distance import cdist
import seaborn as sns

from wordcloud import WordCloud
import matplotlib.pyplot as plt

cwd = os.getcwd()

#import sys
#sys.path.insert(1, cwd)
#import config
# Spotify Credentials
#os.environ["SPOTIPY_CLIENT_ID"] = config.SPOTIPY_CLIENT_ID
#os.environ["SPOTIPY_CLIENT_SECRET"] = config.SPOTIPY_CLIENT_SECRET
#os.environ['SPOTIPY_REDIRECT_URI'] = config.SPOTIPY_REDIRECT_URI  # Needed for user authorization
#print("STS", st.secrets["SPOTIPY_CLIENT_ID"])

log_filename = os.path.join(cwd, 'data', 'read_spotify_mpd_log.txt')
feedback_db_file = os.path.join(cwd, 'data', 'user_feedback.db')

# Pickled models
model_path = os.path.join(cwd, 'models', 'KMeans_K17_20000_sample_model.sav')
tsne_path = os.path.join(cwd, 'models', 'openTSNETransformer.sav')
scaler_path = os.path.join(cwd, 'models', 'StdScaler.sav')
playlists_db_path = os.path.join(cwd, 'data', 'spotify_20K_playlists.db')
train_data_scaled_path = os.path.join(cwd, 'data' , 'scaled_data.csv')
openTSNE_path = os.path.join(cwd, 'data' , 'openTSNE_20000.csv')

def get_public_ip():
    try:
        data = str(urlopen('http://checkip.dyndns.com/').read())
        # data = '<html><head><title>Current IP Check</title></head><body>Current IP Address: 65.96.168.198</body></html>\r\n'
        ip = re.compile(r'Address: (\d+\.\d+\.\d+\.\d+)').search(data).group(1)
    except:
        ip = ''
    return ip

def get_num_tracks_fig(opt='total', rows=200):
    with open(log_filename) as log_file:
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

class User_FeedbackDB():
    db_file = None
    conn = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_file = feedback_db_file
        self.create_connection()
        self.create_table()

    def create_connection(self):
        """ create a database connection to the SQLite database specified by db_file
        :return: None
        """
        try:
            self.conn = sqlite3.connect(self.db_file)
        except Error as e:
            print(e)

    def create_table(self):
        """ create a feedback table
        :return: None
        """
        try:
            sql_create_table_feedback = """ CREATE TABLE IF NOT EXISTS feedback (
                                            hostname text NOT NULL,
                                            user_ip text NOT NULL,
                                            feedback text NOT NULL,
                                            rec_type text NOT NULL,
                                            rec_name text NOT NULL,
                                            ml_model_options text,
                                            username text
                                            ); """
            cur = self.conn.cursor()
            cur.execute(sql_create_table_feedback)
        except Error as e:
            print(e)
            print('Failed to create feedback table')

    def check_feedback_exists(self, feedback):
        """
        Query feedback by user_ip, rec_name, ml_model_options, username
        :param feedback_list:
        :return: True/False
        """
        cur = self.conn.cursor()
        cur.execute("SELECT feedback FROM feedback WHERE user_ip=? and rec_name=? and ml_model_options=? and username=?", (feedback[1], feedback[4], feedback[5], feedback[6]))
        rows = cur.fetchall()
        feedback_exists = False
        if len(rows) > 0:
            feedback_exists = True
        return feedback_exists

    def add_user_feedback(self, feedback):
        """
        Add a new feedback
        :param feedback list:
        :return: None
        """
        hostname = platform.node()
        user_ip = get_public_ip()
        data_tuple = tuple([hostname, user_ip] + feedback)
        if self.check_feedback_exists(data_tuple):
            return
        sql = ''' INSERT INTO feedback(hostname, user_ip, feedback, rec_type, rec_name, ml_model_options, username)
                VALUES(?,?,?,?,?,?,?) '''
        try:
            cur = self.conn.cursor()
            cur.execute(sql, data_tuple)
            self.conn.commit()
        except:
            print('Failed to add feedback')

    def get_feedback_plot(self):
        feedback_df = pd.read_sql('select feedback, rec_type from feedback', self.conn)
        fig = None
        if len(feedback_df) > 0:
            feedback_df = feedback_df.value_counts().reset_index().rename(columns={0: 'count'})
            fb_order = {'Love it': 0, 'Like it': 1, 'Okay': 2, 'Hate it': 3}
            feedback_df = feedback_df.sort_values(by='feedback', key=lambda x: x.map(fb_order))
            fig = px.bar(feedback_df,
                        y='count',
                        x='feedback',
                        color='rec_type',
                        barmode='group')
        return fig

    def get_all_feedbacks_df(self):
        all_feedbacks_df = pd.read_sql('select * from feedback', self.conn)
        return all_feedbacks_df

    def add_feedback_df(self, feedback_df):
        feedback_df.to_sql(name='feedback', con=self.conn, if_exists='replace', index=False)

class SPR_ML_Model():
    def __init__(self):
        """
        Inits class with hard coded values for the Spotify instance and gets the paths for all the models and data
        """
        # Model loading
        self.model = pickle.load(open(model_path, 'rb'))
        self.tsne_transformer = pickle.load(open(tsne_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))

        # Data loading
        self.playlists_db = playlists_db_path
        conn = sqlite3.connect(playlists_db_path)
        self.tracks_df = pd.read_sql('select * from tracks', conn)
        self.playlists_df = pd.read_sql('select * from playlists', conn)
        self.playlists_df['cluster'] = pd.Categorical(self.model.labels_)
        self.features_df = pd.read_sql('select * from features', conn)
        self.ratings_df = pd.read_sql('select * from ratings', conn)
        if conn:
            conn.close()
        
        self.train_scaled_data = np.loadtxt(train_data_scaled_path, delimiter=',')
        self.train_data_scaled_feats_df = pd.DataFrame(self.train_scaled_data)
        self.train_data_scaled_feats_df['cluster'] = pd.Categorical(self.model.labels_)
        self.openTSNE_df = pd.read_csv(openTSNE_path)
        self.openTSNE_df['cluster'] = pd.Categorical(self.model.labels_)
        
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
    def __init__(self, playlist_uri=None, sp_user=None):
        """
        Inits class with hard coded values for the Spotify instance and gets the paths for all the models and data
        """
        self.feat_cols_user = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

        self.playlist_uri = playlist_uri
        self.len_of_favs = 'all_time'
        self.log_output = None
        sequential =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 
                    'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        self.color = random.choice(sequential)
        
        if self.playlist_uri is not None:
            self.sp = spotipy.Spotify(client_credentials_manager = SpotifyClientCredentials())
        else:
            # Hardcoded init variables
            # Defining scope to read user playlist and write playlist to user
            #self.scope = 'user-library-read user-follow-read playlist-modify-private playlist-modify'
            self.scope = "user-library-read"
            self.sp_oauth = SpotifyOAuth(scope = self.scope, username=sp_user)
            try:            
                token_info = self.sp_oauth.get_cached_token()
                access_token = token_info['access_token']
                self.sp = spotipy.Spotify(access_token)
                #token = spotipy.util.prompt_for_user_token(sp_user, self.scope)
                #self.sp = spotipy.Spotify(auth=token)
                #self.sp_oauth = SpotifyOAuth(scope = self.scope, requests_session=True, requests_timeout=10, username=sp_user)
                #self.sp = spotipy.Spotify(auth_manager=self.sp_oauth)
            except:
                self.sp = None
            #print(self.sp.me())

    def init_sp(self, response_url):
        code = self.sp_oauth.parse_response_code(response_url)
        #print(self.sp_oauth.get_access_token(code))
        token_info = self.sp_oauth.get_access_token(code, check_cache=False)
        access_token = token_info['access_token']
        self.sp = spotipy.Spotify(access_token)

    def get_html_for_login(self):
        auth_url = self.sp_oauth.get_authorize_url()
        return auth_url

    def set_ml_model(self, ml_model):
        # Model loading
        self.model = ml_model.model
        self.tsne_transformer = ml_model.tsne_transformer
        self.scaler = ml_model.scaler

        # Data loading
        self.tracks_df = ml_model.tracks_df
        self.playlists_df = ml_model.playlists_df
        self.features_df = ml_model.features_df
        self.ratings_df = ml_model.ratings_df
        self.train_data_scaled_feats_df = ml_model.train_data_scaled_feats_df
        self.openTSNE_df = ml_model.openTSNE_df

    def get_audio_features_df(self, track_uris_list=None, playlist_pids_list=None):
        self.log_output('Getting Audio features for the tracks')
        # Get all track_uri for playlists
        if playlist_pids_list is not None:
            self.log_output('Getting audio features for tracks in Top Playlists in the Cluster\n' + ','.join([str(pid) for pid in playlist_pids_list]))
            ratings_df = self.ratings_df[self.ratings_df['pid'].isin(playlist_pids_list)].copy()
            tracks_df = self.tracks_df[self.tracks_df['track_id'].isin(ratings_df['track_id'].values)].copy()
            track_uris_list = tracks_df['track_uri'].values
            self.log_output('Tracks in this list: ' + str(len(track_uris_list)))
        
        self.log_output('Unique tracks in this list: ' + str(len(set(track_uris_list))))
        # Find audio features if track_uri is already in the database:
        tracks_df = self.tracks_df[self.tracks_df['track_uri'].isin(set(track_uris_list))][['track_id', 'track_uri']].copy()
        exist_audio_feats_df = self.features_df[self.features_df['track_id'].isin(tracks_df['track_id'].values)].copy()
        exist_audio_feats_df = exist_audio_feats_df.merge(tracks_df, on='track_id')
        exist_audio_feats_df.drop(columns='track_id', inplace=True)
        exist_audio_feats_df.rename(columns={'track_uri':'uri'}, inplace=True)
        if len(exist_audio_feats_df) == len(set(track_uris_list)):
            self.log_output('Got all audio features from database for tracks: ' + str(len(exist_audio_feats_df)))
            return exist_audio_feats_df
        
        track_uris_list = list(set(track_uris_list) - set(tracks_df['track_uri'].tolist()))

        # Extract audio features from Spotify
        audio_feats = []
        chunks_uris = [track_uris_list[i:i + 100] for i in range(0, len(track_uris_list), 100)]
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
        track_uris_list = audio_feats_df['id'].tolist()
        audio_feats_df = audio_feats_df[self.feat_cols_user]
        audio_feats_df['uri'] = track_uris_list
        #audio_feats_df.insert(column='uri', value=track_uris_list)
        self.log_output('Extracted audio features from Spotify: ' + str(len(audio_feats_df)))
        if len(exist_audio_feats_df) > 0:
            self.log_output('Got some audio features from database for tracks: ' + str(len(exist_audio_feats_df)))
            audio_feats_df = pd.concat([exist_audio_feats_df, audio_feats_df])
        return audio_feats_df

    def get_tracks_from_playlist_or_user_favorites(self):
        if self.playlist_uri:
            self.log_output('---\nGetting all tracks for Playlist')
            # Get all tracks in the playlist
            results = self.sp.playlist(self.playlist_uri)['tracks']
            tracks = results['items']
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
        else:
            self.log_output('Getting all tracks for User Favorites')
            "Get all favorite tracks from current user and return them in a dataframe"
            results = self.sp.current_user_saved_tracks()
            tracks = results['items']
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])

        songs_df = pd.json_normalize(tracks, record_path=['track', 'artists'], meta=[['added_at'], ['track', 'id'], ['track', 'name']])
        songs_df = songs_df.drop_duplicates(subset='track.id', keep="first")
        songs_df['added_at'] = pd.to_datetime(songs_df['added_at'])
        songs_df = songs_df.sort_values(by='added_at', ascending=True).set_index('added_at')
        songs_df = songs_df[['name', 'id', 'track.id', 'track.name']]
        songs_df.rename(columns={'track.id':'uri', 'track.name': 'song', 'name': 'artist', 'id': 'artist_uri'}, inplace=True)
        self.artist_uri = songs_df['artist_uri'].tolist()
        self.log_output('Found unique tracks: ' + str(len(songs_df)))
        return songs_df

    def get_tracks_audio_features(self): 
        "Extract audio features from each track from the user's favorite tracks and return a dataframe"
        songs_df = self.get_tracks_from_playlist_or_user_favorites()
        if self.len_of_favs == 'last_month':
            songs_df = songs_df.last('1M')
        elif self.len_of_favs == '6_months':
            songs_df = songs_df.last('6M')
        else:
            pass

        track_uris = songs_df['uri'].tolist()
        audio_feats_df = self.get_audio_features_df(track_uris_list=track_uris)
        self.songs_feats_df = songs_df.merge(audio_feats_df, how='right', on="uri")

    def get_raw_y(self):
        "Get user 'y' vector without scaling"
        try:
            self.songs_feats_df
        except:
            self.get_tracks_audio_features()

        self.raw_y = self.songs_feats_df[self.feat_cols_user].mean()
        return self.raw_y

    def get_scaled_y_vector(self):
        "Get user 'y' vector after scaling in a numpy array with shape of (1,n)"
        try:
            self.raw_y # Checks if it exist else runs the function to get the variable
        except:
            self.get_raw_y()
            
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
            self.get_scaled_y_vector()

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
            for idx in self.top_playlists:
                self.log_output('---')
                self.log_output('Playlist: {}\tpid:{}'.format(self.playlists_df[self.playlists_df['pid'] == idx]['name'].iloc[0], idx))
                ratings_df = self.ratings_df[self.ratings_df['pid'] == idx].copy()
                tracks_df = self.tracks_df[self.tracks_df['track_id'].isin(ratings_df['track_id'].values)].copy()
                for _, song in tracks_df.iloc[0:3].iterrows():
                    self.log_output('Artist: {}\t Song:{}'.format(song['artist_name'], song['track_name']))
            self.log_output('---')
        
        return self.top_playlists

    def get_songs_recommendations(self, n=30, printing=False):
        """
        This function computes the variance, of each song in the given playlists, to the user's favorite songs (raw_y)
        Parameters:
            - n (int): number of songs to recommend, default to 30.
            - printing (bool): Flag to print or not the song recommendations, default to False.
        """
        try:
            self.top_playlists
        except:
            self.get_top_n_playlists(printing=True)

        playlist_audio_features_df = self.get_audio_features_df(playlist_pids_list=self.top_playlists)
        array_audio_feats = playlist_audio_features_df[self.feat_cols_user].to_numpy()
        
        y_vector = np.array(self.raw_y).reshape(1,-1)
        low_variance_indices = np.sum(np.square((y_vector-array_audio_feats)),axis=1).argsort(axis=None)
        self.song_uris = playlist_audio_features_df.loc[low_variance_indices]['uri']
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
        try:
            self.song_uris
        except:
            self.get_songs_recommendations()

        items = self.song_uris.to_list()
        #user_id = self.sp.current_user()['id']
        #new_playlist = self.sp.user_playlist_create(user_id, playlist_name, description=description)
        #self.sp.playlist_add_items(new_playlist['id'],items=items)
        return items

    def get_genre_wordcloud_fig(self):
        "Get Spotify Wrapped for current user"
        try:
            self.artist_uri
        except:
            self.get_tracks_from_playlist_or_user_favorites()

        if self.playlist_uri is None:
            user = self.sp.current_user()['display_name']
            followers = self.sp.current_user()['followers']['total']
            self.log_output("Hello {}!".format(user))
            self.log_output("We are happy that you are using our product. Let's see some of your personal Spotify stats.\n")

            if followers >= 1:
                self.log_output("At this moment you have a total of {} followers, that's not bad at all!\nThey know you have an amazing music taste.\n".format(followers))
            else:
                self.log_output("Ouch, at this moment you don't have any followers, let me know if you want me to follow you. I'll be happy to see what type of music you listen to.\n")

            top_artists = []
            genres = []
            try:
                for artist in self.sp.current_user_top_artists(time_range='long_term')['items']:
                    top_artists.append(artist['name'])
                    genres.append(artist['genres'])
                self.log_output("These are your top artist of all time:")
                for i in top_artists[:5]:
                    self.log_output(i)
                self.log_output("\n")
            except:
                self.log_output("Ooops, it seems that you don't have top artists at the moment.\n")

            try:
                self.log_output("And these are your top tracks of all time:")
                for i in self.sp.current_user_top_tracks(time_range='long_term')['items'][:5]:
                    self.log_output("{} - {}".format(i['name'], i['artists'][0]['name']))
            except:
                self.log_output("Ooops, it seems that you don't have top tracks at the moment.\n")

        genres = []
        for artist in self.artist_uri:
            genres.append(self.sp.artist(artist)['genres'])

        text = [item for sublist in genres for item in sublist]
        text = ' '.join(text)
        wc = WordCloud(background_color ='white',relative_scaling=0, width=500, height=500, colormap=self.color).generate(text)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.title.set_text('Genres you listen to the most\n')
        return fig

    def get_playlist_wordcloud_fig(self):        
        # User Playlist Cluster
        text = ' '.join(self.playlists_df[self.playlists_df['cluster']==self.user_cluster[0]]["name"])
        wc = WordCloud(background_color ='white',relative_scaling=0, width=500, height=500, colormap=self.color).generate(text)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.title.set_text('Playlist names in your cluster {}\n'.format(self.user_cluster))
        return fig

    def get_user_cluster_all_fig(self):
        # Transform user fav songs to TSNE to plot in vector space
        try:
            self.scaled_y
        except:
            self.get_scaled_y_vector()
        try:
            self.user_cluster
        except:
            self.user_cluster = self.model.predict(self.scaled_y)
            
        user_tsne = self.tsne_transformer.transform(self.scaled_y)[0]

        # Blob all clusters
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.scatterplot(ax=ax, x='X', y='Y', hue='cluster', style='cluster', data=self.openTSNE_df, legend=None)
        ax.scatter(x=user_tsne[0], y=user_tsne[1], color='yellow', marker='*', s=500)
        ax.title.set_text('You (Star) are here in the 17 Clusters')
        return fig
        #plt.show()

    def get_user_cluster_single_fig(self):
        try:
            self.scaled_y
        except:
            self.get_scaled_y_vector()
        try:
            self.user_cluster
        except:
            self.user_cluster = self.model.predict(self.scaled_y)
            
        user_tsne = self.tsne_transformer.transform(self.scaled_y)[0]

        # Blob user cluster
        palette = {c:'purple' if c==self.user_cluster else 'darkgrey' for c in self.openTSNE_df.cluster.unique()}
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.scatterplot(ax=ax, x='X', y='Y', hue='cluster', style='cluster', data=self.openTSNE_df, legend=None, palette=palette)
        ax.scatter(x=user_tsne[0], y=user_tsne[1], color='yellow', marker='*', s=500)
        ax.title.set_text('You are in cluster {}'.format(self.user_cluster))
        return fig
        #plt.show()

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
# After tuning, run againn build_spotify_playlist() or  build_spotify_playlist()

# Examples:
#x.build_spotify_playlist(playlist='71vjvXmodX7GgWNV7oOb64') # From a previously generated playlist
#x.build_spotify_playlist(playlist_name = 'On User') # On current user but giving a playlist name
#x.build_spotify_playlist(playlist_name = 'Metal Essentials', playlist='37i9dQZF1DWWOaP4H0w5b0') # Based on a Metal essentials playlist
#x.build_spotify_playlist(playlist_name = 'Classical Essentials', playlist='37i9dQZF1DWWEJlAGA9gs0') # Based on a Classical essentials playlist
#x.build_spotify_playlist() # On current user with default values
#x.get_genre_wordcloud_fig()

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
        #print(headers)
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        #print(lookup_url)
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        #print(r.json())
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
        #print(query_params)
        return self.base_search(query_params)
   
